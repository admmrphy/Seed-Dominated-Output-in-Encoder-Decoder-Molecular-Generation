"""
train.py

Unified training template for the Colab runs, organised into the four families
you described. Every family routes through protocol.train_condition, so all of
them inherit the same fixed split, the same per-condition training seeds, and
(on the generation side) the same seeds, temperatures, and molecule counts.

Families
--------
pretrained         : NOT trained. It is the released checkpoint used as a
                     generation baseline only. See load_pretrained_for_generation().

full fine-tuning   : standard, weighted, weighted_dedup.
                     Identical training code; the ONLY difference is the corpus,
                     which corpus.build_corpus_for_condition() selects. This is
                     where the weighted duplicate ablation lives: weighted uses
                     the implicit upweighting-by-duplication, weighted_dedup uses
                     the same chemistry with duplicates removed.

prefix tuning (MPT): mpt_standard, mpt_weighted.
                     Same corpus split as the full-FT pair, but the base model is
                     frozen and only prefix vectors train. merge_and_unload() is
                     never called (it is a LoRA operation and silently returns the
                     unmodified base model for a prefix-tuning adapter).

feedback / ranking : feedback_composite, feedback_qed.
                     A second stage on top of a fine-tuned checkpoint, adding a
                     BRIO-style rank loss. Because the ranking objective is
                     sensitive to its hyperparameters, this family ships with a
                     small controlled grid (learning rate, ranking weight,
                     candidates per seed, generation temperature) and a monitor
                     that tracks validity, uniqueness, novelty, scaffold and
                     internal diversity, and the QED / Fsp3 / LogP / MW /
                     solubility distributions, not the loss alone.

The genuinely reusable logic (corpus selection, seed handling, the masking
collator, the rank loss, the scoring functions, the monitoring metrics, the
grid, and the dispatch) is implemented here. The two points that depend on the
exact dataset and model objects in your notebooks are marked with `HOOK:` and
kept small, so you can drop your existing code straight in.

Colab usage (paste into a thin notebook cell):

    !pip install -q -r requirements-colab.txt
    from train import mount_drive, train_family
    mount_drive()
    ckpts = train_family("weighted")          # trains both seed runs
    ckpts = train_family("weighted_dedup")    # the ablation
    # feedback: search first, then train the two runs with the chosen config
    from train import run_feedback_grid, train_feedback
    results = run_feedback_grid("feedback_composite", base_checkpoint=ckpts_standard[0])
    train_feedback("feedback_composite", chosen_hp=results.best, base_checkpoints=ckpts_standard)

Written for Python 3.9 (Optional[...], no X | Y unions). Heavy imports (torch,
transformers, peft, rdkit) are done inside functions so this module imports on
the CPU-only local machine as well.
"""

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple
import os

from seed_config import (
    DEFAULT_N_GENERATION_SEEDS,
    DEFAULT_TRAIN_PHASE,
    get_data_seed,
    get_num_runs,
)
import protocol
import corpus
from data_split import DEFAULT_SPLITS_PATH


# ---------------------------------------------------------------------------
# SHARED CONSTANTS
# ---------------------------------------------------------------------------
BASE_MODEL: str = "zjunlp/MolGen-large"  # Fang et al., ICLR 2024

DEFAULT_PROJECT_ROOT: str = "/content/drive/MyDrive/MOSESproject"
DEFAULT_CSV: str = "moses_full_properties_with_selfies.csv"

# Training hyperparameters shared by the full-FT and MPT families, matching the
# study's methods section so the two adaptation regimes differ only in whether
# the base weights are updated, not in schedule or optimiser.
# These match the values in the existing training script. NOTE: the paper draft
# currently states 2 epochs and 50 prefix tokens, but the trained models used 3
# epochs and 100 prefix tokens. Reconcile the manuscript with the values below
# (or retrain to the manuscript's values) so the two agree.
EPOCHS: int = 3
PER_DEVICE_BATCH: int = 16
GRAD_ACCUM: int = 2                 # effective batch size 32
LEARNING_RATE: float = 2e-5
LR_SCHEDULER: str = "cosine"
WARMUP_RATIO: float = 0.05
MAX_LEN: int = 128
FP16: bool = True

# Random masking of 15% of non-special encoder tokens. This is REQUIRED: without
# it, MolGen generation collapses and every output matches the seed regardless of
# temperature. It is applied by the denoising collator below.
MASK_PROB: float = 0.15

# Prefix tuning. The existing MPT script uses 100 virtual tokens (the paper
# draft says 50; see the EPOCHS note above).
PREFIX_VIRTUAL_TOKENS: int = 100

# Sampling settings held constant across every condition's generation, matching
# the existing scripts, so between-condition differences cannot be attributed to
# different sampling. Keep the generation hooks using these too.
GENERATION_TOP_K: int = 50
GENERATION_TOP_P: float = 0.95

# Post-training validation monitoring.
VALIDATION_SAMPLE_SIZE: int = 200
VALIDATION_TEMPERATURE: float = 1.0
VALIDATION_SEED_POOL: int = 100  # how many validation-split molecules to seed from
VALIDATION_GEN_SEED: int = 99    # fixed, so validation differences reflect the model, not RNG

FAMILY_OF: Dict[str, str] = {
    "pretrained": "pretrained",
    "standard": "full_ft",
    "weighted": "full_ft",
    "weighted_dedup": "full_ft",
    "mpt_standard": "mpt",
    "mpt_weighted": "mpt",
    "feedback_composite": "feedback",
    "feedback_qed": "feedback",
}


# ---------------------------------------------------------------------------
# COLAB / DRIVE HELPERS
# ---------------------------------------------------------------------------
def mount_drive() -> None:
    """Mount Google Drive at /content/drive (no-op if already mounted)."""
    from google.colab import drive  # type: ignore

    drive.mount("/content/drive")


def project_path(*parts: str, root: str = DEFAULT_PROJECT_ROOT) -> str:
    return os.path.join(root, *parts)


def checkpoint_dir(condition: str, run_index: int, root: str = DEFAULT_PROJECT_ROOT) -> str:
    """One consistent checkpoint location per (condition, run_index)."""
    return project_path("checkpoints", f"{condition}_run{run_index}", root=root)


# ---------------------------------------------------------------------------
# BASE MODEL + MASKING COLLATOR
# ---------------------------------------------------------------------------
def load_base_model_and_tokenizer():
    """Load MolGen-large and its tokenizer. Imports transformers lazily."""
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForSeq2SeqLM.from_pretrained(BASE_MODEL)
    return model, tokenizer


def make_denoising_collator(tokenizer, mask_prob: float = MASK_PROB):
    """Return a collator that masks `mask_prob` of non-special encoder tokens.

    The decoder target is the clean sequence, so the objective is SELFIES
    reconstruction from a corrupted input, matching MolGen's pre-training. The
    masking is the piece that must not be omitted.
    """
    import torch

    special_ids = set(tokenizer.all_special_ids)
    mask_id = tokenizer.mask_token_id
    if mask_id is None:
        raise ValueError(
            "Tokenizer has no mask token; MolGen's denoising objective needs one."
        )

    def collate(features: List[Dict[str, List[int]]]) -> Dict[str, "torch.Tensor"]:
        pad_id = tokenizer.pad_token_id
        input_ids = [list(f["input_ids"]) for f in features]
        max_len = max(len(x) for x in input_ids)

        clean = torch.full((len(input_ids), max_len), pad_id, dtype=torch.long)
        for i, ids in enumerate(input_ids):
            clean[i, : len(ids)] = torch.tensor(ids, dtype=torch.long)

        attn = (clean != pad_id).long()
        labels = clean.clone()
        labels[clean == pad_id] = -100

        # Vectorised masking: mask mask_prob of tokens that are neither padding
        # nor special tokens. Done with tensor ops rather than a per-token Python
        # loop so it is fast enough for a 100k-molecule corpus over several epochs.
        is_special = torch.zeros_like(clean, dtype=torch.bool)
        for sid in special_ids:
            is_special |= clean == sid
        maskable = (clean != pad_id) & (~is_special)
        do_mask = maskable & (torch.rand(clean.shape) < mask_prob)
        corrupted = clean.clone()
        corrupted[do_mask] = mask_id

        return {
            "input_ids": corrupted,
            "attention_mask": attn,
            "labels": labels,
        }

    return collate


def tokenize_selfies(tokenizer, selfies_list: Sequence[str]) -> List[Dict[str, List[int]]]:
    """Tokenize a list of SELFIES strings into input_ids feature dicts."""
    out = []
    for s in selfies_list:
        enc = tokenizer(s, truncation=True, max_length=MAX_LEN)
        out.append({"input_ids": enc["input_ids"]})
    return out


def build_training_arguments(output_dir: str, seed: int):
    """Shared Seq2SeqTrainingArguments for full-FT and MPT. Imports lazily.

    output_dir should be a LOCAL scratch path (for example under /content), not a
    Google Drive path: writing intermediate checkpoints straight to Drive during
    training is slow and prone to stalls. The final model is copied to Drive after
    training. evaluation is left off here (monitoring happens after training via
    validate_after_training), which also avoids the evaluation_strategy argument
    name change between transformers versions.

    seed MUST be the run's training seed. The Trainer re-seeds itself from
    args.seed at the start of train(), so if this is left at the default (42) every
    run trains with identical shuffling, dropout and mask noise and produces
    identical weights, regardless of any earlier set_seed call. Passing the run
    seed here is what makes the runs genuinely independent. data_seed is set to the
    same value so the data sampler varies per run too.
    """
    from transformers import Seq2SeqTrainingArguments

    return Seq2SeqTrainingArguments(
        output_dir=output_dir,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=PER_DEVICE_BATCH,
        per_device_eval_batch_size=PER_DEVICE_BATCH,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LEARNING_RATE,
        lr_scheduler_type=LR_SCHEDULER,
        warmup_ratio=WARMUP_RATIO,
        fp16=FP16,
        logging_steps=100,
        save_strategy="epoch",
        save_total_limit=1,
        report_to="none",
        seed=seed,
        data_seed=seed,
    )


def local_scratch_dir(condition: str, run_index: int) -> str:
    """Local (non-Drive) directory for a trainer's intermediate checkpoints."""
    return f"/content/ckpt_{condition}_run{run_index}"


# ---------------------------------------------------------------------------
# GENERATION + POST-TRAINING VALIDATION MONITORING
# ---------------------------------------------------------------------------
def selfies_to_canonical_smiles(raw_selfies: str) -> Optional[str]:
    """Decode a SELFIES string to canonical SMILES, or None if it fails."""
    import selfies as sf
    from rdkit import Chem

    try:
        smi = sf.decoder(raw_selfies)
    except Exception:
        return None
    if not smi:
        return None
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True)


def _apply_mask_noise(input_ids, tokenizer, mask_prob: float = MASK_PROB):
    """Mask mask_prob of non-special tokens in a batch, for generation seeding.

    This mirrors the masking used at training time. It is required: without it,
    generation collapses to copying the seed regardless of temperature. Vectorised
    (no per-token Python loop) so it is not a bottleneck during generation.
    """
    import torch

    noised = input_ids.clone()
    is_special = torch.zeros_like(noised, dtype=torch.bool)
    for sid in set(tokenizer.all_special_ids):
        is_special |= noised == sid
    do_mask = (~is_special) & (torch.rand(noised.shape) < mask_prob)
    noised[do_mask] = tokenizer.mask_token_id
    return noised


def generate_molecules(
    model,
    tokenizer,
    seed_selfies: Sequence[str],
    n: int,
    temperature: float,
    device: Optional[str] = None,
    batch_size: int = 16,
    n_return: int = 4,
    max_attempts_multiplier: int = 50,
    seed: Optional[int] = None,
) -> List[str]:
    """Generate n unique valid canonical SMILES from the model.

    Faithful to the project's proven generation loop: seed molecules are masked,
    decoded with sampling (temperature, top_k, top_p), converted to canonical
    SMILES, and de-duplicated until n are collected or the attempt budget runs
    out. This is the concrete implementation the protocol generation hook and the
    validation step below both use.

    seed, when given, seeds Python's random module and torch before generation, so
    the output is reproducible and the protocol's per-cell generation seed actually
    takes effect. Leaving it None keeps whatever global RNG state exists. Pass the
    generation seed here from any generate_cell_fn wrapper you build for the
    protocol, otherwise the five generation seeds have no effect on the output.
    """
    import random
    import torch

    if seed is not None:
        random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    device = device or ("cuda" if _cuda_available() else "cpu")
    model = model.to(device)
    model.eval()

    seen: set = set()
    out: List[str] = []
    attempts = 0
    max_attempts = n * max_attempts_multiplier

    while len(out) < n and attempts < max_attempts:
        batch = random.choices(list(seed_selfies), k=batch_size)
        inputs = tokenizer(batch, return_tensors="pt", padding=True,
                           truncation=True, max_length=MAX_LEN)
        noised = _apply_mask_noise(inputs["input_ids"], tokenizer).to(device)
        attn = inputs["attention_mask"].to(device)
        attempts += batch_size * n_return

        with torch.no_grad():
            outputs = model.generate(
                input_ids=noised,
                attention_mask=attn,
                max_length=MAX_LEN,
                do_sample=True,
                temperature=temperature,
                top_k=GENERATION_TOP_K,
                top_p=GENERATION_TOP_P,
                num_return_sequences=n_return,
            )

        for row in outputs:
            if len(out) >= n:
                break
            raw = tokenizer.decode(row, skip_special_tokens=True,
                                   clean_up_tokenization_spaces=False).replace(" ", "")
            canonical = selfies_to_canonical_smiles(raw)
            if canonical is None or canonical in seen:
                continue
            seen.add(canonical)
            out.append(canonical)

    return out


def _cuda_available() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


def validate_after_training(
    model,
    tokenizer,
    condition: str,
    run_index: int,
    csv_path: str,
    root: str = DEFAULT_PROJECT_ROOT,
    splits_path: str = DEFAULT_SPLITS_PATH,
    n: int = VALIDATION_SAMPLE_SIZE,
) -> Dict[str, float]:
    """Generate a small validation sample and compute the full monitoring panel.

    Runs for every trained model (full-FT and MPT), not just feedback, so each
    condition reports validity, uniqueness, novelty, scaffold diversity, internal
    diversity, and the QED / Fsp3 / LogP / MW / solubility means, rather than the
    loss alone. Seeds come from the held-out validation split; novelty is scored
    against the training-pool SMILES.
    """
    import json

    from data_split import get_validation_frame, get_training_pool_frame, resolve_selfies_column, resolve_smiles_column

    val_df = get_validation_frame(csv_path, splits_path)
    selfies_col = resolve_selfies_column(val_df)
    seed_selfies = val_df[selfies_col].dropna().sample(
        n=min(VALIDATION_SEED_POOL, len(val_df)), random_state=99
    ).tolist()

    smiles = generate_molecules(model, tokenizer, seed_selfies, n, VALIDATION_TEMPERATURE,
                                seed=VALIDATION_GEN_SEED)

    pool_df = get_training_pool_frame(csv_path, splits_path)
    pool_smiles = set(pool_df[resolve_smiles_column(pool_df)].astype(str).tolist())

    metrics = evaluate_generation_sample(smiles, reference_smiles=pool_smiles)

    out_dir = checkpoint_dir(condition, run_index, root)
    os.makedirs(out_dir, exist_ok=True)
    import pandas as pd
    pd.DataFrame({"smiles": smiles}).to_csv(os.path.join(out_dir, "validation_sample.csv"), index=False)
    with open(os.path.join(out_dir, "validation_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"[{condition} run{run_index}] validation metrics:")
    for k in ("validity", "uniqueness", "novelty", "scaffold_diversity", "internal_diversity",
              "qed_mean", "fsp3_mean", "logp_mean", "mw_mean", "esol_mean"):
        if k in metrics:
            print(f"    {k:<20} {metrics[k]:.4f}")
    return metrics


# ---------------------------------------------------------------------------
# FULL FINE-TUNING FAMILY  (standard, weighted, weighted_dedup)
# ---------------------------------------------------------------------------
def make_full_ft_train_fn(
    csv_path: str,
    root: str = DEFAULT_PROJECT_ROOT,
    splits_path: str = DEFAULT_SPLITS_PATH,
) -> protocol.TrainRunFn:
    """Build the train_run_fn for the full fine-tuning family."""

    def train_run_fn(condition: str, run_index: int, train_seed: int, data_seed: int) -> str:
        from transformers import Seq2SeqTrainer
        from datasets import Dataset

        model, tokenizer = load_base_model_and_tokenizer()

        train_corpus = corpus.build_corpus_for_condition(condition, csv_path, data_seed, splits_path)
        print(f"[{condition} run{run_index}] corpus: {corpus.duplicate_report(train_corpus)}")

        features = tokenize_selfies(tokenizer, train_corpus["selfies"].tolist())
        train_ds = Dataset.from_list(features)
        collator = make_denoising_collator(tokenizer)

        out_dir = checkpoint_dir(condition, run_index, root)          # Drive: final model
        scratch = local_scratch_dir(condition, run_index)             # local: intermediate
        trainer = Seq2SeqTrainer(
            model=model,
            args=build_training_arguments(scratch, train_seed),
            train_dataset=train_ds,
            data_collator=collator,
        )
        trainer.train()
        os.makedirs(out_dir, exist_ok=True)
        trainer.save_model(out_dir)
        tokenizer.save_pretrained(out_dir)

        # Monitor the full metric panel, not just the loss.
        try:
            validate_after_training(model, tokenizer, condition, run_index, csv_path, root, splits_path)
        except Exception as e:  # never lose a trained checkpoint to a monitoring hiccup
            print(f"[{condition} run{run_index}] validation monitoring skipped: {e}")
        return out_dir

    return train_run_fn


# ---------------------------------------------------------------------------
# PREFIX TUNING FAMILY  (mpt_standard, mpt_weighted)
# ---------------------------------------------------------------------------
def make_mpt_train_fn(
    csv_path: str,
    root: str = DEFAULT_PROJECT_ROOT,
    splits_path: str = DEFAULT_SPLITS_PATH,
) -> protocol.TrainRunFn:
    """Build the train_run_fn for the prefix-tuning family."""

    def train_run_fn(condition: str, run_index: int, train_seed: int, data_seed: int) -> str:
        from transformers import Seq2SeqTrainer
        from datasets import Dataset
        from peft import PrefixTuningConfig, TaskType, get_peft_model

        model, tokenizer = load_base_model_and_tokenizer()

        peft_config = PrefixTuningConfig(
            task_type=TaskType.SEQ_2_SEQ_LM,
            num_virtual_tokens=PREFIX_VIRTUAL_TOKENS,
        )
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()  # expect roughly 0.66% trainable

        train_corpus = corpus.build_corpus_for_condition(condition, csv_path, data_seed, splits_path)
        print(f"[{condition} run{run_index}] corpus: {corpus.duplicate_report(train_corpus)}")

        features = tokenize_selfies(tokenizer, train_corpus["selfies"].tolist())
        train_ds = Dataset.from_list(features)
        collator = make_denoising_collator(tokenizer)

        out_dir = checkpoint_dir(condition, run_index, root)          # Drive: final adapter
        scratch = local_scratch_dir(condition, run_index)             # local: intermediate
        trainer = Seq2SeqTrainer(
            model=model,
            args=build_training_arguments(scratch, train_seed),
            train_dataset=train_ds,
            data_collator=collator,
        )
        trainer.train()
        os.makedirs(out_dir, exist_ok=True)
        # Save the ADAPTER only. Do not call merge_and_unload(): it is a LoRA
        # operation and returns the unmodified base for a prefix-tuning adapter.
        model.save_pretrained(out_dir)
        tokenizer.save_pretrained(out_dir)

        # Monitor the full metric panel, not just the loss.
        try:
            validate_after_training(model, tokenizer, condition, run_index, csv_path, root, splits_path)
        except Exception as e:  # never lose a trained adapter to a monitoring hiccup
            print(f"[{condition} run{run_index}] validation monitoring skipped: {e}")
        return out_dir

    return train_run_fn


# ---------------------------------------------------------------------------
# FEEDBACK / RANKING FAMILY  (feedback_composite, feedback_qed)
# ---------------------------------------------------------------------------
@dataclass
class FeedbackHP:
    """One point in the feedback hyperparameter grid."""

    learning_rate: float
    rank_weight_alpha: float       # weight of the rank loss in L = Lce + alpha*Lrank
    candidates_per_seed: int       # candidate molecules generated per seed molecule
    generation_temperature: float  # temperature used when sampling candidates
    margin_gamma: float = 0.01     # base margin in the pairwise rank loss


# Feedback hyperparameter grid ranges (as advised). The search is staged rather
# than run as a full Cartesian product, which would be expensive.
FEEDBACK_LEARNING_RATES: List[float] = [1e-5, 3e-5, 1e-4]
FEEDBACK_RANK_WEIGHTS: List[float] = [1.0, 3.0, 10.0, 30.0]
FEEDBACK_CANDIDATES_PER_SEED: List[int] = [8, 16, 32]
FEEDBACK_SCREEN_CANDIDATES: int = 16          # stage A holds candidates at 16
FEEDBACK_CANDIDATE_TEMPERATURE: float = 1.0   # candidate-generation temperature, fixed for the grid


def feedback_grid_stage_a() -> List[FeedbackHP]:
    """Stage A: hold candidates_per_seed at 16 and cross every learning rate with
    every ranking weight (3 x 4 = 12 configs).

    Run each config with ONE training seed (the screening phase) and the five
    generation seeds. This compares the learning-rate and ranking-weight axes
    cheaply before spending anything on the more expensive candidate-count axis.
    """
    grid: List[FeedbackHP] = []
    for lr in FEEDBACK_LEARNING_RATES:
        for alpha in FEEDBACK_RANK_WEIGHTS:
            grid.append(FeedbackHP(lr, alpha, FEEDBACK_SCREEN_CANDIDATES, FEEDBACK_CANDIDATE_TEMPERATURE))
    return grid


def feedback_grid_stage_b(promising: Sequence[FeedbackHP]) -> List[FeedbackHP]:
    """Stage B: for the promising configs kept from stage A, test the other two
    candidate counts, 8 and 32 (16 was already evaluated in stage A). Learning
    rate, ranking weight and temperature carry over unchanged. Keep the list of
    promising configs short so this stage stays small.
    """
    grid: List[FeedbackHP] = []
    for hp in promising:
        for c in (8, 32):
            grid.append(FeedbackHP(hp.learning_rate, hp.rank_weight_alpha, c, hp.generation_temperature))
    return grid


def feedback_grid() -> List[FeedbackHP]:
    """Default screening grid is stage A (learning rate x ranking weight at 16
    candidates per seed). Expand the promising subset with feedback_grid_stage_b().
    """
    return feedback_grid_stage_a()


# --- property scoring (pure numpy, so it is unit-testable) -----------------
def score_qed(props: Dict[str, "float"]) -> float:
    """QED-only score for the feedback_qed condition."""
    return float(props["qed"])


def score_composite(
    props: Dict[str, "float"],
    w_qed: float = 0.34,
    w_fsp3: float = 0.33,
    w_esol: float = 0.33,
    esol_target: float = -3.0,
    esol_tolerance: float = 3.0,
) -> float:
    """Composite score for the feedback_composite condition.

    QED and Fsp3 are already on 0..1. ESOL (logS) is folded into 0..1 by
    distance from the target solubility, so being close to esol_target scores
    high. Replace the weights or the ESOL transform here to match the exact
    Stage 2 definition used in the project if it differs.
    """
    qed = float(props["qed"])
    fsp3 = float(props["fsp3"])
    logs = float(props["esol"])
    esol_norm = max(0.0, 1.0 - abs(logs - esol_target) / esol_tolerance)
    return w_qed * qed + w_fsp3 * fsp3 + w_esol * esol_norm


def score_fn_for(condition: str) -> Callable[[Dict[str, float]], float]:
    if condition == "feedback_qed":
        return score_qed
    if condition == "feedback_composite":
        return score_composite
    raise KeyError(f"'{condition}' is not a feedback condition.")


def rank_loss(candidate_logprobs, candidate_scores, gamma: float = 0.01):
    """BRIO-style pairwise margin rank loss (torch).

    candidate_logprobs: 1D tensor, f(S) = sum_t log p_theta for each candidate.
    candidate_scores:   1D tensor of property scores PS(S) for the same set.
    Candidates are sorted by property score descending; for every pair i<j the
    loss penalises the model when the higher-property candidate does not have
    the higher log-probability by at least the rank-scaled margin.
    """
    import torch

    order = torch.argsort(candidate_scores, descending=True)
    lp = candidate_logprobs[order]
    n = lp.shape[0]
    loss = lp.new_zeros(())
    for i in range(n):
        for j in range(i + 1, n):
            margin = (j - i) * gamma
            loss = loss + torch.clamp(lp[j] - lp[i] + margin, min=0.0)
    return loss


def ranking_likelihood_correlation(scores, logprobs) -> float:
    """Spearman correlation between candidate property scores and their sequence
    log-likelihoods.

    This is the central feedback diagnostic. The rank loss can only teach the
    model when there is a coherent relationship for it to sharpen. A strong
    positive value means the model already tends to assign higher likelihood to
    better molecules, so the objective has signal. A value near zero means it does
    not, and the rank loss has little to propagate through the frozen
    cross-attention. Monitor this alongside the property and diversity metrics
    when choosing a configuration, not the training loss alone.
    """
    import numpy as np
    from scipy.stats import spearmanr

    s = np.asarray(scores, dtype=float)
    lp = np.asarray(logprobs, dtype=float)
    if len(s) < 3 or len(s) != len(lp):
        return float("nan")
    rho, _ = spearmanr(s, lp)
    return float(rho)


def sequence_log_likelihood(
    model,
    tokenizer,
    encoder_selfies: str,
    decoder_selfies: str,
    device: Optional[str] = None,
) -> float:
    """Teacher-forced log-likelihood f(S) = sum_t log p(target_t | target_<t, seed).

    This is the same quantity the rank loss ranks, computed for one (seed,
    candidate) pair so it can be correlated with that candidate's property score.
    """
    import torch

    device = device or ("cuda" if _cuda_available() else "cpu")
    model = model.to(device)
    model.eval()

    enc = tokenizer(encoder_selfies, return_tensors="pt", truncation=True, max_length=MAX_LEN).to(device)
    dec = tokenizer(decoder_selfies, return_tensors="pt", truncation=True, max_length=MAX_LEN).to(device)
    labels = dec["input_ids"].clone()
    labels[labels == tokenizer.pad_token_id] = -100

    with torch.no_grad():
        out = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"], labels=labels)

    # HF returns mean cross-entropy over non-ignored tokens; scale back to a sum.
    n_tokens = int((labels != -100).sum().item())
    return float(-out.loss.item() * n_tokens)


def evaluate_generation_sample(
    smiles: Sequence[str],
    reference_smiles: Optional[set] = None,
    diversity_subsample: int = 1000,
) -> Dict[str, float]:
    """Monitoring metrics for the feedback runs.

    Returns validity, uniqueness, scaffold diversity, internal diversity, and
    the mean of QED, Fsp3, LogP, MW and ESOL over the valid molecules. Novelty
    is included only if a reference set of canonical SMILES is supplied. This is
    intended for a monitoring subsample during the grid search; the full novelty
    and NNS metrics against MOSES stay in the FPSim2-based evaluation script.
    """
    from rdkit import Chem
    from rdkit.Chem import Crippen, Descriptors, QED, rdMolDescriptors
    from rdkit.Chem.Scaffolds import MurckoScaffold
    from rdkit import DataStructs
    import numpy as np

    mols, canon = [], []
    for smi in smiles:
        m = Chem.MolFromSmiles(smi)
        if m is not None:
            mols.append(m)
            canon.append(Chem.MolToSmiles(m))

    n_raw = len(smiles)
    n_valid = len(mols)
    metrics: Dict[str, float] = {
        "validity": n_valid / n_raw if n_raw else 0.0,
        "uniqueness": len(set(canon)) / n_valid if n_valid else 0.0,
    }
    if reference_smiles is not None and n_valid:
        novel = sum(1 for c in canon if c not in reference_smiles)
        metrics["novelty"] = novel / n_valid

    if n_valid:
        scaffolds = {MurckoScaffold.MurckoScaffoldSmiles(mol=m) for m in mols}
        metrics["scaffold_diversity"] = len(scaffolds) / n_valid

        qeds, fsp3s, logps, mws, esols = [], [], [], [], []
        for m in mols:
            qeds.append(QED.qed(m))
            fsp3s.append(rdMolDescriptors.CalcFractionCSP3(m))
            logp = Crippen.MolLogP(m)
            mw = Descriptors.MolWt(m)
            logps.append(logp)
            mws.append(mw)
            # ESOL (Delaney 2004) uses the NUMBER OF AROMATIC RINGS
            # (CalcNumAromaticRings), not the aromatic-atom proportion. Using the
            # proportion is the earlier bug that inflated solubility by ~1 log
            # unit; this matches the corrected project formula.
            rb = rdMolDescriptors.CalcNumRotatableBonds(m)
            n_arom_rings = rdMolDescriptors.CalcNumAromaticRings(m)
            esol = 0.16 - 0.63 * logp - 0.0062 * mw + 0.066 * rb - 0.74 * n_arom_rings
            esols.append(esol)
        metrics["qed_mean"] = float(np.mean(qeds))
        metrics["fsp3_mean"] = float(np.mean(fsp3s))
        metrics["logp_mean"] = float(np.mean(logps))
        metrics["mw_mean"] = float(np.mean(mws))
        metrics["esol_mean"] = float(np.mean(esols))

        # internal diversity on a subsample (quadratic in set size)
        take = min(diversity_subsample, n_valid)
        fps = [rdMolDescriptors.GetMorganFingerprintAsBitVect(mols[i], 2, 2048) for i in range(take)]
        sims = []
        for i in range(take):
            row = DataStructs.BulkTanimotoSimilarity(fps[i], fps[i + 1:])
            sims.extend(row)
        metrics["internal_diversity"] = float(1.0 - np.mean(sims)) if sims else 0.0

    return metrics


# Floors used when selecting a feedback config, so a diversity collapse cannot be
# rewarded just because the composite score went up.
DIVERSITY_FLOOR_INTERNAL: float = 0.70
UNIQUENESS_FLOOR: float = 0.90


def _config_objective(row: Dict[str, object]) -> float:
    """Selection objective: target-property gains, novelty, and a bonus for a
    positive ranking-likelihood correlation. Deliberately does NOT reward lower
    diversity; diversity is handled as a floor in rank_configs()."""
    m = row["metrics"]  # type: ignore
    corr = m.get("ranking_likelihood_spearman", 0.0)
    if corr != corr:  # nan
        corr = 0.0
    return (m.get("qed_mean", 0.0) + m.get("fsp3_mean", 0.0)
            + 0.5 * m.get("novelty", 0.0)
            + 0.3 * max(0.0, corr))


def rank_configs(
    rows: List[Dict[str, object]],
    min_internal_diversity: float = DIVERSITY_FLOOR_INTERNAL,
    min_uniqueness: float = UNIQUENESS_FLOOR,
) -> List[Dict[str, object]]:
    """Rank feedback configs best-first, applying the diversity/uniqueness floors.

    Configs that fall below either floor are set aside first, so a config that
    lifts the composite score while collapsing diversity cannot rank at the top.
    Only if no config clears the floors does it fall back to ranking them all, so
    the caller still gets an ordering (and can see, via the diagnostics, that the
    fallback happened).
    """
    def passes(m: Dict[str, float]) -> bool:
        return (m.get("internal_diversity", 0.0) >= min_internal_diversity
                and m.get("uniqueness", 0.0) >= min_uniqueness)

    survivors = [r for r in rows if passes(r["metrics"])]  # type: ignore
    pool = survivors if survivors else list(rows)
    return sorted(pool, key=_config_objective, reverse=True)


def select_best_config(
    rows: List[Dict[str, object]],
    min_internal_diversity: float = DIVERSITY_FLOOR_INTERNAL,
    min_uniqueness: float = UNIQUENESS_FLOOR,
) -> Tuple["FeedbackHP", Dict[str, object]]:
    """Choose the best feedback config, looking beyond training loss.

    Implements the caution that a config which lifts the composite score while
    substantially cutting diversity is not automatically superior: any config
    whose internal diversity or uniqueness falls below the floors is set aside
    first. Among the survivors, configs are ranked by target-property gains,
    novelty, and the ranking-likelihood correlation (which indicates whether the
    rank objective had signal to work with). Returns (best_hp, diagnostics).
    """
    ranked = rank_configs(rows, min_internal_diversity, min_uniqueness)
    n_survivors = sum(
        1 for r in rows
        if r["metrics"].get("internal_diversity", 0.0) >= min_internal_diversity  # type: ignore
        and r["metrics"].get("uniqueness", 0.0) >= min_uniqueness  # type: ignore
    )
    best = ranked[0]
    diagnostics = {
        "used_diversity_fallback": n_survivors == 0,
        "n_survivors": n_survivors,
        "n_total": len(rows),
    }
    return best["hp"], diagnostics  # type: ignore


@dataclass
class GridResult:
    rows: List[Dict[str, object]]

    @property
    def best(self) -> "FeedbackHP":
        """Best config under the diversity-guarded, correlation-aware criterion."""
        hp, _ = select_best_config(self.rows)
        return hp

    def best_with_diagnostics(self) -> Tuple["FeedbackHP", Dict[str, object]]:
        return select_best_config(self.rows)


@dataclass
class FeedbackSample:
    """What one feedback config produces for evaluation.

    candidate_scores and candidate_logprobs are paired: candidate i has property
    score candidate_scores[i] and sequence log-likelihood candidate_logprobs[i].
    Their Spearman correlation is the central ranking diagnostic. validation_smiles
    is a separately generated sample used for the validity, diversity, and
    property metrics.
    """

    candidate_scores: Sequence[float]
    candidate_logprobs: Sequence[float]
    validation_smiles: Sequence[str]


# The one model-specific piece of the feedback screen:
#   (hp, base_checkpoint, train_seed, generation_seeds) -> FeedbackSample
FeedbackTrainSampleFn = Callable[[FeedbackHP, str, int, List[int]], FeedbackSample]

# The one model-specific piece of a final feedback run:
#   (base_checkpoint, chosen_hp, train_seed, out_dir) -> optional (model, tokenizer)
# Returning the trained model and tokenizer lets the wrapper run the validation
# panel; returning None skips it.
FeedbackStepFn = Callable[[str, FeedbackHP, int, str], Optional[Tuple[object, object]]]


def _training_pool_smiles(csv_path: str, splits_path: str) -> set:
    from data_split import get_training_pool_frame, resolve_smiles_column

    pool = get_training_pool_frame(csv_path, splits_path)
    return set(pool[resolve_smiles_column(pool)].astype(str).tolist())


def run_feedback_grid(
    condition: str,
    base_checkpoint: str,
    train_and_sample_fn: FeedbackTrainSampleFn,
    grid: Optional[List[FeedbackHP]] = None,
    n_generation_seeds: int = DEFAULT_N_GENERATION_SEEDS,
    train_phase: str = "screening",
    csv_path: str = DEFAULT_CSV,
    root: str = DEFAULT_PROJECT_ROOT,
    splits_path: str = DEFAULT_SPLITS_PATH,
) -> GridResult:
    """Feedback hyperparameter screening (workflow step 1).

    Runs each grid config with ONE training seed (the screening phase) and the
    five generation seeds, computes the full monitoring panel plus the
    ranking-likelihood Spearman correlation, and returns a GridResult whose .best
    applies the diversity-guarded, correlation-aware selection.

    train_and_sample_fn is the only model-specific piece: given a config, the base
    checkpoint, the screening train seed, and the generation seeds, it trains a
    short feedback pass and returns candidate (score, logprob) pairs plus a
    validation sample. Everything else here (the grid, the metrics, the
    correlation, the selection) is generic and runs as provided.
    """
    from seed_config import generation_base_seeds, train_seeds_for_phase

    if condition not in ("feedback_composite", "feedback_qed"):
        raise KeyError(f"'{condition}' is not a feedback condition.")
    grid = grid if grid is not None else feedback_grid()
    train_seed = train_seeds_for_phase(train_phase)[0]
    gen_seeds = generation_base_seeds(n_generation_seeds)
    reference = _training_pool_smiles(csv_path, splits_path)

    rows: List[Dict[str, object]] = []
    for hp in grid:
        sample = train_and_sample_fn(hp, base_checkpoint, train_seed, gen_seeds)
        metrics = evaluate_generation_sample(sample.validation_smiles, reference_smiles=reference)
        metrics["ranking_likelihood_spearman"] = ranking_likelihood_correlation(
            sample.candidate_scores, sample.candidate_logprobs
        )
        rows.append({"hp": hp, "metrics": metrics})
    return GridResult(rows)


def screen_feedback_two_stage(
    condition: str,
    base_checkpoint: str,
    train_and_sample_fn: FeedbackTrainSampleFn,
    keep_top: int = 3,
    n_generation_seeds: int = DEFAULT_N_GENERATION_SEEDS,
    csv_path: str = DEFAULT_CSV,
    root: str = DEFAULT_PROJECT_ROOT,
    splits_path: str = DEFAULT_SPLITS_PATH,
) -> GridResult:
    """The staged screen: Stage A then a small Stage B.

    Stage A crosses every learning rate with every ranking weight at 16 candidates
    per seed (12 configs), all on one training seed and five generation seeds.
    The top keep_top configs (after the diversity/uniqueness floors) then go to
    Stage B, which tries 8 and 32 candidates per seed for just those. Returns the
    combined GridResult, whose .best gives the configuration to carry into the
    multi-seed runs.
    """
    a = run_feedback_grid(
        condition, base_checkpoint, train_and_sample_fn,
        grid=feedback_grid_stage_a(), n_generation_seeds=n_generation_seeds,
        csv_path=csv_path, root=root, splits_path=splits_path,
    )
    promising = [r["hp"] for r in rank_configs(a.rows)[:keep_top]]  # type: ignore
    b = run_feedback_grid(
        condition, base_checkpoint, train_and_sample_fn,
        grid=feedback_grid_stage_b(promising), n_generation_seeds=n_generation_seeds,
        csv_path=csv_path, root=root, splits_path=splits_path,
    )
    return GridResult(a.rows + b.rows)


def make_feedback_train_fn(
    chosen_hp: FeedbackHP,
    base_checkpoints: Sequence[str],
    feedback_step_fn: FeedbackStepFn,
    csv_path: str = DEFAULT_CSV,
    root: str = DEFAULT_PROJECT_ROOT,
    splits_path: str = DEFAULT_SPLITS_PATH,
) -> protocol.TrainRunFn:
    """Build the train_run_fn for the final feedback runs with a chosen config.

    base_checkpoints is indexed by run_index and gives the fine-tuned model each
    feedback run refines. feedback_step_fn is the model-specific training step: it
    loads the base, runs feedback tuning with chosen_hp at the given train seed,
    saves to out_dir, and optionally returns (model, tokenizer) so the shared
    validation panel can run afterwards.
    """

    def train_run_fn(condition: str, run_index: int, train_seed: int, data_seed: int) -> str:
        base = base_checkpoints[run_index]
        out_dir = checkpoint_dir(condition, run_index, root)
        os.makedirs(out_dir, exist_ok=True)

        result = feedback_step_fn(base, chosen_hp, train_seed, out_dir)

        if result is not None:
            model, tokenizer = result
            try:
                validate_after_training(model, tokenizer, condition, run_index, csv_path, root, splits_path)
            except Exception as e:  # never lose a trained model to a monitoring hiccup
                print(f"[{condition} run{run_index}] validation monitoring skipped: {e}")
        return out_dir

    return train_run_fn


# ---------------------------------------------------------------------------
# DISPATCH
# ---------------------------------------------------------------------------
def train_family(
    condition: str,
    phase: str = DEFAULT_TRAIN_PHASE,
    run_indices: Optional[Sequence[int]] = None,
    csv_path: str = DEFAULT_CSV,
    root: str = DEFAULT_PROJECT_ROOT,
    splits_path: str = DEFAULT_SPLITS_PATH,
) -> List[str]:
    """Train a full-FT or MPT condition's runs via the protocol. Returns paths.

    phase controls how many seeds are available: 'screening' (1), 'initial' (3),
    or 'final' (5). By default every run in the phase trains, in order.

    run_indices lets you train only specific runs instead of the whole phase.
    For example, run_indices=[1] trains only the second seed of the 'initial'
    phase (seed 42), useful for picking a phase up one run at a time or after a
    disconnect, without retraining runs you already have. Indices are 0-based
    and must be valid for the phase (0 for 'screening'; 0, 1, 2 for 'initial';
    0 through 4 for 'final').

    Feedback conditions are trained with train_feedback() instead, because they
    need a base checkpoint and a chosen hyperparameter config.
    """
    family = FAMILY_OF.get(condition)
    if family == "pretrained":
        raise ValueError("The pretrained baseline is not trained; use it for generation only.")
    if family == "full_ft":
        fn = make_full_ft_train_fn(csv_path, root, splits_path)
    elif family == "mpt":
        fn = make_mpt_train_fn(csv_path, root, splits_path)
    elif family == "feedback":
        raise ValueError("Use train_feedback() for feedback conditions.")
    else:
        raise KeyError(f"Unknown condition '{condition}'.")
    return protocol.train_condition(condition, fn, phase, run_indices)


def train_feedback(
    condition: str,
    chosen_hp: FeedbackHP,
    base_checkpoints: Sequence[str],
    feedback_step_fn: FeedbackStepFn,
    phase: str = DEFAULT_TRAIN_PHASE,
    run_indices: Optional[Sequence[int]] = None,
    csv_path: str = DEFAULT_CSV,
    root: str = DEFAULT_PROJECT_ROOT,
    splits_path: str = DEFAULT_SPLITS_PATH,
) -> List[str]:
    """Train a feedback condition's runs (one per training seed in the phase).

    Supply one base checkpoint per run_index you are training, since each
    feedback run refines the corresponding base checkpoint (base_checkpoints[i]
    for run_index i). feedback_step_fn is the model-specific training step (see
    make_feedback_train_fn).

    run_indices restricts training to specific runs, same as in train_family:
    for example run_indices=[1] trains only the second seed of the phase. When
    run_indices is not given, every run in the phase trains, and base_checkpoints
    must have one entry per run in the phase.
    """
    if FAMILY_OF.get(condition) != "feedback":
        raise ValueError(f"'{condition}' is not a feedback condition.")

    indices = list(run_indices) if run_indices is not None else list(range(get_num_runs(condition, phase)))
    required = max(indices) + 1 if indices else 0
    if len(base_checkpoints) < required:
        raise ValueError(
            f"Training run_index up to {max(indices)} needs at least {required} "
            f"base checkpoints (indexed 0..{required - 1}), got {len(base_checkpoints)}."
        )
    fn = make_feedback_train_fn(chosen_hp, base_checkpoints, feedback_step_fn, csv_path, root, splits_path)
    return protocol.train_condition(condition, fn, phase, run_indices)


def load_pretrained_for_generation():
    """The baseline is generated from the released checkpoint, loaded once.

    Pass the returned handle as the single-element checkpoints sequence to
    protocol.generate_condition("pretrained", [handle], ...).
    """
    return load_base_model_and_tokenizer()


if __name__ == "__main__":
    # Safe dry run: exercises the pure-logic pieces without any heavy imports.
    print("families:")
    for cond, fam in FAMILY_OF.items():
        print(f"  {cond:<20} -> {fam}")

    print(f"\nfeedback grid has {len(feedback_grid())} configurations:")
    for hp in feedback_grid():
        print(f"  lr={hp.learning_rate:<7} alpha={hp.rank_weight_alpha:<4} "
              f"cands={hp.candidates_per_seed:<3} temp={hp.generation_temperature}")

    demo_props = {"qed": 0.82, "fsp3": 0.45, "esol": -3.1}
    print(f"\nscore_qed(demo)       = {score_qed(demo_props):.3f}")
    print(f"score_composite(demo) = {score_composite(demo_props):.3f}")
