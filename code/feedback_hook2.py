"""
feedback_hook.py

The one model-specific piece the feedback screen in train.py needs: a
FeedbackTrainSampleFn. The harness (run_feedback_grid / screen_feedback_two_stage)
ALREADY varies learning rate, ranking weight and candidates-per-seed for you by
handing this function a different FeedbackHP each time. You do NOT write that
loop. You write ONE function that reads hp.learning_rate, hp.rank_weight_alpha
and hp.candidates_per_seed and trains/samples accordingly; the grid feeds every
combination through it.

What this hook does per config (hp):
  1. loads the fine-tuned base checkpoint as a FULLY TRAINABLE seq2seq model
     (rank-loss gradients must reach the base weights, not a frozen adapter);
  2. pre-generates a FROZEN candidate set per seed molecule at
     hp.candidates_per_seed and hp.generation_temperature, and scores each
     candidate with the condition's property score (the v3 "frozen candidates"
     design, which was more stable than regenerating candidates live);
  3. trains a short pass with L = Lce + hp.rank_weight_alpha * Lrank at
     hp.learning_rate, where Lce is the denoising reconstruction of the seed and
     Lrank is the BRIO pairwise margin loss over the frozen candidates;
  4. returns a FeedbackSample: paired (score, post-training log-likelihood) over
     the candidates for the ranking-likelihood Spearman diagnostic, plus a small
     generated validation sample for the validity / diversity / property panel.

Drop this next to train.py, corpus.py, protocol.py, etc. Heavy imports are lazy
so the module still imports on the CPU-only local machine.

Written for Python 3.9 (Optional[...], no X | Y unions).
"""

from typing import Dict, List, Tuple

import train
from train import (
    FeedbackHP,
    FeedbackSample,
    MASK_PROB,
    MAX_LEN,
    VALIDATION_SAMPLE_SIZE,
    generate_molecules,
    selfies_to_canonical_smiles,
)
from data_split import get_validation_frame, resolve_selfies_column


# Screening-scale knobs. Kept small so a 12-config Stage A plus a 6-config
# Stage B finishes in reasonable time on a single T4. Raise SEED_POOL / EPOCHS
# for the final multi-seed runs if you want a longer feedback pass.
FEEDBACK_SEED_POOL: int = 2000       # seed molecules used per config
FEEDBACK_SCREEN_EPOCHS: int = 10    # passes over the seed pool per config


# ---------------------------------------------------------------------------
# PROPERTY SCORING HELPERS
# ---------------------------------------------------------------------------
def _props(mol) -> Dict[str, float]:
    """QED / Fsp3 / ESOL for one RDKit mol, using the corrected ESOL formula
    (CalcNumAromaticRings, matching evaluate_generation_sample in train.py)."""
    from rdkit.Chem import Crippen, Descriptors, QED, rdMolDescriptors

    # logP is calculated ONLY as an intermediate step for the ESOL approximation.
    # It is not passed to the ranking loss objective.
    logp = Crippen.MolLogP(mol)
    mw = Descriptors.MolWt(mol)
    rb = rdMolDescriptors.CalcNumRotatableBonds(mol)
    n_arom = rdMolDescriptors.CalcNumAromaticRings(mol)
    esol = 0.16 - 0.63 * logp - 0.0062 * mw + 0.066 * rb - 0.74 * n_arom
    return {
        "qed": float(QED.qed(mol)),
        "fsp3": float(rdMolDescriptors.CalcFractionCSP3(mol)),
        "esol": float(esol),
    }


def _mask_noise(input_ids, tokenizer, mask_prob: float = MASK_PROB):
    """Mask mask_prob of non-special encoder tokens; required for candidate
    diversity, otherwise every candidate collapses to the seed."""
    import torch

    noised = input_ids.clone()
    is_special = torch.zeros_like(noised, dtype=torch.bool)
    for sid in set(tokenizer.all_special_ids):
        is_special |= noised == sid
    do_mask = (~is_special) & (torch.rand(noised.shape) < mask_prob)
    noised[do_mask] = tokenizer.mask_token_id
    return noised


def _generate_candidates(model, tokenizer, seed_selfies, k, temperature, score_fn, device):
    """Frozen candidate set for one seed: a list of (selfies, score) pairs
    (at most k, de-duplicated by canonical SMILES, invalid ones dropped)."""
    import torch
    from rdkit import Chem

    enc = tokenizer(seed_selfies, return_tensors="pt", truncation=True, max_length=MAX_LEN)
    noised = _mask_noise(enc["input_ids"], tokenizer).to(device)
    attn = enc["attention_mask"].to(device)

    model.eval()
    with torch.no_grad():
        out = model.generate(
            input_ids=noised, attention_mask=attn, max_length=MAX_LEN,
            do_sample=True, temperature=temperature,
            top_k=train.GENERATION_TOP_K, top_p=train.GENERATION_TOP_P,
            num_return_sequences=k,
        )

    cands: List[Tuple[str, float]] = []
    seen = set()
    for row in out:
        raw = tokenizer.decode(row, skip_special_tokens=True,
                               clean_up_tokenization_spaces=False).replace(" ", "")
        if not raw:
            continue
        smi = selfies_to_canonical_smiles(raw)
        if smi is None or smi in seen:
            continue
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        seen.add(smi)
        cands.append((raw, float(score_fn(_props(mol)))))
    return cands


def _candidate_logprobs(model, tokenizer, seed_selfies, cand_selfies, device, no_grad):
    """Summed teacher-forced log-likelihood of each candidate given the seed.

    Returns a 1D tensor of length len(cand_selfies). Differentiable unless
    no_grad=True. This is the same f(S) = sum_t log p_theta the rank loss ranks,
    computed for a whole candidate set at once by repeating the encoder input.
    """
    import torch
    import torch.nn.functional as F
    from contextlib import nullcontext

    k = len(cand_selfies)
    enc = tokenizer(seed_selfies, return_tensors="pt", truncation=True, max_length=MAX_LEN)
    dec = tokenizer(list(cand_selfies), return_tensors="pt", padding=True,
                    truncation=True, max_length=MAX_LEN)

    input_ids = enc["input_ids"].repeat(k, 1).to(device)
    attn = enc["attention_mask"].repeat(k, 1).to(device)
    labels = dec["input_ids"].to(device)

    pad_id = tokenizer.pad_token_id
    label_mask = (labels != pad_id)
    masked_labels = labels.clone()
    masked_labels[~label_mask] = -100

    ctx = torch.no_grad() if no_grad else nullcontext()
    with ctx:
        out = model(input_ids=input_ids, attention_mask=attn, labels=masked_labels)
        logp = F.log_softmax(out.logits, dim=-1)          # (k, T, V)
        gather_idx = labels.clamp(min=0).unsqueeze(-1)    # pad -> 0; masked out next
        tok_lp = logp.gather(-1, gather_idx).squeeze(-1)  # (k, T)
        seq_lp = (tok_lp * label_mask).sum(dim=1)         # (k,)
    return seq_lp


# ---------------------------------------------------------------------------
# THE HOOK
# ---------------------------------------------------------------------------
def make_feedback_train_and_sample_fn(
    condition: str,
    csv_path: str,
    root: str,
    splits_path: str,
    save_dir: str = None,
):
    """Build the FeedbackTrainSampleFn the screen calls once per grid config.

    The returned function reads hp.learning_rate, hp.rank_weight_alpha and
    hp.candidates_per_seed, so each grid point trains with a different setting
    without you writing any loop over those axes. Pass the result straight to
    run_feedback_grid(...) or screen_feedback_two_stage(...) as train_and_sample_fn.

    If save_dir is given, the trained full model (all weights, not an adapter)
    and its tokenizer are written there right after the feedback pass, so the
    checkpoint can be reloaded later with AutoModelForSeq2SeqLM.from_pretrained
    for generation. Leave save_dir as None during sweeps so grid runs don't each
    overwrite a folder; set it only for the single config you want to keep.
    """
    
    # Custom Multi-Parameter Optimization (MPO) Score
    # Bypasses train.score_fn_for to ensure QED, Fsp3, and ESOL are evenly weighted.
import math

def _desirability(x, center, width):
    """Smooth 0-1 desirability, 0.5 at the threshold. Sigmoid rather than a
    hard cutoff so candidates on the same side of the threshold still order,
    which a rank loss needs (ties contribute no pairs)."""
    return 1.0 / (1.0 + math.exp(-(x - center) / width))

def score_fn(props):
    d_qed  = _desirability(props.get("qed", 0.0), 0.80, 0.05)    # Bickerton 2012
    d_fsp3 = _desirability(props.get("fsp3", 0.0), 0.42, 0.08)   # Lovering 2009
    d_esol = _desirability(props.get("esol", -99.0), -3.0, 0.50)  # Delaney 2004
    # Geometric mean: no property can compensate for another failing.
    return (d_qed * d_fsp3 * d_esol) ** (1.0 / 3.0)

    def train_and_sample(hp: FeedbackHP, base_checkpoint: str, train_seed: int,
                          generation_seeds: List[int]) -> FeedbackSample:
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, set_seed
        from train import rank_loss

        set_seed(train_seed)
        device = "cuda" if torch.cuda.is_available() else "cpu"

        # Full base model, EVERY parameter trainable (not a prefix adapter): the
        # rank-loss gradient has to reach the base weights to have any effect.
        model = AutoModelForSeq2SeqLM.from_pretrained(base_checkpoint).to(device)
        tokenizer = AutoTokenizer.from_pretrained(base_checkpoint)

        val_df = get_validation_frame(csv_path, splits_path)
        selfies_col = resolve_selfies_column(val_df)
        seed_selfies = (
            val_df[selfies_col].dropna()
            .sample(n=min(FEEDBACK_SEED_POOL, len(val_df)), random_state=train_seed)
            .tolist()
        )

        # 1. Freeze a candidate set per seed at THIS config's candidates_per_seed
        #    and generation_temperature. Frozen (not regenerated during training)
        #    for stability.
        frozen: List[Tuple[str, List[str], List[float]]] = []
        for s in seed_selfies:
            cands = _generate_candidates(
                model, tokenizer, s, hp.candidates_per_seed,
                hp.generation_temperature, score_fn, device,
            )
            if len(cands) >= 2:  # need at least a pair to rank
                frozen.append((s, [c[0] for c in cands], [c[1] for c in cands]))

        # 2. Short feedback pass: L = Lce(seed) + alpha * Lrank(candidates), at
        #    hp.learning_rate. Lce is the denoising reconstruction that keeps the
        #    model generative; Lrank is the BRIO margin loss over frozen candidates.
        optim = torch.optim.AdamW(model.parameters(), lr=hp.learning_rate)
        collate = train.make_denoising_collator(tokenizer)
        model.train()
        for _ in range(FEEDBACK_SCREEN_EPOCHS):
            for seed_s, sel, sco in frozen:
                feat = train.tokenize_selfies(tokenizer, [seed_s])
                batch = {name: t.to(device) for name, t in collate(feat).items()}
                ce = model(**batch).loss

                lp = _candidate_logprobs(model, tokenizer, seed_s, sel, device, no_grad=False)
                scores_t = torch.tensor(sco, dtype=lp.dtype, device=device)
                lrank = rank_loss(lp, scores_t, gamma=hp.margin_gamma)

                loss = ce + hp.rank_weight_alpha * lrank
                optim.zero_grad()
                loss.backward()
                
                # --- STABILITY FIX: Gradient Clipping ---
                # Prevents large alpha values (e.g., 10.0) from exploding the gradients 
                # and turning the model weights into NaNs.
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                
                optim.step()

        # 2b. Save the trained full model so it can be reused for generation.
        #     Placed here (after training, before the eval-only diagnostic) so
        #     the saved weights are exactly the feedback-tuned model. This is a
        #     full checkpoint, not a prefix adapter, so a plain from_pretrained
        #     reloads it directly with no merge_and_unload step.
        if save_dir is not None:
            import os
            os.makedirs(save_dir, exist_ok=True)
            model.save_pretrained(save_dir)
            tokenizer.save_pretrained(save_dir)
            print(
                f"[feedback_hook] saved trained model to {save_dir} "
                f"(alpha={hp.rank_weight_alpha}, lr={hp.learning_rate}, "
                f"temp={hp.generation_temperature}, seed={train_seed})"
            )

        # 3a. Ranking diagnostic: paired (score, POST-training log-likelihood)
        #     over all frozen candidates, flattened across seeds. Its Spearman
        #     correlation is the primary readout the screen reports.
        model.eval()
        flat_scores: List[float] = []
        flat_logprobs: List[float] = []
        for seed_s, sel, sco in frozen:
            lp = _candidate_logprobs(model, tokenizer, seed_s, sel, device, no_grad=True)
            flat_scores.extend(sco)
            flat_logprobs.extend(lp.detach().cpu().tolist())

        # 3b. Validation sample for the validity / diversity / property panel.
        val_smiles = generate_molecules(
            model, tokenizer, seed_selfies, VALIDATION_SAMPLE_SIZE,
            temperature=1.0, seed=generation_seeds[0],
        )

        return FeedbackSample(
            candidate_scores=flat_scores,
            candidate_logprobs=flat_logprobs,
            validation_smiles=val_smiles,
        )

    return train_and_sample


# ---------------------------------------------------------------------------
# READABLE GRID SUMMARY (look beyond the training loss)
# ---------------------------------------------------------------------------
def summarise(result) -> List[str]:
    """One line per grid config: its settings and the metrics that actually
    drive selection. Print these to inspect the whole grid, as advised, rather
    than only the winner. ranking_likelihood_spearman is the central diagnostic:
    a value near zero means the rank objective had no coherent signal to sharpen.
    """
    lines: List[str] = []
    for r in result.rows:
        hp = r["hp"]
        m = r["metrics"]
        lines.append(
            f"lr={hp.learning_rate:<7} alpha={hp.rank_weight_alpha:<5} "
            f"cand={hp.candidates_per_seed:<3} | "
            f"spearman={m.get('ranking_likelihood_spearman', float('nan')):+.3f} "
            f"qed={m.get('qed_mean', float('nan')):.3f} "
            f"fsp3={m.get('fsp3_mean', float('nan')):.3f} "
            f"esol={m.get('esol_mean', float('nan')):.3f} "
            f"intdiv={m.get('internal_diversity', float('nan')):.3f} "
            f"nov={m.get('novelty', float('nan')):.3f} "
            f"uniq={m.get('uniqueness', float('nan')):.3f}"
        )
    return lines