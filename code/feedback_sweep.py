"""
feedback_sweep.py
=================
Configurable chemical-feedback (BRIO-style rank loss) training harness for
MolGen-large, designed to run in Google Colab. Edit ALPHA and LEARNING_RATE at
the top of the CONFIG block and run; everything else is centralised there too.

Design notes
------------
* Candidates are PRE-GENERATED ONCE and FROZEN to disk (this is the v3 approach
  that avoided the validity collapse seen with live candidate generation). The
  same frozen candidates are reused across every (alpha, LR) cell so that the
  only thing changing between cells is the training signal, not the ranking
  targets. Regenerate only when you change NUM_CANDIDATES (16 -> 8 / 32).
* The feedback loop trains the FULL base model (no prefix / no PEFT). Rank-loss
  gradients need base-model trainability; a prefix adapter would starve the
  signal.
* Headline diagnostic: Spearman correlation between candidate property score and
  candidate sequence log-probability, tracked from step 0. This is the quantity
  that revealed the architectural ceiling previously (base model ~ +0.10). Watch
  whether any cell moves it upward.
* Python 3.9: Optional[...] only, never X | Y unions.

Staged workflow (matches the expert's plan)
-------------------------------------------
1. Screening: TRAINING_SEED = 11 (single seed), GENERATION_SEEDS = five seeds,
   NUM_CANDIDATES = 16. Sweep the 3 x 4 grid (LR x alpha) by editing the two
   config values (or use the sweep driver at the bottom).
2. Rank cells by the composite table written to RESULTS_CSV. Keep only cells that
   move the score/log-prob correlation AND hold diversity.
3. For the kept cells, re-run prepare_candidates() at NUM_CANDIDATES = 8 and 32.
4. Repeat the kept cells with 3-5 training seeds, then final comparison on the
   untouched MOSES test split.
"""

# ============================ CONFIG ============================
# --- The two you sweep (edit these directly in Colab) ---
ALPHA = 3.0                 # rank-loss weight; screen over [1, 3, 10, 30]
LEARNING_RATE = 3e-5        # screen over [1e-5, 3e-5, 1e-4]

# --- Candidate / seed settings ---
NUM_CANDIDATES = 16         # candidates per seed molecule (16 -> later 8, 32)
TRAINING_SEED = 11          # single training seed for screening
GENERATION_SEEDS = [101, 202, 303, 404, 505]   # five generation seeds
N_SEED_MOLECULES = 500      # encoder seed molecules used per epoch (cost knob)
N_EVAL_SEEDS = 200          # seeds held out for monitoring generation

# --- Scoring ---
SCORING = "qed"             # "qed" (cleaner single signal) or "composite"
# composite targets (used only when SCORING == "composite")
FSP3_TARGET = 0.42          # Lovering 2009
ESOL_TARGET = -3.0          # Delaney 2004 (log mol/L)
QED_TARGET = 0.80           # Bickerton 2012

# --- Loss / optimisation ---
MARGIN = 0.001              # gamma base margin in the rank loss
LENGTH_PENALTY = 1.0        # 1.0 == mean log-prob per token (length-normalised)
LABEL_SMOOTHING = 0.1       # beta on the CE reconstruction anchor
MASK_PROB = 0.15            # encoder mask noise (required for candidate diversity)
MAX_LEN = 128
EPOCHS = 1
TRAIN_BATCH = 4             # seed molecules per optimisation step
GEN_TEMPERATURE = 1.0       # temperature used ONLY for candidate pre-generation
GRAD_CLIP = 1.0

# --- Selection monitoring (this is what you rank configs on, not the loss) ---
N_MONITOR_POINTS = 4        # light generation snapshots spread across training
MONITOR_N_SEEDS = 80        # eval seeds used for the light through-training snapshots
DIV_RETENTION_MIN = 0.85    # a cell must keep >= 85% of its own pre-feedback diversity
DUMP_DISTRIBUTIONS = True   # save raw fsp3/qed/esol arrays per checkpoint for plots

# --- Paths (Colab / Drive) ---
DRIVE_ROOT   = "/content/drive/MyDrive/MOSESproject"
BASE_MODEL_DIR = f"{DRIVE_ROOT}/molgen_finetuned_standard"     # fully trainable base
MOSES_CSV      = f"{DRIVE_ROOT}/moses_full_properties_with_selfies.csv"
CACHE_DIR      = "/content/fb_cache"       # frozen candidates cached here
OUTPUT_DIR     = "/content/fb_runs"        # per-cell checkpoints (local scratch)
RESULTS_CSV    = f"{DRIVE_ROOT}/results/feedback/sweep_results.csv"
TRAJ_DIR       = f"{DRIVE_ROOT}/results/feedback/trajectories"   # per-cell through-training logs
DIST_DIR       = f"{DRIVE_ROOT}/results/feedback/distributions"  # raw property arrays for plots

# --- CSV column names (adjust to match your file) ---
COL_SELFIES = "SEFLIES"     # NB: misspelled in the source CSV (L and F swapped)
COL_SMILES  = "SMILES"      # canonical SMILES column for exact-match novelty
# ===============================================================

import os
import csv
import json
import time
import hashlib
import random
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from scipy.stats import spearmanr

import selfies as sf
from rdkit import Chem, RDLogger
from rdkit.Chem import QED, Descriptors, DataStructs
from rdkit.Chem import rdMolDescriptors as rdMD
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem import AllChem

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

RDLogger.DisableLog("rdApp.*")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ----------------------------- reproducibility -----------------------------
def hashed_seed(*parts) -> int:
    """MD5-hash a condition tuple into an independent 31-bit random state,
    mirroring the generation pipeline so nearby integers do not correlate."""
    key = "|".join(str(p) for p in parts).encode("utf-8")
    return int(hashlib.md5(key).hexdigest(), 16) % (2 ** 31)


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ----------------------------- chemistry helpers -----------------------------
def esol_logS(mol) -> Optional[float]:
    """Delaney (2004) ESOL estimate. Uses CalcNumAromaticRings (proportion of
    aromatic *rings*, not aromatic atoms)."""
    try:
        mw = Descriptors.MolWt(mol)
        logp = Descriptors.MolLogP(mol)
        rb = Descriptors.NumRotatableBonds(mol)
        ap = rdMD.CalcNumAromaticRings(mol) / max(rdMD.CalcNumRings(mol), 1)
        return 0.16 - 0.63 * logp - 0.0062 * mw + 0.066 * rb - 0.74 * ap
    except Exception:
        return None


def _desirability_peak(value: float, target: float, width: float) -> float:
    """Gaussian desirability peaking at target (used for ESOL)."""
    return float(np.exp(-((value - target) ** 2) / (2 * width ** 2)))


def _desirability_min(value: float, threshold: float, width: float) -> float:
    """Sigmoid desirability increasing past threshold (Fsp3, QED)."""
    return float(1.0 / (1.0 + np.exp(-(value - threshold) / width)))


def score_smiles(smiles: str) -> Optional[float]:
    """Property score for a candidate SMILES. Higher is better.
    Returns None for invalid molecules."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    if SCORING == "qed":
        try:
            return float(QED.qed(mol))
        except Exception:
            return None
    # composite: mean desirability toward the three ADME targets
    try:
        fsp3 = rdMD.CalcFractionCSP3(mol)
        qed_v = QED.qed(mol)
        esol = esol_logS(mol)
        if esol is None:
            return None
        d_fsp3 = _desirability_min(fsp3, FSP3_TARGET, 0.10)
        d_qed = _desirability_min(qed_v, QED_TARGET, 0.10)
        d_esol = _desirability_peak(esol, ESOL_TARGET, 1.0)
        return float((d_fsp3 + d_qed + d_esol) / 3.0)
    except Exception:
        return None


def target_properties(smiles: str) -> Optional[Dict[str, float]]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    esol = esol_logS(mol)
    try:
        return {
            "fsp3": float(rdMD.CalcFractionCSP3(mol)),
            "qed": float(QED.qed(mol)),
            "esol": float(esol) if esol is not None else float("nan"),
            "mw": float(Descriptors.MolWt(mol)),
            "logp": float(Descriptors.MolLogP(mol)),
        }
    except Exception:
        return None


def canonical(smiles: str) -> Optional[str]:
    mol = Chem.MolFromSmiles(smiles)
    return Chem.MolToSmiles(mol) if mol is not None else None


def selfies_str_to_smiles(selfies_str: str) -> Optional[str]:
    try:
        smi = sf.decoder(selfies_str)
        return canonical(smi)
    except Exception:
        return None


def murcko_scaffold(smiles: str) -> Optional[str]:
    try:
        return MurckoScaffold.MurckoScaffoldSmiles(smiles=smiles)
    except Exception:
        return None


def morgan_fp(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)


# ----------------------------- data loading -----------------------------
def load_seed_pool() -> Tuple[List[str], set]:
    """Return (selfies_seed_pool, moses_canonical_smiles_set).
    The SELFIES column name is COL_SELFIES (misspelled 'SEFLIES' in the source)."""
    usecols = [COL_SELFIES, COL_SMILES]
    df = pd.read_csv(MOSES_CSV, usecols=usecols)
    df = df.dropna(subset=[COL_SELFIES])
    selfies_all = df[COL_SELFIES].astype(str).tolist()

    # exact-match novelty reference: canonicalise the MOSES SMILES column once
    moses_canon = set()
    for smi in df[COL_SMILES].astype(str).tolist():
        c = canonical(smi)
        if c is not None:
            moses_canon.add(c)

    # deterministic seed sampling from the training seed
    rng = random.Random(hashed_seed("seedpool", TRAINING_SEED))
    idx = list(range(len(selfies_all)))
    rng.shuffle(idx)
    n_needed = N_SEED_MOLECULES + N_EVAL_SEEDS
    picked = [selfies_all[i] for i in idx[:n_needed]]
    train_seeds = picked[:N_SEED_MOLECULES]
    eval_seeds = picked[N_SEED_MOLECULES:n_needed]
    return train_seeds, eval_seeds, moses_canon


# ----------------------------- model / tokenisation -----------------------------
def load_model_and_tokenizer():
    tok = AutoTokenizer.from_pretrained(BASE_MODEL_DIR)
    model = AutoModelForSeq2SeqLM.from_pretrained(BASE_MODEL_DIR).to(DEVICE)
    return model, tok


def encode(tok, selfies_list: List[str]):
    enc = tok(selfies_list, return_tensors="pt", padding=True,
              truncation=True, max_length=MAX_LEN)
    return enc["input_ids"].to(DEVICE), enc["attention_mask"].to(DEVICE)


def apply_mask_noise(input_ids: torch.Tensor, tok) -> torch.Tensor:
    """Replace ~MASK_PROB of non-special tokens with the mask token."""
    ids = input_ids.clone()
    mask_id = tok.mask_token_id
    special = set(tok.all_special_ids)
    prob = torch.full(ids.shape, MASK_PROB, device=ids.device)
    draw = torch.bernoulli(prob).bool()
    for sid in special:
        draw &= ids != sid
    ids[draw] = mask_id
    return ids


# ----------------------------- log-prob of a candidate given a seed -----------------------------
def sequence_logprob(model, tok, seed_ids, seed_mask, cand_ids) -> torch.Tensor:
    """Length-normalised log p(candidate | seed) for a batch.
    seed_ids: (B,S)  cand_ids: (B,T)  -> returns (B,)"""
    pad_id = tok.pad_token_id
    labels = cand_ids.clone()
    out = model(input_ids=seed_ids, attention_mask=seed_mask, labels=labels)
    logp = F.log_softmax(out.logits, dim=-1)               # (B,T,V)
    tok_logp = logp.gather(-1, cand_ids.unsqueeze(-1)).squeeze(-1)  # (B,T)
    valid = (cand_ids != pad_id).float()
    seq_lp = (tok_logp * valid).sum(-1) / (valid.sum(-1).clamp(min=1) ** LENGTH_PENALTY)
    return seq_lp


# ----------------------------- losses -----------------------------
def rank_loss_for_seed(seq_lp: torch.Tensor, scores: List[float]) -> torch.Tensor:
    """BRIO-style pairwise margin loss for ONE seed's candidate set.
    seq_lp: (K,) log-probs, aligned to candidates already sorted by score DESC.
    Higher score should get higher log-prob."""
    k = seq_lp.shape[0]
    loss = seq_lp.new_zeros(())
    n_pairs = 0
    for i in range(k):
        for j in range(i + 1, k):
            if scores[i] <= scores[j]:
                continue  # only enforce strict score ordering
            margin = MARGIN * (j - i)
            loss = loss + F.relu(seq_lp[j] - seq_lp[i] + margin)
            n_pairs += 1
    if n_pairs > 0:
        loss = loss / n_pairs
    return loss


def ce_reconstruction_loss(model, tok, seed_ids, seed_mask) -> torch.Tensor:
    """Label-smoothed denoising reconstruction of the seed (the normalisation
    anchor, exactly the role the MolGen paper gives the CE term)."""
    pad_id = tok.pad_token_id
    masked = apply_mask_noise(seed_ids, tok)
    labels = seed_ids.clone()
    labels[labels == pad_id] = -100
    out = model(input_ids=masked, attention_mask=seed_mask, labels=labels)
    logits = out.logits
    ce = F.cross_entropy(
        logits.view(-1, logits.size(-1)),
        labels.view(-1),
        ignore_index=-100,
        label_smoothing=LABEL_SMOOTHING,
    )
    return ce


# ----------------------------- candidate pre-generation (frozen) -----------------------------
def candidate_cache_path() -> str:
    seed_hash = hashed_seed("cands", TRAINING_SEED, N_SEED_MOLECULES,
                            NUM_CANDIDATES, SCORING, GEN_TEMPERATURE)
    return os.path.join(CACHE_DIR, f"cands_{seed_hash}.json")


def prepare_candidates(model, tok, seed_selfies: List[str]) -> List[Dict]:
    """Generate NUM_CANDIDATES per seed once, score, sort by score DESC, freeze.
    A seed is kept only if it yields >= 2 valid candidates with a non-degenerate
    score spread (so a ranking actually exists)."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = candidate_cache_path()
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)

    set_all_seeds(hashed_seed("gen", TRAINING_SEED, NUM_CANDIDATES))
    model.eval()
    frozen: List[Dict] = []
    with torch.no_grad():
        for s in seed_selfies:
            seed_ids, seed_mask = encode(tok, [s])
            masked = apply_mask_noise(seed_ids, tok)
            gen = model.generate(
                input_ids=masked,
                attention_mask=seed_mask,
                do_sample=True,
                temperature=GEN_TEMPERATURE,
                num_return_sequences=NUM_CANDIDATES,
                max_length=MAX_LEN,
                num_beams=1,
            )
            cand_selfies = tok.batch_decode(gen, skip_special_tokens=True)
            scored = []
            seen = set()
            for cs in cand_selfies:
                smi = selfies_str_to_smiles(cs)
                if smi is None or smi in seen:
                    continue
                sc = score_smiles(smi)
                if sc is None:
                    continue
                seen.add(smi)
                scored.append((cs, smi, sc))
            if len(scored) < 2:
                continue
            if max(x[2] for x in scored) - min(x[2] for x in scored) < 1e-6:
                continue
            scored.sort(key=lambda x: x[2], reverse=True)  # best first
            frozen.append({
                "seed_selfies": s,
                "cand_selfies": [x[0] for x in scored],
                "cand_smiles":  [x[1] for x in scored],
                "scores":       [x[2] for x in scored],
            })
    with open(path, "w") as f:
        json.dump(frozen, f)
    print(f"[candidates] kept {len(frozen)}/{len(seed_selfies)} seeds "
          f"({NUM_CANDIDATES} generated each) -> {path}")
    return frozen


# ----------------------------- monitoring suite -----------------------------
def diagnostic_correlation(model, tok, frozen: List[Dict], n: int = 100) -> float:
    """Headline metric: mean per-seed Spearman(score, current-model log-prob)
    across candidates. This is the quantity that exposed the ceiling before."""
    model.eval()
    rhos = []
    with torch.no_grad():
        for item in frozen[:n]:
            seed_ids, seed_mask = encode(tok, [item["seed_selfies"]])
            k = len(item["cand_selfies"])
            seed_ids = seed_ids.repeat(k, 1)
            seed_mask = seed_mask.repeat(k, 1)
            cand_ids, _ = encode(tok, item["cand_selfies"])
            # align padded lengths between seed batch and candidate batch is fine;
            # encoder and decoder are independent tensors here.
            lp = sequence_logprob(model, tok, seed_ids, seed_mask, cand_ids)
            lp = lp.detach().cpu().numpy()
            sc = np.array(item["scores"])
            if np.std(lp) < 1e-8 or np.std(sc) < 1e-8:
                continue
            rho, _ = spearmanr(sc, lp)
            if not np.isnan(rho):
                rhos.append(rho)
    return float(np.mean(rhos)) if rhos else float("nan")


def generation_metrics(model, tok, eval_selfies: List[str], moses_canon: set,
                       gen_seeds: Optional[List[int]] = None,
                       n_seeds: Optional[int] = None,
                       tag: str = "") -> Dict[str, float]:
    """Monitoring probe. Generate one molecule per eval seed across gen_seeds and
    report validity, uniqueness, novelty, internal diversity, scaffold diversity,
    and the full distribution (mean, std, p10/p50/p90) of each target property.

    gen_seeds : which generation seeds to use (defaults to all GENERATION_SEEDS;
                pass a single seed for a cheap through-training snapshot).
    n_seeds   : cap on the number of eval seed molecules (None uses all).
    tag       : if set and DUMP_DISTRIBUTIONS, raw property arrays are saved to
                DIST_DIR/<tag>.npz for distribution plots.
    """
    gen_seeds = gen_seeds if gen_seeds is not None else GENERATION_SEEDS
    seeds = eval_selfies if n_seeds is None else eval_selfies[:n_seeds]

    model.eval()
    all_valid: List[str] = []
    total = 0
    with torch.no_grad():
        for gseed in gen_seeds:
            set_all_seeds(hashed_seed("evalgen", gseed, TRAINING_SEED))
            for s in seeds:
                seed_ids, seed_mask = encode(tok, [s])
                masked = apply_mask_noise(seed_ids, tok)
                gen = model.generate(
                    input_ids=masked, attention_mask=seed_mask,
                    do_sample=True, temperature=1.0,
                    num_return_sequences=1, max_length=MAX_LEN, num_beams=1,
                )
                total += 1
                cs = tok.batch_decode(gen, skip_special_tokens=True)[0]
                smi = selfies_str_to_smiles(cs)
                if smi is not None:
                    all_valid.append(smi)

    validity = len(all_valid) / max(total, 1)
    uniq = set(all_valid)
    uniqueness = len(uniq) / max(len(all_valid), 1)
    novelty = sum(1 for s in uniq if s not in moses_canon) / max(len(uniq), 1)

    scaffolds = {murcko_scaffold(s) for s in uniq}
    scaffolds.discard(None)
    scaffold_div = len(scaffolds) / max(len(uniq), 1)

    # internal diversity on a subsample (pairwise cost is quadratic)
    sub = list(uniq)
    if len(sub) > 1000:
        sub = random.Random(hashed_seed("intdiv", TRAINING_SEED)).sample(sub, 1000)
    fps = [fp for fp in (morgan_fp(s) for s in sub) if fp is not None]
    if len(fps) > 1:
        sims: List[float] = []
        for i in range(len(fps)):
            sims.extend(DataStructs.BulkTanimotoSimilarity(fps[i], fps[i + 1:]))
        int_div = 1.0 - (float(np.mean(sims)) if sims else 0.0)
    else:
        int_div = float("nan")

    # target-property DISTRIBUTIONS, not just means
    props = [p for p in (target_properties(s) for s in uniq) if p is not None]
    def col(key):
        return np.array([p[key] for p in props if not np.isnan(p[key])], dtype=float)

    out: Dict[str, float] = {
        "validity": validity,
        "uniqueness": uniqueness,
        "novelty": novelty,
        "scaffold_div": scaffold_div,
        "int_div": int_div,
        "n_valid": float(len(all_valid)),
    }
    for key in ["fsp3", "qed", "esol"]:
        a = col(key)
        if a.size:
            out[f"mean_{key}"] = float(np.mean(a))
            out[f"std_{key}"] = float(np.std(a))
            out[f"p10_{key}"] = float(np.percentile(a, 10))
            out[f"p50_{key}"] = float(np.percentile(a, 50))
            out[f"p90_{key}"] = float(np.percentile(a, 90))
        else:
            for stat in ["mean", "std", "p10", "p50", "p90"]:
                out[f"{stat}_{key}"] = float("nan")

    if DUMP_DISTRIBUTIONS and tag:
        os.makedirs(DIST_DIR, exist_ok=True)
        np.savez(os.path.join(DIST_DIR, f"{tag}.npz"),
                 fsp3=col("fsp3"), qed=col("qed"), esol=col("esol"))
    return out


# ----------------------------- results logging -----------------------------
def log_row(row: Dict) -> None:
    os.makedirs(os.path.dirname(RESULTS_CSV), exist_ok=True)
    write_header = not os.path.exists(RESULTS_CSV)
    with open(RESULTS_CSV, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            w.writeheader()
        w.writerow(row)


def log_trajectory(cell: str, row: Dict) -> None:
    """Append one through-training snapshot to the per-cell trajectory CSV."""
    os.makedirs(TRAJ_DIR, exist_ok=True)
    path = os.path.join(TRAJ_DIR, f"{cell}.csv")
    write_header = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            w.writeheader()
        w.writerow(row)


def monitor_point(model, tok, frozen, eval_seeds, moses_canon,
                  step: int, cell: str) -> Dict:
    """Cheap snapshot used to watch selection metrics move DURING training:
    the score/log-prob correlation plus diversity, validity, and property means
    on a single generation seed. Written to the cell trajectory so a config that
    lifts the score while diversity slides is visible before the run ends."""
    rho = diagnostic_correlation(model, tok, frozen, n=MONITOR_N_SEEDS)
    gm = generation_metrics(model, tok, eval_seeds, moses_canon,
                            gen_seeds=[GENERATION_SEEDS[0]], n_seeds=MONITOR_N_SEEDS)
    row = {
        "step": step,
        "rho": round(rho, 4),
        "int_div": round(gm["int_div"], 4),
        "scaffold_div": round(gm["scaffold_div"], 4),
        "validity": round(gm["validity"], 4),
        "uniqueness": round(gm["uniqueness"], 4),
        "novelty": round(gm["novelty"], 4),
        "mean_qed": round(gm["mean_qed"], 4),
        "mean_fsp3": round(gm["mean_fsp3"], 4),
        "mean_esol": round(gm["mean_esol"], 4),
    }
    log_trajectory(cell, row)
    print(f"[step {step}] rho={row['rho']} int_div={row['int_div']} "
          f"scaf={row['scaffold_div']} valid={row['validity']} qed={row['mean_qed']}")
    return row


def print_run_summary(row: Dict) -> None:
    """Readable per-run results block, printed as soon as a single model finishes:
    that model's own properties, pre-feedback baseline next to final."""
    def g(k):
        v = row.get(k, float("nan"))
        return float(v) if isinstance(v, (int, float)) else float("nan")

    print("\n" + "=" * 58)
    print(f" RESULTS  {row['cell']}")
    print(f" alpha={row['alpha']}  lr={row['learning_rate']}  "
          f"cand={row['num_candidates']}  seed={row['training_seed']}  "
          f"scoring={row['scoring']}  ({row['wall_min']} min)")
    print("=" * 58)
    print(f" {'metric':<24}{'baseline':>14}{'final':>14}")
    print(" " + "-" * 52)
    scalar_rows = [
        ("score/log-prob corr", "base_rho", "fin_rho"),
        ("validity", "base_validity", "fin_validity"),
        ("uniqueness", "base_uniqueness", "fin_uniqueness"),
        ("novelty", "base_novelty", "fin_novelty"),
        ("internal diversity", "base_int_div", "fin_int_div"),
        ("scaffold diversity", "base_scaffold_div", "fin_scaffold_div"),
    ]
    for label, bk, fk in scalar_rows:
        print(f" {label:<24}{g(bk):>14.4f}{g(fk):>14.4f}")
    print(" " + "-" * 52)
    for prop in ["fsp3", "qed", "esol"]:
        b = f"{g('base_mean_' + prop):.3f} ({g('base_std_' + prop):.3f})"
        f = f"{g('fin_mean_' + prop):.3f} ({g('fin_std_' + prop):.3f})"
        print(f" {prop.upper() + ' mean (std)':<24}{b:>14}{f:>14}")
    print("=" * 58 + "\n")


# ----------------------------- training loop -----------------------------
def train_one_cell():
    cell = f"alpha{ALPHA}_lr{LEARNING_RATE}_cand{NUM_CANDIDATES}_ts{TRAINING_SEED}_{SCORING}"
    print(f"\n=== training cell: {cell} ===")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    train_seeds, eval_seeds, moses_canon = load_seed_pool()
    model, tok = load_model_and_tokenizer()

    frozen = prepare_candidates(model, tok, train_seeds)
    if len(frozen) == 0:
        print("[abort] no usable candidate sets; check base model / scoring.")
        return

    # ---- pre-feedback baseline for this cell (the reference for "reduced
    # ---- diversity"): correlation AND a full monitoring generation ----
    base_rho = diagnostic_correlation(model, tok, frozen)
    print(f"[step 0] score/log-prob Spearman = {base_rho:.3f}  (watch this)")
    base_gm = generation_metrics(model, tok, eval_seeds, moses_canon,
                                 tag=f"{cell}__baseline")
    print(f"[step 0] baseline int_div={base_gm['int_div']:.4f} "
          f"scaf={base_gm['scaffold_div']:.4f} qed={base_gm['mean_qed']:.4f}")

    optim = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    order = list(range(len(frozen)))
    total_steps = EPOCHS * ((len(frozen) + TRAIN_BATCH - 1) // TRAIN_BATCH)
    monitor_interval = max(1, total_steps // max(N_MONITOR_POINTS, 1))
    step = 0
    t0 = time.time()

    for epoch in range(EPOCHS):
        set_all_seeds(hashed_seed("train", ALPHA, LEARNING_RATE, TRAINING_SEED, epoch))
        random.shuffle(order)
        for bstart in range(0, len(order), TRAIN_BATCH):
            model.train()
            batch = order[bstart:bstart + TRAIN_BATCH]
            optim.zero_grad()

            ce_accum = torch.zeros((), device=DEVICE)
            rank_accum = torch.zeros((), device=DEVICE)

            for bi in batch:
                item = frozen[bi]
                seed_ids, seed_mask = encode(tok, [item["seed_selfies"]])

                # CE anchor (denoising reconstruction of the seed)
                ce = ce_reconstruction_loss(model, tok, seed_ids, seed_mask)

                # rank loss over this seed's candidate set
                k = len(item["cand_selfies"])
                seed_rep = seed_ids.repeat(k, 1)
                seed_mask_rep = seed_mask.repeat(k, 1)
                cand_ids, _ = encode(tok, item["cand_selfies"])
                lp = sequence_logprob(model, tok, seed_rep, seed_mask_rep, cand_ids)
                rl = rank_loss_for_seed(lp, item["scores"])

                ce_accum = ce_accum + ce
                rank_accum = rank_accum + rl

            ce_accum = ce_accum / len(batch)
            rank_accum = rank_accum / len(batch)
            loss = ce_accum + ALPHA * rank_accum
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optim.step()
            step += 1

            # print the loss every step (short training runs), and take a light
            # monitoring snapshot at the spread-out monitor points
            print(f"[step {step}/{total_steps}] loss={loss.item():.4f} "
                  f"ce={ce_accum.item():.4f} rank={rank_accum.item():.4f}")
            if step % monitor_interval == 0 and step < total_steps:
                monitor_point(model, tok, frozen, eval_seeds, moses_canon, step, cell)

    # ---- final full monitoring suite ----
    final_rho = diagnostic_correlation(model, tok, frozen)
    fin_gm = generation_metrics(model, tok, eval_seeds, moses_canon,
                                tag=f"{cell}__final")

    row = {
        "cell": cell,
        "alpha": ALPHA,
        "learning_rate": LEARNING_RATE,
        "num_candidates": NUM_CANDIDATES,
        "training_seed": TRAINING_SEED,
        "scoring": SCORING,
        "base_rho": round(base_rho, 4),
        "fin_rho": round(final_rho, 4),
        "delta_rho": round(final_rho - base_rho, 4),
        "wall_min": round((time.time() - t0) / 60.0, 1),
    }
    # baseline and final metrics side by side so retention / deltas are explicit
    for k, v in base_gm.items():
        row[f"base_{k}"] = round(v, 4) if isinstance(v, float) else v
    for k, v in fin_gm.items():
        row[f"fin_{k}"] = round(v, 4) if isinstance(v, float) else v
    log_row(row)
    print_run_summary(row)

    ckpt = os.path.join(OUTPUT_DIR, cell)
    model.save_pretrained(ckpt)
    tok.save_pretrained(ckpt)
    print(f"[saved] {ckpt}")
    return row


# ----------------------------- config selection -----------------------------
def select_best(results_csv: Optional[str] = None,
                min_div_retention: Optional[float] = None):
    """Rank cells for selection, not on training loss. A cell must first hold on
    to its own pre-feedback diversity (internal AND scaffold) above the retention
    floor; only among those survivors are cells ordered by movement in the
    score/log-prob correlation and then by QED gain. Returns the ranked frame."""
    results_csv = results_csv or RESULTS_CSV
    floor = DIV_RETENTION_MIN if min_div_retention is None else min_div_retention
    df = pd.read_csv(results_csv)

    df["int_div_retention"] = df["fin_int_div"] / df["base_int_div"]
    df["scaf_div_retention"] = df["fin_scaffold_div"] / df["base_scaffold_div"]
    df["diversity_ok"] = ((df["int_div_retention"] >= floor)
                          & (df["scaf_div_retention"] >= floor))

    df["d_qed"] = df["fin_mean_qed"] - df["base_mean_qed"]
    df["d_fsp3"] = df["fin_mean_fsp3"] - df["base_mean_fsp3"]
    df["d_validity"] = df["fin_validity"] - df["base_validity"]

    ranked = df.sort_values(
        by=["diversity_ok", "delta_rho", "d_qed"],
        ascending=[False, False, False],
    ).reset_index(drop=True)

    cols = ["cell", "alpha", "learning_rate", "diversity_ok",
            "delta_rho", "int_div_retention", "scaf_div_retention",
            "d_qed", "d_fsp3", "fin_validity", "fin_novelty"]
    print("Cells failing the diversity floor are pushed to the bottom "
          "regardless of score movement:")
    print(ranked[cols].to_string(index=False))
    return ranked


# ----------------------------- self-test for the rank loss -----------------------------
def _selftest_rank_loss():
    """Cheap sanity check with no model needed. Run once before the expensive
    parts so you trust the pairwise margin logic."""
    scores = [0.9, 0.5, 0.1]  # sorted desc, as prepare_candidates guarantees

    # model already correct (better score -> higher log-prob): loss ~ 0
    good = torch.tensor([-1.0, -2.0, -3.0])
    assert rank_loss_for_seed(good, scores).item() < 1e-4, "expected ~0 loss"

    # model inverted (worse score -> higher log-prob): loss > 0
    bad = torch.tensor([-3.0, -2.0, -1.0])
    assert rank_loss_for_seed(bad, scores).item() > 0.0, "expected positive loss"

    # ties in score are not penalised
    flat_scores = [0.5, 0.5, 0.5]
    any_lp = torch.tensor([-1.0, -2.0, -3.0])
    assert rank_loss_for_seed(any_lp, flat_scores).item() == 0.0, "ties must not penalise"
    print("[selftest] rank loss OK")


# ----------------------------- entry point -----------------------------
if __name__ == "__main__":
    _selftest_rank_loss()
    # In Colab, run: from google.colab import drive; drive.mount('/content/drive')
    train_one_cell()


# ----------------------------- optional sweep driver -----------------------------
# Screening grid (one training seed). Candidates are regenerated only when
# NUM_CANDIDATES changes, so the 12 cells below share the same frozen set.
#
#   for lr in [1e-5, 3e-5, 1e-4]:
#       for a in [1, 3, 10, 30]:
#           globals()["LEARNING_RATE"] = lr
#           globals()["ALPHA"] = float(a)
#           train_one_cell()
#
# After screening, rank rows in RESULTS_CSV by delta_rho first, then break ties
# on mean_qed / composite property means, and DISCARD any cell whose int_div or
# scaffold_div collapses relative to the pretrained/standard baseline. Re-run the
# survivors at NUM_CANDIDATES = 8 and 32, then repeat with 3-5 training seeds.
