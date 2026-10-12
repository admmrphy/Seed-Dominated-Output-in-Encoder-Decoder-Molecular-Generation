"""
generation.py
=============

Molecule generation driver for the MolGen-large property-weighting study.

Design points implemented and marked in the code:

  (1) SEEDING VIA numpy.random.SeedSequence  (no MD5)
      Each (run, generation-seed) cell derives one SeedSequence from the fixed
      key [train_seed, gen_seed]. NumPy, PyTorch and Python's random are seeded
      from that single sequence, so streams are decorrelated across cells and
      stable across Python versions. The key excludes the model CONDITION, so a
      matched (run, gen seed) cell sees the same encoder pool and sampling state
      under every condition. Between-condition differences are attributable to
      the model, not the seeds.

  (2) DECOUPLED SAMPLING (fixed budget; uniqueness reported, not enforced)
      Each cell draws exactly N_PER_CELL encoder seeds and generates one output
      each. EVERY draw is written to disk, including invalid decodes and exact
      duplicates. Validity, uniqueness and held-out novelty are computed and
      reported afterwards; they are not used to filter the retained sample.

Outputs per cell:
  - a per-molecule CSV under  OUTPUT_DIR/<model_version>/gen<seed>.csv
    with the full property panel (one row per generated draw), and
  - one appended row in       OUTPUT_DIR/generation_summary.csv
    with the aggregate statistics.

Tanimoto near-neighbour and FCD analysis are done separately, later, from the
per-molecule CSVs written here.

Environment: Python 3.9, transformers==4.40.2, peft==0.10.0, torch==2.1.0.
"""

import os
import random
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, Crippen, QED, rdMolDescriptors
from rdkit.Chem.Scaffolds import MurckoScaffold
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

import selfies as sf

RDLogger.DisableLog("rdApp.*")

# ===========================================================================
# CONFIG
# ===========================================================================

BASE_MODEL = "zjunlp/MolGen-large"

# MOSES corpus that encoder seeds are drawn from (needs a SELFIES column; the
# loader tolerates the historical 'SEFLIES' misspelling).
MOSES_CSV = "/content/drive/MyDrive/MOSESproject/moses_full_properties_with_selfies.csv"

# Held-out MOSES split used for the exact-match novelty metric ("Uniqueness vs
# held-out"). Point this at the official MOSES test split (176,075 molecules).
# If left as None or missing, the held-out novelty column is written as NaN.
MOSES_HELDOUT_CSV: Optional[str] = "/content/drive/MyDrive/MOSESproject/moses_test.csv"

# ---------------------------------------------------------------------------
# (3) ENCODER CORRUPTION  --  the fix for seed reconstruction
# ---------------------------------------------------------------------------
# MolGen-large is pretrained by "mapping corrupted SELFIES to their original
# forms" (its own model card). Handed a CLEAN encoder input, the task has one
# correct answer: the input itself. v2 fed clean seeds and the full fine-tuned
# model returned the seed verbatim on 1000/1000 draws.
#
# Masking restores the pretraining condition, so the decoder must choose what
# fills the masked positions, and that choice is where a fine-tuning signal can
# express itself. This mirrors feedback_hook2._mask_noise, whose own comment
# reads: "required for candidate diversity, otherwise every candidate collapses
# to the seed".
#
# MASK_PROB is a direct dial on seed dominance and is meant to be swept:
#   0.00 -> clean seed, ~100% reconstruction (the v2 result)
#   0.15 -> the BART denoising probability used during fine-tuning
#   1.00 -> seed carries no information
# Sweeping it and plotting seed-copy rate + output-to-seed Tanimoto against it
# turns the seed-dominance claim into a measured dose-response curve.
MASK_PROB: float = 0.15

# Sampling filters. MolGen's own published examples use top_k=30, top_p=0.75.
# NOTE: v2's comment claimed "pure temperature sampling, no top-k/p", which was
# wrong: HuggingFace GenerationConfig defaults to top_k=50, so a top_k was
# always being applied, just an unstated one. Setting these explicitly makes the
# protocol match MolGen's and stops the unfiltered tail from producing the
# fragment outputs (32-48% acyclic) seen in the v2 MPT and feedback conditions.
TOP_K: int = 30
TOP_P: float = 0.75

# MolGen's own examples pass min_length=5. With masking on, nothing otherwise
# stops a very short decode, which is one route into the fragment regime (the
# v2 feedback condition produced 22% of outputs under four heavy atoms).
MIN_LENGTH: int = 5

# A NEW tree, not the v2 one: the resume check skips any existing cell with
# >= N_PER_CELL rows, so writing into the old folder would silently keep the
# seed-copied data rather than regenerating it.
#
# The mask level is a subfolder because cells are keyed by (model_version,
# gen_seed). Without it, a second MASK_PROB would find gen201.csv already full,
# skip every cell, and hand back the FIRST level's molecules labelled with the
# second level's mask_prob.
_RESULTS_ROOT = "/content/drive/MyDrive/MOSESproject/results/resultsv2"
OUTPUT_DIR = os.path.join(_RESULTS_ROOT, f"mask{int(round(MASK_PROB * 100)):03d}")
# One summary across all mask levels so the sweep plots from a single file.
# Rows are keyed by (model_version, gen_seed, mask_prob).
SUMMARY_CSV = os.path.join(_RESULTS_ROOT, "generation_summary.csv")

CHECKPOINT_ROOT = "/content/drive/MyDrive/MOSESproject/checkpoints"

# Three "model versions" per condition, labelled run0/run1/run2 on Drive.
# Run index -> training seed keys the SeedSequence (disjoint from generation).
#
# All three runs, matching every other condition: each run index is a model
# trained at its OWN seed, so the across-run spread carries training
# variability rather than only generation noise. This requires three feedback
# checkpoints (see CONDITIONS["feedback_composite"]["paths"]).
RUN_INDICES: List[int] = [0, 1, 2]
RUN_TO_TRAIN_SEED: Dict[int, int] = {0: 11, 1: 42, 2: 73}

# Generation seeds: fixed in advance, new (unused in training), applied
# identically across every condition. 201..210 == 10 seeds. For 11 seeds
# (including 211), change to range(201, 212).
GENERATION_SEEDS: List[int] = list(range(201, 211))

TEMPERATURE: float = 1.0

# Raw draws per (run, gen seed) cell.
# 1000 x 3 runs x 10 gen seeds = 30,000 per condition, identical to every other
# condition: same cell size, same encoder pool per (train_seed, gen_seed) key,
# so the feedback condition is directly comparable to standard_run* and
# weighted_run* cell for cell.
N_PER_CELL: int = 1000

MAX_LENGTH: int = 128
BATCH_SIZE: int = 64
USE_FP16: bool = True

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Condition registry. "type": "full" -> full fine-tuned Seq2Seq checkpoint;
# "peft" -> prefix-tuning adapter on the frozen base. "prefix" is the folder
# stem under CHECKPOINT_ROOT; the run index is appended, e.g. standard_run0.
#
# "paths": an optional {run_idx: absolute_path} map overriding the
# CHECKPOINT_ROOT/<prefix><run_idx> convention. The feedback checkpoints live
# under models/ rather than checkpoints/, so they need it.
#
# It is a MAP, not one path, on purpose. A single shared path would load the
# same weights for run0/1/2 and report three identical outputs as three
# independent replicates, inflating the condition's apparent reliability while
# its across-run SD collapsed to generation noise. Each run index must point at
# a checkpoint trained at ITS OWN seed (see RUN_TO_TRAIN_SEED).
#
# Only run1 (train seed 42) exists so far. Runs 0 and 2 require re-running the
# feedback training at TRAIN_SEED = 11 and 73 and saving to these paths.
_FB = "/content/drive/MyDrive/MOSESproject/models"
#CONDITIONS: Dict[str, Dict] = {
#    "feedback_composite": {
#        "type": "full",
#        "prefix": "feedback_composite_run",
#        "paths": {
#            0: f"{_FB}/molgen_feedback_composite_10epoch_seedpool2000_seed11",
#            1: f"{_FB}/molgen_feedback_composite_10epoch_seedpool2000",
#            2: f"{_FB}/molgen_feedback_composite_10epoch_seedpool2000_seed73",
#        },
#    },
#}

# The original four, kept for reference. Restore any of these to regenerate
# them; they resolve via CHECKPOINT_ROOT and need no "paths" map.
_ORIGINAL_CONDITIONS: Dict[str, Dict] = {
    "finetuned_standard": {"type": "full", "prefix": "standard_run"},
    "finetuned_weighted": {"type": "full", "prefix": "weighted_run"},
    "mpt_standard":       {"type": "peft", "prefix": "mpt_standard_run"},
    "mpt_weighted":       {"type": "peft", "prefix": "mpt_weighted_run"},
}

# ===========================================================================
# PROPERTY DEFINITIONS TO RECONCILE WITH YOUR EXISTING PIPELINE
# ---------------------------------------------------------------------------
# The functions below marked [RECONCILE] have more than one reasonable
# definition. They MUST match the code that built
# moses_full_properties_with_selfies.csv, or generated values will not be
# comparable to your MOSES reference or your weighted-pool thresholds. Confirm
# each against add_averages.py (or paste that code in and swap these out).
#   - esol_log_sol       (ESOL coefficients / aromatic term)
#   - solubility_category (bin edges)
#   - lipinski_pass       (violation convention)
#   - bbb_likelihood      (rule and output categories)
#   - is_degenerate       (definition of a degenerate output)
# The plain RDKit descriptors (MW, LogP, HBD/HBA, TPSA, QED, Fsp3, ring counts,
# heavy atoms, charge, stereocentres, refractivity, Murcko) are unambiguous.
# ===========================================================================

def esol_log_sol(mol, mw: float, logp: float, rot: int) -> float:
    """[RECONCILE] ESOL aqueous solubility (log mol/L).

    Uses the aromatic-ring-count form per project convention (memory:
    'use CalcNumAromaticRings, not aromatic atom proportion'). Confirm the
    coefficients match add_averages.py; the published Delaney (2004) constants
    are used here as a starting point.
    """
    aromatic_rings = rdMolDescriptors.CalcNumAromaticRings(mol)
    return 0.16 - 0.63 * logp - 0.0062 * mw + 0.066 * rot - 0.74 * aromatic_rings


def solubility_category(log_s: float) -> str:
    """[RECONCILE] Coarse solubility class from log S (log mol/L)."""
    if log_s >= 0:
        return "Highly soluble"
    if log_s >= -2:
        return "Soluble"
    if log_s >= -4:
        return "Slightly soluble"
    if log_s >= -6:
        return "Poorly soluble"
    return "Insoluble"


def lipinski_pass(mw: float, logp: float, hbd: int, hba: int) -> bool:
    """[RECONCILE] Lipinski Ro5. Compliant if fewer than two violations."""
    violations = (mw > 500) + (logp > 5) + (hbd > 5) + (hba > 10)
    return violations < 2


def bbb_likelihood(mw: float, tpsa: float, hbd: int) -> str:
    """[RECONCILE] Blood-brain-barrier permeability heuristic (High/Medium/Low).

    Simple TPSA/MW/HBD rule; replace with your project's BBB definition.
    """
    if tpsa <= 60 and mw <= 450 and hbd <= 3:
        return "High"
    if tpsa <= 90 and mw <= 500:
        return "Medium"
    return "Low"


def is_degenerate(mol, canonical_out: str, seed_canonical: Optional[str]) -> bool:
    """[RECONCILE] Flag a degenerate / trivial output.

    Current definition: fewer than four heavy atoms, OR an empty Murcko
    scaffold, OR the output is identical to the encoder seed molecule. Adjust
    to whatever your pipeline means by 'degenerate'.
    """
    if mol.GetNumHeavyAtoms() < 4:
        return True
    try:
        if MurckoScaffold.MurckoScaffoldSmiles(mol=mol) == "":
            return True
    except Exception:
        pass
    if seed_canonical is not None and canonical_out == seed_canonical:
        return True
    return False


def compute_properties(canonical_smiles: str,
                       seed_canonical: Optional[str]) -> Dict:
    """Full property panel for one valid molecule. Keys match requested headers.

    Returns a dict of NaN-filled values if the molecule cannot be parsed.
    """
    nan_row = {k: np.nan for k in PROPERTY_COLUMNS}
    mol = Chem.MolFromSmiles(canonical_smiles)
    if mol is None:
        return nan_row

    try:
        mw = Descriptors.MolWt(mol)
        logp = Crippen.MolLogP(mol)
        hbd = rdMolDescriptors.CalcNumHBD(mol)
        hba = rdMolDescriptors.CalcNumHBA(mol)
        rot = rdMolDescriptors.CalcNumRotatableBonds(mol)
        tpsa = rdMolDescriptors.CalcTPSA(mol)
        qed = QED.qed(mol)
        arom_rings = rdMolDescriptors.CalcNumAromaticRings(mol)
        fsp3 = rdMolDescriptors.CalcFractionCSP3(mol)
        total_rings = rdMolDescriptors.CalcNumRings(mol)
        heavy = mol.GetNumHeavyAtoms()
        charge = Chem.GetFormalCharge(mol)

        Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
        stereo = (rdMolDescriptors.CalcNumAtomStereoCenters(mol)
                  + rdMolDescriptors.CalcNumUnspecifiedAtomStereoCenters(mol))

        heteroatoms = sum(1 for a in mol.GetAtoms()
                          if a.GetAtomicNum() not in (1, 6))
        hetero_rings = rdMolDescriptors.CalcNumHeterocycles(mol)
        arom_hetero = rdMolDescriptors.CalcNumAromaticHeterocycles(mol)
        mr = Crippen.MolMR(mol)
        murcko = MurckoScaffold.MurckoScaffoldSmiles(mol=mol)

        log_s = esol_log_sol(mol, mw, logp, rot)

        return {
            "MW": mw,
            "LogP": logp,
            "HBD": hbd,
            "HBA": hba,
            "RotBonds": rot,
            "TPSA": tpsa,
            "QED": qed,
            "Aromatic Rings": arom_rings,
            "Fsp3": fsp3,
            "Total Rings": total_rings,
            "Heavy Atoms": heavy,
            "Formal Charge": charge,
            "Stereo centres": stereo,
            "Heteroatoms": heteroatoms,
            "Heterocyclic Rings": hetero_rings,
            "Aromatic Heterocycles": arom_hetero,
            "Molar Refractivity": mr,
            "Solubility_logmolL": log_s,
            "Solubility_Category": solubility_category(log_s),
            "Lipinski_Pass": lipinski_pass(mw, logp, hbd, hba),
            "BBB_Likelihood": bbb_likelihood(mw, tpsa, hbd),
            "Murcko_Scaffold": murcko,
            "degenerate": is_degenerate(mol, canonical_smiles, seed_canonical),
        }
    except Exception:
        return nan_row


PROPERTY_COLUMNS = [
    "MW", "LogP", "HBD", "HBA", "RotBonds", "TPSA", "QED", "Aromatic Rings",
    "Fsp3", "Total Rings", "Heavy Atoms", "Formal Charge", "Stereo centres",
    "Heteroatoms", "Heterocyclic Rings", "Aromatic Heterocycles",
    "Molar Refractivity", "Solubility_logmolL", "Solubility_Category",
    "Lipinski_Pass", "BBB_Likelihood", "Murcko_Scaffold", "degenerate",
]


# ===========================================================================
# (1) SEEDING  --  numpy.random.SeedSequence, no MD5
# ===========================================================================

def make_cell_rng(train_seed: int, gen_seed: int):
    """Return (numpy Generator, 64-bit int) decorrelated per cell and identical
    across conditions for the same (train_seed, gen_seed)."""
    ss = np.random.SeedSequence([int(train_seed), int(gen_seed)])
    np_rng = np.random.default_rng(ss)
    seed64 = int(ss.generate_state(1, dtype=np.uint64)[0])
    torch.manual_seed(seed64)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed64)
    random.seed(seed64)
    return np_rng, seed64


# ===========================================================================
# DATA / DECODING HELPERS
# ===========================================================================

def _find_smiles_column(df: pd.DataFrame) -> Optional[str]:
    for c in ("canonical_smiles", "SMILES", "smiles", "Smiles"):
        if c in df.columns:
            return c
    return None


def load_seed_selfies(csv_path: str, heldout: Optional[set] = None) -> List[str]:
    """Encoder seed pool, guaranteed disjoint from the held-out MOSES test split.

    The test split must stay COMPLETELY unseen so it remains a valid reference
    for the final novelty / SNN / FCD evaluation. If a test molecule were used
    as an encoder seed, and the model reconstructs or near-copies its seed (which
    is what MolGen-large is pretrained to do), the model would emit a test
    molecule and the evaluation would be scoring against data it was handed.

    Two independent guards, because relying on one is how this goes wrong:
      1. If the CSV carries a SPLIT column, keep only rows marked train.
      2. Regardless of (1), drop any row whose canonical SMILES appears in the
         held-out set. This is the guard that actually enforces the guarantee,
         since it does not depend on the CSV carrying split metadata at all.

    A removal count of 0 at step (2) means the CSV was already train-only, which
    is the expected case if it holds ~1.76M rows (1,936,962 MOSES total minus the
    176,075-molecule test split).

    Assumes the CSV's SMILES column is already canonical, per the project's
    preprocessing. A sample is round-tripped through RDKit to check that.
    """
    df = pd.read_csv(csv_path)
    n_start = len(df)
    print(f"Seed CSV: {n_start} rows")

    sel_col = None
    for candidate in ("SELFIES", "SEFLIES", "selfies"):
        if candidate in df.columns:
            sel_col = candidate
            break
    if sel_col is None:
        raise KeyError(f"No SELFIES column in {csv_path}. Columns: {list(df.columns)}")

    # --- guard 1: split metadata, if present -------------------------------
    split_col = None
    for candidate in ("SPLIT", "split", "Split"):
        if candidate in df.columns:
            split_col = candidate
            break
    if split_col is not None:
        counts = df[split_col].astype(str).str.lower().value_counts().to_dict()
        df = df[df[split_col].astype(str).str.lower() == "train"]
        print(f"  SPLIT column found {counts}; kept {len(df)} train rows")
    else:
        print("  no SPLIT column; the held-out cross-check below is the guarantee")

    # --- guard 2: hard exclusion against the held-out set ------------------
    smi_col = _find_smiles_column(df)
    if heldout is None:
        print("  WARNING: no held-out set loaded, so test-split exclusion CANNOT")
        print("  be verified. Set MOSES_HELDOUT_CSV before the final evaluation.")
    elif smi_col is None:
        print("  WARNING: no SMILES column in the seed CSV, so test-split")
        print("  exclusion cannot be verified.")
    else:
        # Cheap canonicality check: if these are not canonical, the isin() below
        # silently misses overlaps and the guarantee is void.
        sample = df[smi_col].dropna().astype(str).head(200)
        noncanon = 0
        for smi in sample:
            m = Chem.MolFromSmiles(smi)
            if m is not None and Chem.MolToSmiles(m) != smi:
                noncanon += 1
        if noncanon:
            print(f"  WARNING: {noncanon}/200 sampled seed SMILES are not canonical.")
            print("  The held-out exclusion below matches on exact strings and will")
            print("  MISS overlaps. Canonicalise the seed CSV before trusting it.")

        before = len(df)
        df = df[~df[smi_col].astype(str).isin(heldout)]
        removed = before - len(df)
        if removed:
            print(f"  EXCLUDED {removed} seeds that are in the held-out test split")
        else:
            print("  0 overlap with the held-out test split (CSV was already train-only)")

    pool = df[sel_col].dropna().astype(str).tolist()
    if not pool:
        raise ValueError(f"SELFIES column '{sel_col}' is empty after filtering.")
    return pool


def load_heldout_smiles(csv_path: Optional[str]) -> Optional[set]:
    """Load and canonicalise the held-out MOSES split for exact-match novelty."""
    if not csv_path or not os.path.exists(csv_path):
        print(f"[warn] held-out split not found ({csv_path}); "
              f"novelty column will be NaN.")
        return None
    df = pd.read_csv(csv_path)
    col = _find_smiles_column(df)
    if col is None:
        print(f"[warn] no SMILES column in held-out split; novelty will be NaN.")
        return None
    canon = set()
    for smi in df[col].dropna().astype(str):
        m = Chem.MolFromSmiles(smi)
        if m is not None:
            canon.add(Chem.MolToSmiles(m))
    print(f"Held-out split: {len(canon)} canonical SMILES")
    return canon


def clean_selfies(decoded: str) -> str:
    return decoded.replace(" ", "").strip()


def selfies_to_canonical_smiles(selfies_string: str) -> Optional[str]:
    try:
        smiles = sf.decoder(selfies_string)
    except Exception:
        return None
    if not smiles:
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        return Chem.MolToSmiles(mol)
    except Exception:
        return None


# ===========================================================================
# MODEL LOADING
# ===========================================================================

_BASE_CACHE = {"model": None}


def _get_frozen_base():
    if _BASE_CACHE["model"] is None:
        _BASE_CACHE["model"] = AutoModelForSeq2SeqLM.from_pretrained(BASE_MODEL)
    return _BASE_CACHE["model"]


def load_model(condition_type: str, path: str):
    if condition_type == "full":
        model = AutoModelForSeq2SeqLM.from_pretrained(path)
    elif condition_type == "peft":
        # Prefix tuning: load the adapter onto the frozen base. Never
        # merge_and_unload() -- for a prefix adapter it silently returns the
        # unmodified base model.
        from peft import PeftModel
        model = PeftModel.from_pretrained(_get_frozen_base(), path)
    else:
        raise ValueError(f"Unknown condition type: {condition_type!r}")
    model.to(DEVICE)
    if USE_FP16 and DEVICE == "cuda":
        model.half()
    model.eval()
    return model


# ===========================================================================
# (2) GENERATION  --  decoupled: fixed budget, keep everything
# ===========================================================================

def _mask_noise(input_ids, tokenizer, mask_prob: float = MASK_PROB):
    """Replace mask_prob of non-special encoder tokens with [MASK].

    Ported from feedback_hook2._mask_noise. Two details matter:

    - Special tokens are protected via all_special_ids, which includes the pad
      token, so batch padding is never masked and short sequences in a padded
      batch are not corrupted differently from long ones.
    - torch.rand is called while input_ids is still on CPU, before .to(DEVICE),
      exactly as in feedback_hook2. make_cell_rng seeds torch per cell, so the
      mask pattern is reproducible and decorrelated across cells along with
      everything else.
    """
    noised = input_ids.clone()
    is_special = torch.zeros_like(noised, dtype=torch.bool)
    for sid in set(tokenizer.all_special_ids):
        is_special |= noised == sid
    do_mask = (~is_special) & (torch.rand(noised.shape) < mask_prob)
    noised[do_mask] = tokenizer.mask_token_id
    return noised, int(do_mask.sum()), int((~is_special).sum())


def generate_cell(model, tokenizer, encoder_selfies: List[str]) -> List[dict]:
    rows: List[dict] = []
    for start in range(0, len(encoder_selfies), BATCH_SIZE):
        batch = encoder_selfies[start:start + BATCH_SIZE]

        # Tokenize on CPU, corrupt, THEN move to device (see _mask_noise).
        enc = tokenizer(batch, return_tensors="pt", padding=True,
                        truncation=True, max_length=MAX_LENGTH)
        noised_ids, n_masked, n_maskable = _mask_noise(enc["input_ids"], tokenizer)
        noised_ids = noised_ids.to(DEVICE)
        # Attention mask comes from the ORIGINAL encoding: masking substitutes
        # tokens, it does not remove them, so the attended positions are
        # unchanged and [MASK] must stay visible to the encoder.
        attn = enc["attention_mask"].to(DEVICE)

        with torch.inference_mode():
            out = model.generate(
                input_ids=noised_ids,
                attention_mask=attn,
                do_sample=True,
                temperature=TEMPERATURE,
                top_k=TOP_K,
                top_p=TOP_P,
                num_return_sequences=1,
                min_length=MIN_LENGTH,
                max_length=MAX_LENGTH,
            )
        decoded = tokenizer.batch_decode(out, skip_special_tokens=True)

        # Decode the corrupted input too, so every draw records what the encoder
        # actually saw rather than only the clean seed it came from.
        noised_decoded = tokenizer.batch_decode(noised_ids, skip_special_tokens=False)

        realised = (n_masked / n_maskable) if n_maskable else 0.0
        for seed_sf, noised_sf, raw in zip(batch, noised_decoded, decoded):
            raw_sf = clean_selfies(raw)
            canon = selfies_to_canonical_smiles(raw_sf)
            rows.append({
                "encoder_selfies": seed_sf,
                "encoder_selfies_masked": clean_selfies(noised_sf),
                "batch_mask_rate": realised,
                "raw_selfies": raw_sf,
                "canonical_smiles": canon if canon is not None else "",
                "valid": canon is not None,
            })
    return rows


# ===========================================================================
# OUTPUT
# ===========================================================================

def cell_dir_and_path(model_version: str, gen_seed: int):
    d = os.path.join(OUTPUT_DIR, model_version)
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, f"gen{gen_seed}.csv")


def build_cell_dataframe(rows: List[dict], heldout: Optional[set]) -> pd.DataFrame:
    """Attach the property panel to every draw (NaN for invalid)."""
    records = []
    for r in rows:
        base = dict(r)
        seed_canon = selfies_to_canonical_smiles(clean_selfies(r["encoder_selfies"]))
        # Explicit reconstruction flag. `degenerate` conflates three causes
        # (tiny molecule / empty scaffold / seed copy), which is what let 100%
        # seed reconstruction hide behind 100% uniqueness in v2. This column
        # measures only the copy, against the CLEAN seed the draw came from.
        base["copies_seed"] = bool(
            r["valid"] and seed_canon is not None
            and r["canonical_smiles"] == seed_canon
        )
        if r["valid"]:
            props = compute_properties(r["canonical_smiles"], seed_canon)
        else:
            props = {k: np.nan for k in PROPERTY_COLUMNS}
        base.update(props)
        records.append(base)
    return pd.DataFrame(records)


def summarise_cell(df: pd.DataFrame, heldout: Optional[set]) -> Dict:
    n_total = len(df)
    valid = df[df["valid"]]
    n_valid = len(valid)
    validity_pct = 100.0 * n_valid / n_total if n_total else np.nan

    unique_smiles = set(valid["canonical_smiles"])
    n_unique = len(unique_smiles)
    uniqueness_within = 100.0 * n_unique / n_valid if n_valid else np.nan

    # THE diagnostic. Uniqueness compares outputs to OTHER OUTPUTS and so reads
    # ~100% even when every draw copies its own seed (v2: 1000 unique outputs,
    # 1000/1000 exact copies, because the seeds themselves all differ). This
    # compares each output to ITS OWN seed, which is the only way to see it.
    seed_copy_pct = 100.0 * valid["copies_seed"].mean() if n_valid else np.nan
    mask_rate = (float(df["batch_mask_rate"].mean())
                 if "batch_mask_rate" in df.columns else np.nan)

    if heldout is not None and n_unique:
        novel = sum(1 for s in unique_smiles if s not in heldout)
        novelty_heldout = 100.0 * novel / n_unique
    else:
        novelty_heldout = np.nan

    def mean_sd(col):
        vals = pd.to_numeric(valid[col], errors="coerce").dropna()
        if len(vals) == 0:
            return np.nan, np.nan
        return float(vals.mean()), float(vals.std(ddof=1))

    fsp3_m, fsp3_sd = mean_sd("Fsp3")
    sol_m, sol_sd = mean_sd("Solubility_logmolL")
    qed_m, qed_sd = mean_sd("QED")
    mw_m, mw_sd = mean_sd("MW")
    logp_m, logp_sd = mean_sd("LogP")

    return {
        "N Valid": n_valid,
        "Validity (%)": validity_pct,
        "N Unique": n_unique,
        "Uniqueness within-set (%)": uniqueness_within,
        "Seed Copy (%)": seed_copy_pct,          # 100.0 => pure reconstruction
        "Realised Mask Rate": mask_rate,         # sanity-check on MASK_PROB
        "Uniqueness vs Held-out (%)": novelty_heldout,  # exact-match novelty
        "Fsp3 Mean": fsp3_m, "Fsp3 SD": fsp3_sd,
        "Solubility Mean (log mol/L)": sol_m, "Solubility SD": sol_sd,
        "QED Mean": qed_m, "QED SD": qed_sd,
        "MW Mean (Da)": mw_m, "MW SD": mw_sd,
        "LogP Mean": logp_m, "LogP SD": logp_sd,
    }


def upsert_summary_row(row: Dict):
    """Read-modify-write the single summary CSV, replacing any existing row for
    this (model_version, gen_seed, mask_prob).

    mask_prob is part of the key: one summary spans every mask level so the
    sweep can be plotted from one file, and without it a second mask level would
    silently overwrite the first level's rows.
    """
    if os.path.exists(SUMMARY_CSV):
        summary = pd.read_csv(SUMMARY_CSV)
        if "mask_prob" not in summary.columns:
            summary["mask_prob"] = np.nan  # tolerate a pre-mask summary file
        keep = ~((summary["model_version"] == row["model_version"]) &
                 (summary["gen_seed"] == row["gen_seed"]) &
                 (summary["mask_prob"] == row["mask_prob"]))
        summary = summary[keep]
        summary = pd.concat([summary, pd.DataFrame([row])], ignore_index=True)
    else:
        summary = pd.DataFrame([row])
    os.makedirs(os.path.dirname(SUMMARY_CSV), exist_ok=True)
    summary.to_csv(SUMMARY_CSV, index=False)


# ===========================================================================
# MAIN
# ===========================================================================

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Device: {DEVICE}   fp16: {USE_FP16 and DEVICE == 'cuda'}")
    print(f"Temperature: {TEMPERATURE}   N_PER_CELL: {N_PER_CELL}")
    print(f"MASK_PROB: {MASK_PROB}   top_k: {TOP_K}   top_p: {TOP_P}")
    print(f"Output: {OUTPUT_DIR}")
    if MASK_PROB == 0.0:
        print("  NOTE: MASK_PROB=0 feeds clean seeds. MolGen-large is pretrained")
        print("  to map corrupted SELFIES to their original form, so expect near")
        print("  100% seed reconstruction. Valid as a sweep endpoint, not as a")
        print("  generation protocol.")
    print(f"Runs: {RUN_INDICES} -> train seeds {[RUN_TO_TRAIN_SEED[r] for r in RUN_INDICES]}")
    print(f"Generation seeds: {GENERATION_SEEDS}")

    forbidden = set(RUN_TO_TRAIN_SEED.values()) | {101}
    clash = forbidden.intersection(GENERATION_SEEDS)
    if clash:
        raise ValueError(f"Generation seeds collide with train/data seeds: {clash}")

    # Held-out set FIRST: it is passed into the seed loader as the exclusion
    # guard, so the seed pool can be guaranteed disjoint from the test split.
    heldout = load_heldout_smiles(MOSES_HELDOUT_CSV)

    seed_pool = load_seed_selfies(MOSES_CSV, heldout=heldout)
    print(f"Encoder seed pool: {len(seed_pool)} SELFIES")

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    for condition, spec in CONDITIONS.items():
        ctype, prefix = spec["type"], spec["prefix"]
        print(f"\n=== Condition: {condition}  (type={ctype}) ===")

        for run_idx in RUN_INDICES:
            train_seed = RUN_TO_TRAIN_SEED[run_idx]
            model_version = f"{prefix}{run_idx}"
            # Per-run path if the condition supplies a "paths" map; otherwise
            # the CHECKPOINT_ROOT/<prefix><run_idx> convention.
            paths = spec.get("paths") or {}
            ckpt = paths.get(run_idx) or os.path.join(CHECKPOINT_ROOT, model_version)
            if not os.path.isdir(ckpt):
                raise FileNotFoundError(
                    f"No checkpoint for {model_version} (train seed {train_seed}) "
                    f"at:\n  {ckpt}\n"
                    f"Each run index needs a model trained at ITS OWN seed "
                    f"({RUN_TO_TRAIN_SEED}). Either train it, or set "
                    f"RUN_INDICES to only the runs that exist."
                )
            print(f"\n-- {model_version} (train seed {train_seed}): loading {ckpt}")
            model = load_model(ctype, ckpt)

            for gen_seed in GENERATION_SEEDS:
                out_path = cell_dir_and_path(model_version, gen_seed)
                if os.path.exists(out_path):
                    existing = pd.read_csv(out_path)
                    if len(existing) >= N_PER_CELL:
                        print(f"   [skip] {model_version}/gen{gen_seed} "
                              f"({len(existing)} rows)")
                        continue

                # (1) SeedSequence-derived RNGs for this cell
                np_rng, _ = make_cell_rng(train_seed, gen_seed)

                # encoder seeds: reproducible, identical across conditions
                n_draw = min(N_PER_CELL, len(seed_pool))
                idx = np_rng.choice(len(seed_pool), size=n_draw, replace=False)
                encoder_selfies = [seed_pool[i] for i in idx]

                # (2) decoupled generation, keep every draw
                rows = generate_cell(model, tokenizer, encoder_selfies)

                df = build_cell_dataframe(rows, heldout)
                df.insert(0, "condition", condition)
                df.insert(1, "model_version", model_version)
                df.insert(2, "run_idx", run_idx)
                df.insert(3, "train_seed", train_seed)
                df.insert(4, "gen_seed", gen_seed)
                df.insert(5, "temperature", TEMPERATURE)
                df.insert(6, "mask_prob", MASK_PROB)
                df.to_csv(out_path, index=False)

                summary = summarise_cell(df, heldout)
                summary_row = {
                    "condition": condition,
                    "model_version": model_version,
                    "run_idx": run_idx,
                    "train_seed": train_seed,
                    "gen_seed": gen_seed,
                    "temperature": TEMPERATURE,
                    "mask_prob": MASK_PROB,
                    "top_k": TOP_K,
                    "top_p": TOP_P,
                    **summary,
                }
                upsert_summary_row(summary_row)

                print(f"   [done] gen{gen_seed}: "
                      f"valid={summary['N Valid']} "
                      f"({summary['Validity (%)']:.1f}%)  "
                      f"unique={summary['N Unique']}  "
                      f"seedcopy={summary['Seed Copy (%)']:.1f}%  "
                      f"QED={summary['QED Mean']:.3f}  "
                      f"Fsp3={summary['Fsp3 Mean']:.3f}")

            del model
            if DEVICE == "cuda":
                torch.cuda.empty_cache()

    print(f"\nAll cells complete. Summary: {SUMMARY_CSV}")


if __name__ == "__main__":
    main()
