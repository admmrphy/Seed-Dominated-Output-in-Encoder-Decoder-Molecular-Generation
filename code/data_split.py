"""
data_split.py

Produces and loads the ONE fixed train / validation / test split that every
condition in the study shares. This is the single artifact the whole protocol
depends on: standard FT, weighted FT, MPT standard, MPT weighted, and both
feedback models all draw their training corpus from the SAME training pool and
are all evaluated against the SAME test reference.

Design decision (worth confirming with the supervisor):
  * The MOSES OFFICIAL test split is left untouched and used as the benchmark
    reference for novelty, NNS and FCD, exactly as in the current paper. Making
    up a fresh random test set would break comparability with MOSES and with
    every result already computed.
  * A validation set is carved deterministically out of the OFFICIAL train rows
    using SPLIT_SEED. Corpus sampling (standard random pool, weighted 10/10/50
    pool) then draws only from the remaining training pool, so validation is
    genuinely held out from training rather than overlapping it.
  * Any 'test_scaffolds' rows in the MOSES SPLIT column are preserved under
    their own label and never mixed into train/validation/test.

If the source CSV has no SPLIT column, the module falls back to a fully
deterministic three-way split from SPLIT_SEED and prints a clear warning, so it
still runs but the departure from the MOSES benchmark is never silent.

The source CSV is treated as fixed-order ground truth. The split assignment is
saved as a compact CSV (row index + canonical SMILES key + split label) that
every downstream script loads, so the assignment is stable even if the split is
rebuilt on another machine.

Written for Python 3.9 (Optional[...], no X | Y unions).
"""

from typing import Dict, List, Optional
import os

import numpy as np
import pandas as pd

from seed_config import SPLIT_SEED

# Fraction of the OFFICIAL MOSES train rows to hold out as validation. Around
# 10% of the current training data, used for checkpoint selection, hyperparameter
# tuning, and comparing feedback configurations. The official test split stays
# untouched for final evaluation only.
DEFAULT_VALIDATION_FRACTION: float = 0.10

# Canonical split labels used everywhere downstream.
TRAIN = "train"
VALIDATION = "validation"
TEST = "test"
TEST_SCAFFOLDS = "test_scaffolds"

DEFAULT_SPLITS_PATH = "protocol_splits.csv"


# ---------------------------------------------------------------------------
# COLUMN RESOLUTION
# ---------------------------------------------------------------------------
# The real MOSES CSV has a SELFIES column MISSPELLED as "SEFLIES" (L and F
# swapped, capitalised). Scripts that read "selfies" lowercase silently miss it.
# These resolvers match case-insensitively and accept the known misspelling so
# the split builder never falls over on that quirk.

_SMILES_ALIASES = ("smiles", "canonical_smiles")
_SELFIES_ALIASES = ("selfies", "seflies")  # "seflies" is the known misspelling
_SPLIT_ALIASES = ("split", "set", "subset")


def _resolve_column(df: pd.DataFrame, aliases: tuple) -> Optional[str]:
    lowered = {c.lower(): c for c in df.columns}
    for alias in aliases:
        if alias in lowered:
            return lowered[alias]
    return None


def resolve_smiles_column(df: pd.DataFrame) -> str:
    col = _resolve_column(df, _SMILES_ALIASES)
    if col is None:
        raise KeyError(
            f"No SMILES column found. Looked for {_SMILES_ALIASES} "
            f"(case-insensitive) among {list(df.columns)}."
        )
    return col


def resolve_selfies_column(df: pd.DataFrame) -> str:
    col = _resolve_column(df, _SELFIES_ALIASES)
    if col is None:
        raise KeyError(
            f"No SELFIES column found. Looked for {_SELFIES_ALIASES} "
            f"(case-insensitive, including the known 'SEFLIES' misspelling) "
            f"among {list(df.columns)}."
        )
    return col


def resolve_split_column(df: pd.DataFrame) -> Optional[str]:
    return _resolve_column(df, _SPLIT_ALIASES)


def _normalise_split_value(value: str) -> Optional[str]:
    """Map a raw SPLIT cell to one of our canonical labels, or None if unknown."""
    v = str(value).strip().lower()
    if v in ("train", "training"):
        return TRAIN
    if v in ("test", "testing"):
        return TEST
    if v in ("test_scaffolds", "test_scaffold", "scaffold_test", "scaffolds"):
        return TEST_SCAFFOLDS
    return None


# ---------------------------------------------------------------------------
# BUILD
# ---------------------------------------------------------------------------
def build_splits(
    csv_path: str,
    splits_path: str = DEFAULT_SPLITS_PATH,
    validation_fraction: float = DEFAULT_VALIDATION_FRACTION,
    split_seed: int = SPLIT_SEED,
    overwrite: bool = False,
) -> pd.DataFrame:
    """Build the fixed split assignment and write it to splits_path.

    Returns a DataFrame with columns: index, smiles, protocol_split.
    'index' is the row position in the source CSV (fixed-order ground truth).
    """
    if os.path.exists(splits_path) and not overwrite:
        raise FileExistsError(
            f"{splits_path} already exists. Pass overwrite=True to rebuild, or "
            f"call load_splits() to use the existing assignment."
        )
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be strictly between 0 and 1.")

    df = pd.read_csv(csv_path)
    smiles_col = resolve_smiles_column(df)
    split_col = resolve_split_column(df)

    assignment = pd.Series([None] * len(df), index=df.index, dtype=object)

    if split_col is None:
        print(
            f"WARNING: no SPLIT column in {csv_path}. Falling back to a fully "
            f"deterministic 3-way split from SPLIT_SEED={split_seed}. This "
            f"DEPARTS from the MOSES official benchmark split; confirm this is "
            f"intended before using these results against MOSES."
        )
        rng = np.random.default_rng(split_seed)
        shuffled = rng.permutation(len(df))
        n_test = int(round(0.15 * len(df)))
        n_val = int(round(validation_fraction * (len(df) - n_test)))
        test_idx = set(shuffled[:n_test].tolist())
        val_idx = set(shuffled[n_test:n_test + n_val].tolist())
        for i in range(len(df)):
            if i in test_idx:
                assignment.iloc[i] = TEST
            elif i in val_idx:
                assignment.iloc[i] = VALIDATION
            else:
                assignment.iloc[i] = TRAIN
    else:
        # Respect the official split. Carve validation only out of train rows.
        raw = df[split_col].map(_normalise_split_value)
        unknown = df[split_col][raw.isna()].unique().tolist()
        if unknown:
            raise ValueError(
                f"SPLIT column '{split_col}' has unrecognised values {unknown}. "
                f"Extend _normalise_split_value() to handle them explicitly "
                f"rather than guessing."
            )
        assignment = raw.copy()

        train_positions = np.where(assignment.values == TRAIN)[0]
        rng = np.random.default_rng(split_seed)
        rng.shuffle(train_positions)
        n_val = int(round(validation_fraction * len(train_positions)))
        val_positions = train_positions[:n_val]
        assignment.iloc[val_positions] = VALIDATION

    out = pd.DataFrame(
        {
            "index": np.arange(len(df)),
            "smiles": df[smiles_col].values,
            "protocol_split": assignment.values,
        }
    )
    out.to_csv(splits_path, index=False)
    _print_summary(out, split_seed, validation_fraction, from_official=split_col is not None)
    return out


def _print_summary(out: pd.DataFrame, split_seed, validation_fraction, from_official) -> None:
    counts = out["protocol_split"].value_counts().to_dict()
    print(f"Split written with SPLIT_SEED={split_seed}, "
          f"validation_fraction={validation_fraction}, "
          f"official_test_preserved={from_official}")
    for label in (TRAIN, VALIDATION, TEST, TEST_SCAFFOLDS):
        if label in counts:
            print(f"  {label:<16}{counts[label]:>10}")


# ---------------------------------------------------------------------------
# LOAD
# ---------------------------------------------------------------------------
def load_splits(splits_path: str = DEFAULT_SPLITS_PATH) -> pd.DataFrame:
    """Load the fixed split assignment written by build_splits()."""
    if not os.path.exists(splits_path):
        raise FileNotFoundError(
            f"{splits_path} not found. Run build_splits() once first; every "
            f"training and corpus-building script then loads this same file."
        )
    return pd.read_csv(splits_path)


def get_split_indices(
    label: str, splits_path: str = DEFAULT_SPLITS_PATH
) -> np.ndarray:
    """Return the source-CSV row indices belonging to one split label."""
    splits = load_splits(splits_path)
    valid = set(splits["protocol_split"].unique().tolist())
    if label not in valid:
        raise KeyError(f"Unknown split label '{label}'. Available: {sorted(valid)}")
    return splits.loc[splits["protocol_split"] == label, "index"].to_numpy()


def get_training_pool_frame(
    csv_path: str, splits_path: str = DEFAULT_SPLITS_PATH
) -> pd.DataFrame:
    """Return the source rows that make up the training pool.

    This is the ONLY frame that corpus construction should sample from. Both
    the standard random corpus and the weighted 10/10/50 corpus must be drawn
    from here so validation and test never leak into training.
    """
    df = pd.read_csv(csv_path)
    idx = get_split_indices(TRAIN, splits_path)
    return df.iloc[idx].reset_index(drop=True)


def get_validation_frame(
    csv_path: str, splits_path: str = DEFAULT_SPLITS_PATH
) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    idx = get_split_indices(VALIDATION, splits_path)
    return df.iloc[idx].reset_index(drop=True)


def get_test_frame(
    csv_path: str, splits_path: str = DEFAULT_SPLITS_PATH
) -> pd.DataFrame:
    """Return the benchmark test reference (MOSES official test split).

    FINAL EVALUATION ONLY. This split should be scored only after all modelling
    decisions (checkpoint selection, hyperparameters, feedback configuration) are
    finalised on the validation set. Do not use it for tuning or model choice.
    """
    df = pd.read_csv(csv_path)
    idx = get_split_indices(TEST, splits_path)
    return df.iloc[idx].reset_index(drop=True)


def sample_standard_corpus(
    csv_path: str,
    n: int,
    data_seed: int,
    splits_path: str = DEFAULT_SPLITS_PATH,
) -> pd.DataFrame:
    """Draw the standard (unweighted) corpus from the training pool.

    Weighted corpus construction is not implemented here on purpose: it keeps
    the project's existing 10/10/50 stratified logic. To bring the weighted
    pool under the same protocol, have that logic operate on
    get_training_pool_frame(...) and seed its sampling with
    get_data_seed('weighted'), exactly as this function does for standard.
    """
    pool = get_training_pool_frame(csv_path, splits_path)
    if n > len(pool):
        raise ValueError(f"Requested {n} molecules but training pool has {len(pool)}.")
    return pool.sample(n=n, random_state=data_seed).reset_index(drop=True)


if __name__ == "__main__":
    # Demonstrates the full build/load cycle on the mock CSV. On the real
    # project, run once as:  build_splits("moses_full_properties_with_selfies.csv")
    demo_csv = "mock_moses.csv"
    demo_splits = "mock_protocol_splits.csv"
    if os.path.exists(demo_splits):
        os.remove(demo_splits)
    build_splits(demo_csv, splits_path=demo_splits, overwrite=True)
    idx = get_split_indices(VALIDATION, demo_splits)
    print(f"\nvalidation rows: {len(idx)} (sample indices {idx[:5].tolist()})")
    corpus = sample_standard_corpus(demo_csv, n=200, data_seed=101, splits_path=demo_splits)
    print(f"standard corpus sample: {len(corpus)} rows, "
          f"unique={corpus.iloc[:,0].nunique()}")
