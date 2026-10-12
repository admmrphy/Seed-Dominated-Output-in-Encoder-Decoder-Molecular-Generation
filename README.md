# Seed-Dominated Output in Encoder-Decoder Molecular Generation

Data, code and model weights for the manuscript on property-weighted
fine-tuning of MolGen-large on MOSES.

## Design

A 2x2 factorial over adaptation regime (full fine-tuning, prefix tuning)
and training-pool composition (standard, property-weighted). Each of the
four conditions was trained under three seeds (11, 42, 73) and sampled
under ten generation seeds, giving 120 cells of 1,000 molecules and
120,000 generated molecules in total.

Generation: temperature 1.0, top-k 30, top-p 0.75, encoder masking
probability 0.15. Base model: zjunlp/MolGen-large.

## Contents

- `data/generated/` - all 120,000 molecules with computed properties,
  one gzipped CSV per condition. Includes the encoder seed, the masked
  encoder input, a `copies_seed` flag and 20 computed descriptors.
- `data/aggregated/` - per-cell (120 rows), per-run (12 rows) and
  per-condition tables, plus the single-cell chemical feedback pilot.
- `models/adapters/` - the six prefix-tuning adapters with their configs.
- `code/` - training, generation and analysis scripts.

Full fine-tuning checkpoints (1.4 GB each) are attached as assets to the
tagged release, not stored in this repository.

## Key result

A fully fine-tuned MolGen-large reproduces its encoder seed unchanged in
74.00% of outputs under the standard pool and 71.31% under the weighted
pool. Pool composition has a negligible effect on every target property.

## Citation

Manuscript under review. DOI to follow.
