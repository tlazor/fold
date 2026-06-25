# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**FOLD** (Fourier Overlap Linguistic Distance) is a research project that measures linguistic distance between languages by applying spectral analysis to token-level signals (likelihoods or embeddings) produced by multilingual language models (mBERT, XLM-R). The computed pairwise distances are then correlated against ground-truth linguistic measures: FSI difficulty scale, mutual intelligibility, lexical similarity, and phonetic similarity.

## Commands

All Python commands must be run from the `src/` directory since imports are relative (no package structure):

```bash
cd src

# Run the likelihood-based pipeline
uv run python pipeline.py

# Run the embedding-based pipeline
uv run python embed_pipeline.py

# Linting
uv run ruff check src/
uv run ruff format src/
```

Docker (for GPU workloads, requires NVIDIA container runtime):
```bash
docker compose run --rm fold
# Inside container:
cd /fold/src && uv run python pipeline.py
```

## Configuration

All pipeline behavior is controlled by editing `src/pipeline_options.py` — there are no CLI arguments. Key toggles in `PipelineOptions`:

| Option | Effect |
|---|---|
| `use_bible` / `use_un6` | Switch dataset (Bible aligned, UN 6-way, or default XNLI) |
| `use_bert` | `True` = mBERT, `False` = XLM-R |
| `no_spectra` | Skip spectral step; use raw token signals directly |
| `straight_spectra` | Use FFT (circular average) instead of Welch PSD |
| `layers` | Which transformer hidden layers to extract (embedding pipeline only) |
| `num_bands` | Number of frequency sub-bands to analyze |
| `analyze_pearson_contrib` | Plot per-pair Pearson contribution heatmaps |
| `use_cache` | Load pkl cache from `cache/pipeline_outputs/` instead of recomputing |

Output files are written to the repo root, named `{prefix}_{model}_likelihood_output.txt` / `{prefix}_{model}_embedding_output.txt`.

## Architecture

Both pipelines are `sklearn.Pipeline` chains built at runtime; `joblib.Memory` caches intermediate steps to `.cache/joblib/`. The pipeline data flow is:

```
DataLoader → TokenTransform → SampleTokens → [LikelihoodEstimator | EmbedTransformer]
    → [SpectralTransformer | PsdEstimator + PsdNormalizer | NoOpTransformer]
    → BandSelectTransformer → MetricTransformer → analyze_output()
```

**Data loaders** (`TsvToDataFrame`, `BibleTransformer`, `Un6Transformer`) each produce a DataFrame where rows are parallel sentences and columns are ISO 639-1 language codes.

**Signal extraction** produces a list (one element per sample) of arrays shaped `(num_langs, num_tokens)` for likelihoods, or `(num_langs, num_tokens, hidden_dim)` for embeddings.

**Spectral step** converts token sequences to power spectra:
- `SpectralTransformer`: circular-averaged FFT power spectrum
- `PsdEstimator` + `PsdNormalizer`: Welch method PSD, interpolated to a common frequency grid

**`MetricTransformer`** applies one of `compute_overlaps`, `kl_divergence_matrix`, `mae_matrix`, or `coherence_matrix` to each sample, producing `(num_langs, num_langs)` distance matrices that are stacked into `(num_samples, num_langs, num_langs)`.

**`analyze_output()`** (in `pipeline.py`) takes that stack, computes the median across samples, and correlates with FSI scale, mutual intelligibility (`constants.py`), lexical similarity, and phonetic similarity via Pearson and Spearman r. Results are printed as markdown tables.

**`constants.py`** stores all static linguistic ground-truth data: language lists for BERT/XLM-R, FSI difficulty ratings, lexical/phonetic similarity matrices, mutual intelligibility tables, language family mappings, and Wikipedia corpus sizes used for bias analysis.

**`fold_globals.py`** holds the global `DEVICE` (CPU/CUDA/MPS), set at pipeline startup.

## Data

- `data/XNLI-15way/` — 15-language parallel NLI corpus (default dataset)
- `data/aligned/` — Bible text aligned across languages (one file per book per language)
- `data/wals-v2020.4/` and `data/asjp-v20/` — WALS and ASJP data for linguistic feature analysis

The GPU-intensive steps (likelihood and embedding extraction) are joblib-cached; delete `.cache/` to force recomputation.
