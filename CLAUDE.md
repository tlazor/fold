# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**FOLD** (Fourier Overlap Linguistic Distance) is a research project that measures linguistic distance between languages by applying spectral analysis to token-level signals (likelihoods or embeddings) produced by multilingual language models (mBERT, XLM-R). The computed pairwise distances are then correlated against ground-truth linguistic measures: FSI difficulty scale, mutual intelligibility, lexical similarity, and phonetic similarity.

## Commands

All commands run from the **project root** (no `cd src/` required):

```bash
# Run tests
uv run pytest

# Run the unified pipeline (likelihood or embedding mode)
uv run python src/run_pipeline.py --signal-mode likelihood --dataset xnli --model bert
uv run python src/run_pipeline.py --signal-mode embedding --dataset bible --model xlmr --layers 6 12

# After `uv sync`, the installed entry point is also available:
uv run fold-pipeline --signal-mode likelihood --help

# Linting
uv run ruff check src/
uv run ruff format src/
```

Docker (for GPU workloads, requires NVIDIA container runtime):
```bash
docker compose run --rm fold
# Inside container (project root is /fold):
uv run python src/run_pipeline.py --signal-mode likelihood
```

## Configuration

All pipeline behavior is controlled via CLI flags passed to `run_pipeline.py`. There are no longer any in-source config edits needed.

| Flag | Choices | Default | Effect |
|---|---|---|---|
| `--signal-mode` | `likelihood`, `embedding` | `likelihood` | Whether to extract masked LM probabilities or hidden-layer embeddings |
| `--dataset` | `xnli`, `bible`, `un6` | `xnli` | Parallel corpus |
| `--model` | `bert`, `xlmr` | `bert` | Multilingual model (mBERT or XLM-R) |
| `--spectral-mode` | `welch`, `fft`, `none` | `welch` | Spectral transform applied before computing metrics |
| `--layers` | integers | `12` | Hidden layers to extract (embedding mode only) |
| `--num-bands` | integer | `1` | Number of equal-width frequency sub-bands |
| `--use-cache` | flag | off | Load cached embedding pipeline outputs |
| `--analyze-pearson-contrib` | flag | off | Plot per-pair Pearson contribution heatmaps |
| `--output-dir` | path | `.` | Directory for output `.txt` and `.json` files |

A JSON snapshot of the resolved config is saved alongside every output `.txt` file automatically.

Output files: `{output_dir}/{dataset}_{model}_{signal_mode}_output.txt`

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

**`analyze_output()`** (in `analysis.py`) takes that stack, computes the median across samples, and correlates with FSI scale, mutual intelligibility (`constants.py`), lexical similarity, and phonetic similarity via Pearson and Spearman r. Results are printed as markdown tables.

**`constants.py`** stores all static linguistic ground-truth data: language lists for BERT/XLM-R, FSI difficulty ratings, lexical/phonetic similarity matrices, mutual intelligibility tables, language family mappings, and Wikipedia corpus sizes used for bias analysis.

**`fold_globals.py`** holds the global `DEVICE` (CPU/CUDA/MPS), set at pipeline startup.

## Data

- `data/XNLI-15way/` — 15-language parallel NLI corpus (default dataset)
- `data/aligned/` — Bible text aligned across languages (one file per book per language)
- `data/wals-v2020.4/` and `data/asjp-v20/` — WALS and ASJP data for linguistic feature analysis

The GPU-intensive steps (likelihood and embedding extraction) are joblib-cached; delete `.cache/` to force recomputation.
