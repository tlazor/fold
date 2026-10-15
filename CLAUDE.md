# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**FOLD** (Fourier Overlap Linguistic Distance) is a research project that measures linguistic distance between languages by applying spectral analysis to token-level signals (likelihoods or embeddings) produced by multilingual language models (mBERT, XLM-R). The computed pairwise distances are then correlated against ground-truth linguistic measures: FSI difficulty scale, mutual intelligibility, lexical distance, phonetic distance, and WALS typological similarity.

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
docker compose -f docker/docker-compose.yaml run --rm fold
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
| `--analyze-pearson-contrib` | flag | off | Plot per-pair Pearson contribution heatmaps |
| `--output-dir` | path | `.` | Directory for output `.txt` files |

Output files: `{output_dir}/{dataset}_{model}_{signal_mode}_output.txt`

## Architecture

Both pipelines are `sklearn.Pipeline` chains built at runtime; `joblib.Memory` caches intermediate steps to `.cache/joblib/`. The pipeline data flow is:

```
DataLoader → TokenTransform → SampleTokens → [LikelihoodEstimator | EmbedTransformer]
    → [SpectralTransformer | PsdEstimator + PsdNormalizer | NoOpTransformer]
    → BandSelectTransformer → MetricTransformer → analyze_output()
```

**Data loaders** (`TsvToDataFrame`, `BibleTransformer`, `Un6Transformer`) each produce a DataFrame where rows are parallel sentences and columns are exactly `langs`, in order. Later steps identify languages by position and `analyze_output()` relabels results with `langs`, so loaders must raise rather than skip a missing language.

**Signal extraction** produces a list (one element per sample) of arrays shaped `(num_langs, num_tokens)` for likelihoods, or `(num_langs, num_tokens, hidden_dim)` for embeddings.

**Spectral step** converts token sequences to power spectra:
- `SpectralTransformer`: circular-averaged FFT power spectrum
- `PsdEstimator` + `PsdNormalizer`: Welch method PSD, interpolated to a common frequency grid

**`MetricTransformer`** applies one of `compute_overlaps`, `kl_divergence_matrix`, `mae_matrix`, or `coherence_matrix` to each sample, producing `(num_langs, num_langs)` distance matrices that are stacked into `(num_samples, num_langs, num_langs)`.

**`analyze_output()`** (in `analysis.py`) takes that stack, computes the median across samples, and correlates it with each ground-truth measure via Pearson, Spearman and Kendall. Results are printed as markdown tables with one row per measure:

| Metric | Measure | Direction |
|---|---|---|
| `mut_int` | Mutual intelligibility, % understood; asymmetric (rows = listener, columns = language heard) | similarity |
| `lex_dist` | Lexical distance (0–100) | distance |
| `pho_dist` | ASJP LDND phonetic distance | distance |
| `wals_sim` | IC-weighted WALS feature agreement | similarity |
| `fsi` | FSI weeks of study for English speakers | distance from English |

Read a correlation's sign against the measure's direction.

**Language identity.** Every language is identified by its pipeline code (ISO 639-1, else ISO 639-3, e.g. `yue`). `data/constants/languages.csv` maps each code to the identifier every external source uses (Bible filename, mBERT/XLM-R language-list name, WALS ID, pinned ASJP doculect, lexical-spreadsheet abbreviation); `languages.py` exposes it via `source_ids()` / `from_source_ids()`. Never convert codes on the fly (e.g. ISO 639-1 → 639-3 or code → English name): sources disagree on macrolanguages, names and dialects, and mismatches silently drop languages. Add or fix a registry row instead.

**`baselines.py`** loads each ground-truth measure lazily as a `Baseline`: a square matrix keyed by pipeline code (registry languages only, NaN diagonal) plus its name, kind (distance/similarity), symmetry and scale. `baselines.pairwise()` is what `analyze_output()` iterates; `baselines.fsi()` gives the per-language FSI series.

**`constants.py`** stores static data kept in code: dataset language lists, the mBERT/XLM-R language-name lists, FSI ratings, language family mappings, and Wikipedia / XLM-R corpus sizes used for bias analysis.

**`tests/test_coverage.py`** pins, for each dataset and measure, how many language pairs reach the correlation and which languages have none. If a change moves those numbers, update the table in the same commit and say why.

## Data

- `data/XNLI-15way/` — 15-language parallel NLI corpus (default dataset)
- `data/aligned/` — Bible text aligned across languages (one file per book per language)
- `data/wals-v2020.4/` and `data/asjp-v20/` — WALS and ASJP data for linguistic feature analysis
- `data/constants/` — the language registry (`languages.csv`) and the Germanic/Romance/Slavic mutual-intelligibility tables
- `data/asjp_ldnd_distances_by_iso2.csv` — phonetic distances, generated by `src/compute_lex_dist.py` (CPU, ~1 min); rerun it after changing the registry's `asjp_id` column
- `data/lexical-distance-matrix.xlsx` — lexical distances

The GPU-intensive steps (likelihood and embedding extraction) are joblib-cached; delete `.cache/` to force recomputation.
