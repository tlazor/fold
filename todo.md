# Refactor TODO

## 1. Add a test suite before touching anything else

The metric functions (`compute_overlaps`, `kl_divergence_matrix`, `mae_matrix`, `coherence_matrix`) and every sklearn transformer have clear input/output contracts that can be verified with small synthetic arrays — no GPU required. Write these first so that any later refactor can be validated.

- Test each metric function for output shape, symmetry (where expected), and known analytic values (e.g., `kl_divergence_matrix` of a distribution against itself should be 0; `compute_overlaps` of identical distributions should equal 1).
- Test each transformer's `transform()` in isolation with small fixed inputs (2 languages, 5 tokens). Assert output shape and dtype match expectations.
- Test `SampleTokens` filtering logic: samples with fewer than `minimum_tokens` must be dropped; random seed must produce reproducible output.
- Test `BandSelectTransformer` for both 2D and 3D inputs, asserting the correct frequency slice is selected.
- Test `analyze_output` / `calculate_correlations_new` with constructed DataFrames where the Pearson r is analytically known.
- Use `pytest` and add a `tests/` directory at the repo root.

## 2. Make runs reproducible via config files or CLI, not source edits

Currently you change behavior by editing `pipeline_options.py`. This makes it impossible to reproduce a past run from git history alone.

- Replace `PipelineOptions` with a YAML/TOML config file (one file per experiment) parsed at startup, or add a CLI (e.g., `argparse` or `typer`) that accepts all the same flags.
- Store a copy of the resolved config alongside each output file so every result is self-documenting.
- The two mutually-exclusive boolean flags `no_spectra` / `straight_spectra` should become a single enum or string choice: `spectral_mode: none | fft | welch`.

## 3. Unify the two pipeline scripts

`pipeline.py` and `embed_pipeline.py` share the same data-loading, spectra, band-selection, and metric-computation structure. The only real difference is `LikelihoodEstimator` vs `EmbedTransformer`.

- Create a single entry point (e.g., `run_pipeline.py`) that accepts a `signal_mode: likelihood | embedding` option and builds the appropriate pipeline.
- Move `analyze_output`, `get_langs`, `get_overlap`, and the correlation helpers out of `pipeline.py` into a dedicated `analysis.py` module, since `embed_pipeline.py` already imports from `pipeline.py`, making it both a script and an implicit library.

## 4. Make it a proper package runnable from the project root

All imports currently require `cd src/` before running. This is fragile and incompatible with test runners that operate from the repo root.

- Add `src/__init__.py` (or reorganize into a `fold/` package directory).
- Add a `[project.scripts]` entry in `pyproject.toml` so the pipeline can be invoked as `uv run fold-pipeline` from anywhere.
- Fix the hard-coded CWD-relative paths inside transformers (`TokenTransform` constructs `.cache/joblib` relative to wherever you run from; `BibleTransformer` checks `data/aligned/1-b.GEN/...` relative to CWD even though `file_path` is an explicit parameter).

## 5. Eliminate global mutable state

`fold_globals.py` is a module with a mutable `DEVICE = "cpu"` that gets overwritten at startup. This makes test isolation hard.

- Pass `device` explicitly to transformers that need it (`LikelihoodEstimator`, `EmbedTransformer`) instead of reading from a global.
- Determine the device once in the entry point and inject it.

## 6. Unify and document the caching strategy

There are two independent caching mechanisms: `joblib.Memory` for pipeline steps, and manual `pickle` in `embed_pipeline.py`. They use different keys, different directories (`.cache/joblib/` vs `cache/pipeline_outputs/`), and are invalidated differently.

- Pick one mechanism for the whole project or document clearly which layer each handles.
- Remove the `TokenTransform.transform()` anti-pattern of constructing a `Memory` object and defining a `@memory.cache` decorated function on every call to `transform()`.

## 7. Clean up `constants.py`

The file mixes language lists for BERT and XLM-R, FSI ratings, lexical/phonetic similarity matrices, mutual intelligibility tables, language family mappings, and Wikipedia corpus sizes — all as hard-coded Python literals.

- Move large static tables (lexical similarity, phonetic similarity, intelligibility matrices) to data files (CSV or JSON) loaded at import time.
- Keep language family and FSI mappings as constants but group them with section headers.
- Remove the unused `import constants` in `SampleTokens`.

## 8. Clean up leftover debug noise

Several `print()` calls and commented-out code blocks were left in production paths:

- `PsdNormalizer.transform()` prints the input shape on every call.
- `PsdEstimator.transform()` prints the input shape on every call.
- `pipeline.py` has multiple `# print(...)` comment blocks and a `show_plot = False` dead-code branch.
- `EmbedTransformer.transform()` has commented-out debug prints.

Either remove these or convert them to `logging` calls at an appropriate level so they can be toggled without touching source.
