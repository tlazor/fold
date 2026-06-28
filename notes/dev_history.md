# FOLD Development History

---

## `get_overlap` upper-triangle mask bug (2026-06-26)

**Commit:** `d96d81b` — *fix: apply upper-triangle mask after label reordering in get_overlap*
**File:** `src/analysis.py`, function `get_overlap`

### Background

`get_overlap` is called in `analyze_output` to align the pipeline's pairwise distance matrix (`xnli_df`) with a ground-truth baseline (e.g. `LEXICAL_SIMILARITY`, `PHONETIC_SIMILARITY`). When the baseline is symmetric (same value for (A,B) and (B,A)), the `symmetrical=True` flag limits the comparison to the upper triangle to avoid counting each pair twice.

### The bug

The old code built the upper-triangle mask from the baseline's **native shape and row/column order**, then applied it **before** subsetting to the sorted language intersection:

```python
# OLD — buggy
upper_triangle_mask = np.triu(np.ones(baseline.shape), k=0).astype(bool)
baseline_df = baseline.where(upper_triangle_mask)   # mask applied in Excel order

df_a_sub = xnli_df.loc[intersection, intersection]
df_b_sub = baseline_df.loc[intersection, intersection]  # subset after masking
```

`LEXICAL_SIMILARITY` is loaded from an Excel file whose rows and columns follow an arbitrary non-alphabetical order (starting: `['sq', 'lv', 'lt', 'eu', ...]`). The `np.triu` mask aligned to positions in *that* native order, not to the alphabetically-sorted `intersection` used everywhere else. After the `.loc[intersection, intersection]` reindex, mask and data were misaligned.

Concretely: in the native Excel order `sq` is row 0 and `bg` is row ~15, so the cell at native position `(sq, bg)` falls in the *upper* triangle and its value is kept. But in the alphabetically-sorted intersection `bg < sq`, so the correctly-masked upper-triangle cell is `(bg, sq)`, not `(sq, bg)`. The old code therefore silently associated the fold distance `xnli_df['sq', 'bg']` with the lex_sim value for the pair `{bg, sq}`.

### The fix

Subset to the sorted intersection first, then build the mask from the resulting `(n × n)` matrix so mask indices match the sorted label order:

```python
# NEW — correct
df_a_sub = xnli_df.loc[intersection, intersection]
df_b_sub = baseline.loc[intersection, intersection]   # subset first

if symmetrical:
    n = len(intersection)
    mask = np.triu(np.ones((n, n), dtype=bool), k=0)
    df_b_sub = df_b_sub.where(mask)                  # mask in sorted order
```

### Impact

This was verified by comparing outputs from two runs on the same corpus/model (Bible + mBERT):

- **`significant_output/bible_bert_*.txt`** — generated with the old code (`no_spectra=True`)
- **`after_refactor/bible_bert_*.txt`** — generated with the fixed code (`spectral_mode=welch`)

The 35 `lex_sim` language pairs that appear in each output cover the **same 35 unordered pairs** (the intersection of the pipeline's language set with LEXICAL_SIMILARITY), but every pair whose alphabetical order disagrees with its native Excel row order appears with a **reversed direction label**:

| Old label (Excel order) | New label (alphabetical order) |
|---|---|
| `(sq, bg)` | `(bg, sq)` |
| `(sk, bg)` | `(bg, sk)` |
| `(nl, da)` | `(da, nl)` |
| `(en, de)` | `(de, en)` |
| `(lt, de)` | `(de, lt)` |
| `(pt, es)` | `(es, pt)` |
| ... | ... |

For **symmetric** metrics (`compute_overlaps`, `coherence_matrix`): the distance matrix satisfies `D[i,j] == D[j,i]`, so the direction flip changes only the pair label, not the fold value. The `lex_sim` correlation coefficients are numerically identical for those metrics (modulo any spectral mode difference between runs).

For **asymmetric** metrics (`kl_divergence_matrix`): `KL(A‖B) ≠ KL(B‖A)`. The old code paired the lex_sim value for `{bg, sq}` with the fold distance `KL(sq → bg)` instead of the correct `KL(bg → sq)`, producing silently wrong correlation coefficients for any ground-truth baseline used with `symmetrical=True`.

The `LEXICAL_SIMILARITY` matrix was confirmed to store numeric values in both triangles for essentially all covered pairs (108 non-NaN in upper, 109 in lower after alphabetical sort, with the single extra lower-only pair being `(mt, ar)` — Maltese/Arabic — which is not in the Bible BERT language set). This means the lex_sim values themselves are unaffected; only the fold-distance direction was wrong.
