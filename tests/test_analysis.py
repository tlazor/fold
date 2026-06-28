import numpy as np
import pandas as pd
import pytest

from analysis import get_overlap, calculate_correlations_new


def _make_df(langs, seed=0):
    rng = np.random.default_rng(seed)
    n = len(langs)
    return pd.DataFrame(rng.random((n, n)), index=langs, columns=langs)


# ===========================================================================
# get_overlap
# ===========================================================================

class TestGetOverlap:
    def test_column_names(self):
        xnli = _make_df(["a", "b", "c"])
        base = _make_df(["a", "b", "c"])
        result = get_overlap(xnli, base, "my_metric", symmetrical=True)
        assert list(result.columns) == ["fold", "my_metric"]

    def test_intersection_determines_row_count(self):
        # ["a","b","c","d"] ∩ ["b","c","d","e"] = ["b","c","d"] → 3×3 = 9 rows
        xnli = _make_df(["a", "b", "c", "d"])
        base = _make_df(["b", "c", "d", "e"])
        result = get_overlap(xnli, base, "test", symmetrical=False)
        assert len(result) == 9

    def test_symmetrical_masks_lower_triangle(self):
        # 3×3 intersection with symmetrical=True → 3 NaN in baseline column
        # (positions (b,a), (c,a), (c,b) are in the lower triangle)
        xnli_vals = np.full((3, 3), 0.5)
        np.fill_diagonal(xnli_vals, 0)
        base_vals = np.arange(9, dtype=float).reshape(3, 3)
        langs = ["a", "b", "c"]
        xnli = pd.DataFrame(xnli_vals, index=langs, columns=langs)
        base = pd.DataFrame(base_vals, index=langs, columns=langs)
        result = get_overlap(xnli, base, "base", symmetrical=True)
        assert result["base"].isna().sum() == 3

    def test_non_symmetrical_no_nans_from_mask(self):
        xnli_vals = np.full((3, 3), 0.5)
        base_vals = np.arange(9, dtype=float).reshape(3, 3)
        langs = ["a", "b", "c"]
        xnli = pd.DataFrame(xnli_vals, index=langs, columns=langs)
        base = pd.DataFrame(base_vals, index=langs, columns=langs)
        result = get_overlap(xnli, base, "base", symmetrical=False)
        assert result["base"].isna().sum() == 0

    def test_fold_column_contains_xnli_values(self):
        xnli_vals = np.eye(3) * 0.9
        base_vals = np.ones((3, 3))
        langs = ["a", "b", "c"]
        xnli = pd.DataFrame(xnli_vals, index=langs, columns=langs)
        base = pd.DataFrame(base_vals, index=langs, columns=langs)
        result = get_overlap(xnli, base, "b", symmetrical=False)
        # select diagonal entries (where row label == col label) via boolean mask
        diagonal = result[[idx[0] == idx[1] for idx in result.index]]["fold"]
        np.testing.assert_allclose(diagonal.values, 0.9, rtol=1e-10)


# ===========================================================================
# calculate_correlations_new
# ===========================================================================

class TestCalculateCorrelationsNew:
    def test_perfectly_positive_correlation(self):
        x = np.arange(1, 11, dtype=float)
        df = pd.DataFrame({"fold": x, "metric": 2 * x + 3})
        result = calculate_correlations_new({"d": df})
        assert len(result) == 1
        np.testing.assert_allclose(result.iloc[0]["p_coef"], 1.0, atol=1e-10)

    def test_perfectly_negative_correlation(self):
        x = np.arange(1, 11, dtype=float)
        df = pd.DataFrame({"fold": x, "metric": -x + 15.0})
        result = calculate_correlations_new({"d": df})
        np.testing.assert_allclose(result.iloc[0]["p_coef"], -1.0, atol=1e-10)

    def test_zero_correlation(self):
        # x=[1,2,3,4,5], y=[1,-1,1,-1,1]
        # (x - mean_x) = [-2,-1,0,1,2], (y - mean_y) = [0.8,-1.2,0.8,-1.2,0.8]
        # dot product = -1.6 + 1.2 + 0 - 1.2 + 1.6 = 0 exactly
        df = pd.DataFrame({
            "fold":   [1.0,  2.0, 3.0,  4.0, 5.0],
            "metric": [1.0, -1.0, 1.0, -1.0, 1.0],
        })
        result = calculate_correlations_new({"d": df})
        assert abs(result.iloc[0]["p_coef"]) < 1e-10

    def test_pval_significant_for_perfect_correlation(self):
        x = np.arange(1, 11, dtype=float)
        df = pd.DataFrame({"fold": x, "metric": x})
        result = calculate_correlations_new({"d": df})
        assert result.iloc[0]["p_pval"] < 0.001

    def test_required_columns_present(self):
        df = pd.DataFrame({"fold": [1.0, 2.0, 3.0], "m": [1.0, 2.0, 3.0]})
        result = calculate_correlations_new({"d": df})
        required = {"metric", "p_coef", "p_pval", "s_coef", "s_pval", "num_points"}
        assert required.issubset(set(result.columns))

    def test_metric_column_contains_second_column_name(self):
        df = pd.DataFrame({"fold": [1.0, 2.0, 3.0], "expected_name": [1.0, 2.0, 3.0]})
        result = calculate_correlations_new({"d": df})
        assert result.iloc[0]["metric"] == "expected_name"

    def test_all_nan_dataframe_skipped(self):
        df = pd.DataFrame({"fold": [np.nan, np.nan], "m": [np.nan, np.nan]})
        result = calculate_correlations_new({"d": df})
        assert len(result) == 0

    def test_multiple_dataframes(self):
        x = np.arange(1, 11, dtype=float)
        df1 = pd.DataFrame({"fold": x, "m1": x})
        df2 = pd.DataFrame({"fold": x, "m2": -x})
        result = calculate_correlations_new({"d1": df1, "d2": df2})
        assert len(result) == 2
        by_metric = result.set_index("metric")["p_coef"]
        assert by_metric["m1"] > 0.99
        assert by_metric["m2"] < -0.99
