"""Backward-compatibility shim — all functions have moved to analysis.py."""
from analysis import (  # noqa: F401
    show_heatmap,
    calculate_correlations_new,
    get_full_mut_int,
    star_fmt,
    formatters,
    analyze_output,
    analyze_pearson_contrib,
    plot_pearson_contrib,
    analyze_wikisize,
    calculate_coef_and_pval,
    get_overlap,
    get_langs,
)
