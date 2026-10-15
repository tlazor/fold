import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import contextlib
    import io
    import os

    import marimo as mo

    import paths

    # Use the models in data/models (as the Docker image does) rather than
    # downloading them; must be set before transformers is imported.
    os.environ.setdefault("HF_HOME", str(paths.DATA_DIR / "models"))

    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from matplotlib.colors import LinearSegmentedColormap
    from scipy.stats import pearsonr, spearmanr

    import analysis
    import baselines
    import constants
    from pipeline_options import PipelineOptions
    from experiment import (
        configure_torch,
        experiment_layers,
        metric_names,
        run_experiment,
        write_results,
    )

    return (
        LinearSegmentedColormap,
        PipelineOptions,
        analysis,
        baselines,
        configure_torch,
        constants,
        contextlib,
        experiment_layers,
        io,
        metric_names,
        mo,
        np,
        pd,
        pearsonr,
        plt,
        run_experiment,
        spearmanr,
        write_results,
    )


@app.cell
def _(mo):
    mo.md(r"""
    # FOLD experiments

    **FOLD** (Fourier Overlap Linguistic Distance) asks whether a multilingual
    language model "sees" languages as close in the way linguists do. For each of
    600 sentences that are translations of each other, we take a token-level
    signal from the model for every language, either how predictable each token
    is (*likelihood*) or its hidden-layer representation (*embedding*), and turn
    that sequence into a frequency spectrum. Comparing the spectra of every
    pair of languages gives a FOLD matrix per sentence; the median over sentences
    is the model's view of how far apart the languages are.

    We then correlate that matrix with independent ground truth: how well
    speakers understand each other, how much vocabulary and sound they share,
    how alike their grammars are, and how long English speakers take to learn
    them.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 1. Choose an experiment
    """)
    return


@app.cell
def _(mo):
    # Starting values can come from the command line (same flags as
    # run_pipeline.py), e.g. to render a report without the editor:
    #   marimo export html notebooks/experiments.py -o report.html -- --run --dataset bible
    _args = mo.cli_args()
    auto_run = "run" in _args or mo.app_meta().mode == "script"

    def _dropdown(options, flag, default, label):
        value = _args.get(flag, default)
        labels = {v: k for k, v in options.items()}
        return mo.ui.dropdown(options, value=labels[value], label=label)

    dataset = _dropdown(
        {"XNLI (15 languages)": "xnli", "Bible": "bible", "UN (6 languages)": "un6"},
        "dataset",
        "xnli",
        "Corpus",
    )
    model = _dropdown({"mBERT": "bert", "XLM-R": "xlmr"}, "model", "bert", "Model")
    signal_mode = _dropdown(
        {"likelihood": "likelihood", "embedding": "embedding"},
        "signal-mode",
        "likelihood",
        "Signal",
    )
    spectral_mode = _dropdown(
        {"Welch PSD": "welch", "FFT": "fft", "none (raw signal)": "none"},
        "spectral-mode",
        "welch",
        "Spectrum",
    )
    layers = mo.ui.multiselect(
        [str(i) for i in range(1, 13)],
        value=[str(layer) for layer in _args.get_all("layers")] or ["12"],
        label="Layers (embedding only)",
    )
    num_bands = mo.ui.number(
        start=1, stop=8, value=int(_args.get("num-bands", 1)), label="Frequency bands"
    )
    run_button = mo.ui.run_button(label="Run experiment")

    mo.vstack(
        [
            mo.hstack([dataset, model, signal_mode, spectral_mode], justify="start"),
            mo.hstack([layers, num_bands, run_button], justify="start"),
        ]
    )
    return (
        auto_run,
        dataset,
        layers,
        model,
        num_bands,
        run_button,
        signal_mode,
        spectral_mode,
    )


@app.cell
def _(
    PipelineOptions,
    dataset,
    layers,
    model,
    num_bands,
    signal_mode,
    spectral_mode,
):
    config = PipelineOptions(
        dataset=dataset.value,
        model=model.value,
        signal_mode=signal_mode.value,
        spectral_mode=spectral_mode.value,
        layers=sorted(int(layer) for layer in layers.value) or [12],
        num_bands=int(num_bands.value),
    )
    return (config,)


@app.cell
def _(mo):
    mo.md(r"""
    ## 2. Which comparisons the data supports

    Each ground-truth measure only covers some languages: the intelligibility
    study tested Germanic, Romance and Slavic languages, the lexical chart links
    nearby European languages, and so on. This table shows how many language
    pairs each measure can score for the chosen corpus *before* anything runs; a
    correlation over 4 pairs deserves much less weight than one over 105.
    """)
    return


@app.cell
def _(analysis, config, contextlib, io, np, pd):
    with contextlib.redirect_stdout(io.StringIO()):
        planned_langs = analysis.get_langs(config.dataset, config.use_bert)

    _ones = np.ones((len(planned_langs), len(planned_langs)))
    np.fill_diagonal(_ones, np.nan)
    _pairs = analysis.baseline_pairs(
        pd.DataFrame(_ones, index=planned_langs, columns=planned_langs)
    )
    _rows = []
    for _name, _df in _pairs.items():
        _usable = _df.dropna()
        if _name == "fsi":
            _covered = set(_usable.index) | {"en"}
        else:
            _covered = {lang for pair in _usable.index for lang in pair}
        _rows.append(
            {
                "measure": _name,
                "pairs": len(_usable),
                "languages without data": ", ".join(
                    sorted(set(planned_langs) - _covered)
                )
                or "–",
            }
        )
    _coverage = pd.DataFrame(_rows)
    _coverage
    return (planned_langs,)


@app.cell
def _(mo, planned_langs):
    mo.md(f"""
    **{len(planned_langs)} languages:** {", ".join(planned_langs)}
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 3. Run

    Signal extraction is the only slow step, and it is cached per sentence, so
    re-running a configuration you have run before takes seconds. A new one
    takes minutes on the GPU (about two for XNLI with mBERT likelihoods).
    """)
    return


@app.cell
def _(
    auto_run,
    config,
    configure_torch,
    experiment_layers,
    mo,
    run_button,
    run_experiment,
):
    mo.stop(
        not (run_button.value or auto_run),
        mo.callout(
            mo.md("Choose an experiment above and press **Run experiment**."),
            kind="info",
        ),
    )
    configure_torch()
    with mo.status.spinner(
        title=f"Running {config.dataset} / {config.model} / {config.signal_mode}"
    ):
        langs, outputs = run_experiment(config)

    mo.md(
        f"Done: {len(langs)} languages, {len(config.freq_bands)} band(s), "
        f"{len(experiment_layers(config))} layer(s)."
    )
    return langs, outputs


@app.cell
def _(mo):
    mo.md(r"""
    ## 4. How FOLD lines up with each measure

    Read each correlation's sign against what the two sides measure. Spectral
    overlap and coherence are **similarities** (higher means closer); KL
    divergence is a **distance**. Among the measures, mutual intelligibility
    and WALS are similarities, while lexical and phonetic distance and FSI weeks
    are distances. When FOLD and a measure point the same way, agreement shows
    up as a *positive* correlation; when they point opposite ways, as a
    *negative* one. The **agreement sign** column spells this out.
    Significance: \* p < 0.05, \*\* p < 0.01, \*\*\* p < 0.001.
    """)
    return


@app.cell
def _(analysis, baselines, pd):
    FOLD_KIND = {
        "compute_overlaps": "similarity",
        "kl_divergence_matrix": "distance",
        "coherence_fun": "similarity",
    }
    METRIC_LABEL = {
        "compute_overlaps": "Spectral overlap",
        "kl_divergence_matrix": "KL divergence",
        "coherence_fun": "Coherence",
    }
    MEASURE_LABEL = {
        "mut_int": "Mutual intelligibility (%)",
        "lex_dist": "Lexical distance (0–100)",
        "pho_dist": "Phonetic distance (ASJP LDND)",
        "wals_sim": "WALS similarity (0–1)",
        "fsi": "FSI weeks of study from English",
    }
    MEASURE_KIND = {b.name: b.kind for b in baselines.pairwise()} | {"fsi": "distance"}

    def correlation_table(results, metric):
        """Readable correlations for one FOLD metric (see calculate_correlations_new)."""
        rows = []
        for _, r in results.iterrows():
            kind = MEASURE_KIND[r["metric"]]
            rows.append(
                {
                    "measure": MEASURE_LABEL[r["metric"]],
                    "is a": kind,
                    "agreement sign": "+" if kind == FOLD_KIND[metric] else "−",
                    "Pearson r": round(r["p_coef"], 3),
                    "Pearson p": analysis.star_fmt(r["p_pval"]),
                    "Spearman ρ": round(r["s_coef"], 3),
                    "Spearman p": analysis.star_fmt(r["s_pval"]),
                    "Kendall τ": round(r["kt_coef"], 3),
                    "pairs": int(r["num_points"]),
                }
            )
        return pd.DataFrame(rows)

    return FOLD_KIND, MEASURE_KIND, MEASURE_LABEL, METRIC_LABEL, correlation_table


@app.cell
def _(
    METRIC_LABEL,
    analysis,
    config,
    correlation_table,
    experiment_layers,
    langs,
    metric_names,
    mo,
    outputs,
):
    _tabs = {}
    for _name in metric_names(config):
        _sections = []
        for _k, _band in enumerate(config.freq_bands):
            for _layer in experiment_layers(config):
                _heading = f"Band {_band[0]:.2f}–{_band[1]:.2f} of Nyquist"
                if _layer is not None:
                    _heading += f", layer {_layer}"
                _results = analysis.correlate_with_baselines(
                    outputs[(_k, _name, _layer)], langs
                )
                _sections += [
                    mo.md(f"**{_heading}**"),
                    mo.ui.table(
                        correlation_table(_results, _name),
                        selection=None,
                        show_download=True,
                    ),
                ]
        _tabs[METRIC_LABEL[_name]] = mo.vstack(_sections)
    mo.ui.tabs(_tabs)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 5. Look at one comparison

    The scatter shows every language pair the chosen measure covers (for FSI,
    every language's FOLD value from English), with the three pairs furthest
    from the trend labelled. The heatmap shows the full median FOLD matrix,
    grouped by language family.
    """)
    return


@app.cell
def _(MEASURE_LABEL, METRIC_LABEL, config, experiment_layers, mo, outputs):
    band_pick = mo.ui.dropdown(
        {f"{b[0]:.2f}–{b[1]:.2f}": k for k, b in enumerate(config.freq_bands)},
        value=f"{config.freq_bands[0][0]:.2f}–{config.freq_bands[0][1]:.2f}",
        label="Band",
    )
    metric_pick = mo.ui.dropdown(
        {METRIC_LABEL[name]: name for name in {key[1] for key in outputs}},
        value=METRIC_LABEL["compute_overlaps"],
        label="FOLD metric",
    )
    _layers = experiment_layers(config)
    layer_pick = mo.ui.dropdown(
        {("–" if layer is None else str(layer)): layer for layer in _layers},
        value="–" if _layers[0] is None else str(_layers[0]),
        label="Layer",
    )
    measure_pick = mo.ui.dropdown(
        {label: name for name, label in MEASURE_LABEL.items()},
        value=MEASURE_LABEL["fsi"],
        label="Compare with",
    )
    mo.hstack([metric_pick, measure_pick, band_pick, layer_pick], justify="start")
    return band_pick, layer_pick, measure_pick, metric_pick


@app.cell
def _(mo):
    # Chart colours: one series hue and a one-hue sequential ramp, stepped for
    # the notebook's light or dark theme (validated reference palette).
    _dark = mo.app_meta().theme == "dark"
    _ramp = [
        "#cde2fb",
        "#9ec5f4",
        "#6da7ec",
        "#3987e5",
        "#256abf",
        "#184f95",
        "#0d366b",
    ]
    STYLE = {
        "surface": "#1a1a19" if _dark else "#fcfcfb",
        "primary": "#ffffff" if _dark else "#0b0b0b",
        "secondary": "#c3c2b7" if _dark else "#52514e",
        "muted": "#898781",
        "grid": "#2c2c2a" if _dark else "#e1e0d9",
        "axis": "#383835" if _dark else "#c3c2b7",
        "series": "#3987e5" if _dark else "#2a78d6",
        # Low values recede toward the surface in either theme.
        "ramp": _ramp[::-1] if _dark else _ramp,
    }
    return (STYLE,)


@app.cell
def _(
    FOLD_KIND,
    LinearSegmentedColormap,
    MEASURE_KIND,
    MEASURE_LABEL,
    METRIC_LABEL,
    STYLE,
    constants,
    np,
    pearsonr,
    plt,
    spearmanr,
):
    def _chrome(fig, ax):
        fig.set_facecolor(STYLE["surface"])
        ax.set_facecolor(STYLE["surface"])
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(STYLE["axis"])
        ax.tick_params(colors=STYLE["axis"], labelcolor=STYLE["secondary"], labelsize=8)

    def scatter_figure(pairs, measure, metric):
        """FOLD value against a measure, one dot per language pair."""
        x, y = pairs[measure].to_numpy(float), pairs["fold"].to_numpy(float)
        fig, ax = plt.subplots(figsize=(5.2, 4.2), layout="constrained")
        _chrome(fig, ax)
        ax.grid(True, color=STYLE["grid"], linewidth=0.6)
        ax.set_axisbelow(True)
        ax.scatter(
            x, y, s=49, color=STYLE["series"], edgecolor=STYLE["surface"], linewidth=1.5
        )

        title = f"n = {len(x)}"
        if len(x) >= 3 and np.ptp(x) > 0 and np.ptp(y) > 0:
            title = (
                f"Pearson r = {pearsonr(x, y)[0]:.2f}, "
                f"Spearman ρ = {spearmanr(x, y)[0]:.2f}, " + title
            )
            # Label up to three of the points furthest from the least-squares
            # line, skipping any that would sit on top of a label already placed.
            slope, intercept = np.polyfit(x, y, 1)
            xn, yn = (x - x.min()) / np.ptp(x), (y - y.min()) / np.ptp(y)
            placed = []
            for i in np.argsort(-np.abs(y - (slope * x + intercept))):
                if len(placed) == 3:
                    break
                if any(np.hypot(xn[i] - xn[j], yn[i] - yn[j]) < 0.06 for j in placed):
                    continue
                placed.append(i)
                label = pairs.index[i]
                label = "–".join(label) if isinstance(label, tuple) else f"en–{label}"
                ax.annotate(
                    label,
                    (x[i], y[i]),
                    xytext=(5, 4),
                    textcoords="offset points",
                    fontsize=8,
                    color=STYLE["secondary"],
                )
        ax.set_title(title, loc="left", fontsize=10, color=STYLE["primary"])
        ax.set_xlabel(
            f"{MEASURE_LABEL[measure]} — {MEASURE_KIND[measure]}",
            fontsize=9,
            color=STYLE["secondary"],
        )
        ax.set_ylabel(
            f"{METRIC_LABEL[metric]} — {FOLD_KIND[metric]}",
            fontsize=9,
            color=STYLE["secondary"],
        )
        return fig

    def fold_heatmap(fold, metric):
        """The median FOLD matrix, languages grouped by family."""
        order = sorted(
            fold.index, key=lambda lang: (constants.lang2family.get(lang, "~"), lang)
        )
        fold = fold.loc[order, order]
        cmap = LinearSegmentedColormap.from_list("fold", STYLE["ramp"])
        cmap.set_bad(STYLE["surface"])  # the diagonal (self-pairs) is not data

        size = max(4.2, 0.17 * len(order) + 1.6)
        fig, ax = plt.subplots(figsize=(size + 1.0, size), layout="constrained")
        _chrome(fig, ax)
        image = ax.imshow(fold.to_numpy(float), cmap=cmap)
        ax.set_xticks(range(len(order)), order, rotation=90)
        ax.set_yticks(range(len(order)), order)
        for side in ("left", "bottom"):
            ax.spines[side].set_visible(False)

        # A thin surface-coloured gap between language families.
        families = [constants.lang2family.get(lang, "~") for lang in order]
        for i in range(1, len(order)):
            if families[i] != families[i - 1]:
                ax.axhline(i - 0.5, color=STYLE["surface"], linewidth=2)
                ax.axvline(i - 0.5, color=STYLE["surface"], linewidth=2)

        bar = fig.colorbar(image, ax=ax, shrink=0.8)
        bar.outline.set_visible(False)
        bar.ax.tick_params(
            colors=STYLE["axis"], labelcolor=STYLE["secondary"], labelsize=8
        )
        bar.set_label(
            f"{METRIC_LABEL[metric]} ({FOLD_KIND[metric]})",
            fontsize=9,
            color=STYLE["secondary"],
        )
        ax.set_title(
            "Median over sentences, grouped by family",
            loc="left",
            fontsize=10,
            color=STYLE["primary"],
        )
        return fig

    return fold_heatmap, scatter_figure


@app.cell
def _(
    analysis,
    band_pick,
    fold_heatmap,
    langs,
    layer_pick,
    measure_pick,
    metric_pick,
    mo,
    outputs,
    plt,
    scatter_figure,
):
    _fold = analysis.fold_matrix(
        outputs[(band_pick.value, metric_pick.value, layer_pick.value)], langs
    )
    _pairs = analysis.baseline_pairs(_fold)[measure_pick.value].dropna()

    if len(_pairs) < 2:
        _scatter = mo.callout(
            mo.md("This measure covers fewer than two pairs for this corpus."),
            kind="warn",
        )
    else:
        _scatter = scatter_figure(_pairs, measure_pick.value, metric_pick.value)
    _heatmap = fold_heatmap(_fold, metric_pick.value)

    _view = mo.vstack(
        [
            mo.hstack([_scatter, _heatmap], justify="start", align="start"),
            mo.accordion(
                {
                    "Values plotted": mo.ui.table(
                        _pairs.reset_index(), selection=None, show_download=True
                    )
                }
            ),
        ]
    )
    plt.close("all")
    _view
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 6. The results file

    `run_pipeline.py` writes the same correlations as plain markdown tables;
    this is what it would write for the run above.
    """)
    return


@app.cell
def _(config, io, langs, mo, outputs, write_results):
    _buffer = io.StringIO()
    write_results(config, langs, outputs, _buffer)
    mo.accordion({config.get_output_filename(): mo.plain_text(_buffer.getvalue())})
    return


if __name__ == "__main__":
    app.run()
