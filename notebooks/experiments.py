import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import io
    import os

    import marimo as mo

    import paths

    # Use the models in data/models (as the Docker image does) rather than
    # downloading them; must be set before transformers is imported.
    os.environ.setdefault("HF_HOME", str(paths.DATA_DIR / "models"))

    import matplotlib.pyplot as plt

    import experiment
    import report
    from analysis import (
        baseline_pairs,
        correlate_with_baselines,
        fold_matrix,
    )
    from pipeline_options import PipelineOptions

    return (
        PipelineOptions,
        baseline_pairs,
        correlate_with_baselines,
        experiment,
        fold_matrix,
        io,
        mo,
        plt,
        report,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # FOLD experiments

    Do multilingual language models see languages as close in the way linguists
    do? **FOLD** (Fourier Overlap Linguistic Distance) reads a signal off the
    model for every token of a sentence, in many languages at once, and
    compares the frequency content of those signals between languages.
    Taken over hundreds of sentences, that gives the model's picture of how far
    apart languages are, which we compare with independent measures: how well
    speakers understand each other, how much vocabulary and sound they share,
    how alike their grammars are, and how long English speakers take to learn
    them.

    ## The experiment

    Choose a corpus of translated sentences, a model, and which signal to read
    from it, then press **Run**. The code below is the whole experiment; the
    mechanics of each step live in `src/experiment.py`. Runs are cached
    sentence by sentence, so repeating one takes seconds; a new one takes a few
    minutes on the GPU.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    # Starting values can come from the command line (the run_pipeline.py
    # flags), e.g. to render a report without the editor:
    #   marimo export html notebooks/experiments.py -o report.html -- --run --dataset bible
    _args = mo.cli_args()
    auto_run = "run" in _args or mo.app_meta().mode == "script"

    def _dropdown(options, flag, default, label):
        labels = {value: key for key, value in options.items()}
        return mo.ui.dropdown(
            options, value=labels[_args.get(flag, default)], label=label
        )

    dataset = _dropdown(
        {"XNLI (15 languages)": "xnli", "Bible": "bible", "UN (6 languages)": "un6"},
        "dataset",
        "xnli",
        "Corpus",
    )
    model = _dropdown({"mBERT": "bert", "XLM-R": "xlmr"}, "model", "bert", "Model")
    signal_mode = _dropdown(
        {"token likelihood": "likelihood", "hidden state": "embedding"},
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
        label="Layers (hidden state only)",
    )
    num_bands = mo.ui.number(
        start=1, stop=8, value=int(_args.get("num-bands", 1)), label="Frequency bands"
    )
    run_button = mo.ui.run_button(label="Run")

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
def _(config, correlate_with_baselines, experiment, mo, run_requested):
    mo.stop(not run_requested, mo.md("_Choose the settings above and press **Run**._"))

    # The languages: those with text in the corpus that the model was trained on.
    langs = experiment.languages(config)

    # 600 sentences, each a translation of the same text into every language.
    sentences = experiment.sample_parallel_sentences(config, langs)

    # For each layer: a signal per token (how predictable it is, or its hidden
    # state), turned into spectra and compared for every pair of languages,
    # sentence by sentence. Each FOLD matrix holds one value per language pair.
    fold = {}
    for _layer in experiment.experiment_layers(config):
        _signals = experiment.extract_signals(sentences, config, _layer)
        for (_band, _metric), _matrices in experiment.compare_languages(
            _signals, config
        ).items():
            fold[_band, _metric, _layer] = _matrices

    # How well the model's median view agrees with each ground-truth measure.
    correlations = {
        key: correlate_with_baselines(matrices, langs) for key, matrices in fold.items()
    }
    return correlations, fold, langs


@app.cell(hide_code=True)
def _(config, correlations, experiment, mo, report):
    _bands = list(enumerate(config.freq_bands))
    _layers = experiment.experiment_layers(config)
    _tabs = {}
    for _name in experiment.metric_names(config):
        _sections = []
        for _k, _band in _bands:
            for _layer in _layers:
                if len(_bands) * len(_layers) > 1:
                    _title = f"Band {report.band_label(_band)} of Nyquist"
                    if _layer is not None:
                        _title += f", layer {_layer}"
                    _sections.append(mo.md(f"**{_title}**"))
                _sections.append(
                    mo.ui.table(
                        report.correlation_table(
                            correlations[_k, _name, _layer], _name
                        ),
                        selection=None,
                        show_download=True,
                    )
                )
        _tabs[report.METRIC_LABEL[_name]] = mo.vstack(_sections)

    mo.vstack(
        [
            mo.md(r"""
            ### How FOLD lines up with each measure

            Spectral overlap and coherence are similarities (higher means
            closer) and KL divergence is a distance, and so is each measure
            marked *distance*. When FOLD agrees with a measure, the correlation
            has the **agreement sign**: positive when both point the same way,
            negative when they point opposite ways. Significance: \* p < 0.05,
            \*\* p < 0.01, \*\*\* p < 0.001.
            """),
            mo.ui.tabs(_tabs),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Look at one comparison

    Every correlation above summarises a cloud of language pairs. Pick a FOLD
    metric and a measure to see that cloud, with the pairs furthest from the
    trend labelled, next to the full FOLD matrix grouped by language family.
    """)
    return


@app.cell
def _(
    band_choice,
    baseline_pairs,
    fold,
    fold_matrix,
    langs,
    layer_choice,
    measure_choice,
    metric_choice,
):
    # The model's view: the median FOLD matrix over all sentences.
    matrix = fold_matrix(
        fold[band_choice.value, metric_choice.value, layer_choice.value], langs
    )

    # Each language pair's FOLD value next to the chosen measure's value.
    pairs = baseline_pairs(matrix)[measure_choice.value].dropna()
    return matrix, pairs


@app.cell(hide_code=True)
def _(
    band_choice,
    config,
    experiment,
    layer_choice,
    matrix,
    measure_choice,
    metric_choice,
    mo,
    pairs,
    plt,
    report,
):
    _dark = mo.app_meta().theme == "dark"
    _controls = [metric_choice, measure_choice]
    if len(config.freq_bands) > 1:
        _controls.append(band_choice)
    if len(experiment.experiment_layers(config)) > 1:
        _controls.append(layer_choice)

    if len(pairs) < 2:
        _scatter = mo.callout(
            mo.md("This measure covers fewer than two pairs for this corpus."),
            kind="warn",
        )
    else:
        _scatter = report.scatter_figure(
            pairs, measure_choice.value, metric_choice.value, dark=_dark
        )
    _heatmap = report.fold_heatmap(matrix, metric_choice.value, dark=_dark)

    _view = mo.vstack(
        [
            mo.hstack(_controls, justify="start"),
            mo.hstack([_scatter, _heatmap], justify="start", align="start"),
            mo.accordion(
                {
                    "Values plotted": mo.ui.table(
                        pairs.reset_index(), selection=None, show_download=True
                    )
                }
            ),
        ]
    )
    plt.close("all")
    _view
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Appendix
    """)
    return


@app.cell(hide_code=True)
def _(config, experiment, mo, report):
    _langs = experiment.languages(config)

    mo.accordion(
        {
            "Which language pairs each measure covers": mo.vstack(
                [
                    mo.md(
                        "Each measure only covers some languages (the "
                        "intelligibility study tested Germanic, Romance and Slavic "
                        "languages; the lexical chart links nearby European "
                        "languages), so some correlations rest on few pairs."
                    ),
                    report.coverage_table(_langs),
                ]
            )
        }
    )
    return


@app.cell(hide_code=True)
def _(config, experiment, fold, io, langs, mo):
    _buffer = io.StringIO()
    experiment.write_results(config, langs, fold, _buffer)
    mo.accordion(
        {
            f"The results file ({config.get_output_filename()})": mo.plain_text(
                _buffer.getvalue()
            )
        }
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Mechanics

    The cells below turn the controls into settings; marimo runs cells in
    dependency order, so they can live down here.
    """)
    return


@app.cell(hide_code=True)
def _(
    PipelineOptions,
    auto_run,
    dataset,
    experiment,
    layers,
    model,
    num_bands,
    run_button,
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
    run_requested = run_button.value or auto_run
    experiment.set_verbose(False)  # progress bars only, no step logs
    return config, run_requested


@app.cell(hide_code=True)
def _(config, experiment, mo, report):
    metric_choice = mo.ui.dropdown(
        {report.METRIC_LABEL[name]: name for name in experiment.metric_names(config)},
        value=report.METRIC_LABEL["compute_overlaps"],
        label="FOLD metric",
    )
    measure_choice = mo.ui.dropdown(
        {label: name for name, label in report.MEASURE_LABEL.items()},
        value=report.MEASURE_LABEL["fsi"],
        label="Compared with",
    )
    band_choice = mo.ui.dropdown(
        {report.band_label(band): k for k, band in enumerate(config.freq_bands)},
        value=report.band_label(config.freq_bands[0]),
        label="Band",
    )
    _layers = experiment.experiment_layers(config)
    layer_choice = mo.ui.dropdown(
        {str(layer): layer for layer in _layers},
        value=str(_layers[0]),
        label="Layer",
    )
    return band_choice, layer_choice, measure_choice, metric_choice


if __name__ == "__main__":
    app.run()
