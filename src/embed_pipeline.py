from pathlib import Path
import pickle
import os

from joblib import Memory
from sklearn.pipeline import Pipeline
import torch
import numpy as np
from transformers import AutoTokenizer

from BibleTransformer import BibleTransformer
from EmbedTransformer import EmbedTransformer
from BandSelectTransformer import BandSelectTransformer
from MetricTransformer import (
    MetricTransformer,
    coherence_matrix,
    compute_overlaps,
    kl_divergence_matrix,
    mae_matrix,
)
from NoOpTransformer import NoOpTransformer
from PsdEstimator import PsdEstimator
from PsdNormalizer import PsdNormalizer
from SampleTokens import SampleTokens
from SpectralTransformer import SpectralTransformer
from TokenTransform import TokenTransform
from TsvToDataFrame import TsvToDataFrame
from Un6Transformer import Un6Transformer
import constants
import fold_globals
from pipeline import analyze_output, get_langs
from pipeline_options import PipelineOptions
from functools import partial

if __name__ == "__main__":
    torch.cuda.empty_cache()

    torch.set_float32_matmul_precision("high")
    torch._dynamo.config.capture_scalar_outputs = True

    if torch.cuda.is_available():
        fold_globals.DEVICE = torch.device("cuda")
    elif torch.backends.mps.is_available():
        fold_globals.DEVICE = torch.device("mps")
    else:
        fold_globals.DEVICE = torch.device("cpu")
    print("Using device:", fold_globals.DEVICE)

    config = PipelineOptions.from_args()

    pipeline_memory = Memory(config.cachedir, verbose=0)
    config.print_config()

    langs = get_langs(config.dataset, config.use_bert)
    likelihood_pipeline_components = [
        ("load_bible", BibleTransformer(Path("data/aligned"), langs=langs))
        if config.dataset == "bible"
        else ("load_un6", Un6Transformer(Path("data/6way"), langs=langs, nrows=2000))
        if config.dataset == "un6"
        else ("load_tsv", TsvToDataFrame(Path("data/XNLI-15way/xnli.15way.orig.tsv"))),
        ("tokenize", TokenTransform(model_name=config.model_name)),
        ("sample", SampleTokens(num_samples=600, minimum_tokens=20, seed=0)),
    ]
    spectra_component = [
        *(
            [("noop", NoOpTransformer())]
            if config.spectral_mode == "none"
            else [("spectra", SpectralTransformer())]
            if config.spectral_mode == "fft"
            else [
                ("est_psd", PsdEstimator(nperseg=56 * 2 - 1, axis=1)),
                ("norm_psd", PsdNormalizer(axis=1)),
            ]
        )
    ]

    cache_dir = Path("./cache/pipeline_outputs")
    cache_dir.mkdir(parents=True, exist_ok=True)

    output_path = Path(config.get_output_filename("embedding"))
    config.save(output_path.with_suffix(".json"))

    f = open(output_path, "w+", encoding="utf-8")
    config.print_config(file=f)
    for band in config.freq_bands:
        print(f"{band=}", file=f)

        coherence_fun = partial(coherence_matrix, nperseg=10, freq_band=band)
        coherence_fun.__name__ = "coherence_fun"
        metric_funs = (
            [compute_overlaps, kl_divergence_matrix]
            if config.spectral_mode == "none"
            else [compute_overlaps, kl_divergence_matrix, coherence_fun]
        )

        band_component = (
            f"{band[0]:.3f}-{band[1]:.3f} selector",
            BandSelectTransformer(freq_band=band),
        )
        for fun in metric_funs:
            metric_transformer = MetricTransformer(
                name=fun.__name__,
                metric_fun=fun,
                verbose=True if fun == coherence_fun else False,
            )
            metric_component = (metric_transformer.name, metric_transformer)
            print(metric_transformer.name, file=f)

            for layer in config.layers:
                embed_component = (
                    "embeddings",
                    EmbedTransformer(
                        mask_token_id=config.mask_token_id, layer=layer, model_name=config.model_name
                    ),
                )

                cache_key = f"{config.dataset}_{config.model}_{band[0]:.3f}-{band[1]:.3f}_{fun.__name__}_layer{layer}.pkl"
                cache_path = cache_dir / cache_key

                if config.use_cache and cache_path.exists():
                    print(f"Loading cached output from {cache_path}")
                    with open(cache_path, "rb") as cache_file:
                        output = pickle.load(cache_file)
                else:
                    pipeline = Pipeline(
                        likelihood_pipeline_components
                        + [embed_component]
                        + (spectra_component if fun != coherence_fun else [])
                        + ([band_component] if fun != coherence_fun else [])
                        + [metric_component],
                        memory=pipeline_memory,
                        verbose=True,
                    )

                    output = pipeline.fit_transform(None)

                    print(f"Saving output to cache at {cache_path}")
                    with open(cache_path, "wb") as cache_file:
                        pickle.dump(output, cache_file)

                print(f"{layer=}", file=f)
                analyze_output(output, langs, f=f, model_name=config.model_name,
                               flag_analyze_pearson_contrib=config.analyze_pearson_contrib)
    f.close()
