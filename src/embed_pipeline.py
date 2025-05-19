from pathlib import Path

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
from PsdEstimator import PsdEstimator
from PsdNormalizer import PsdNormalizer
from SampleTokens import SampleTokens
from SpectralTransformer import SpectralTransformer
from TokenTransform import TokenTransform
from TsvToDataFrame import TsvToDataFrame
import constants
import fold_globals
from pipeline import analyze_output, get_langs
from functools import partial

if __name__ == "__main__":
    cachedir = Path(".cache/joblib/tmp/output_coherence")
    pipeline_memory = Memory(cachedir, verbose=0)

    torch.set_float32_matmul_precision("high")
    torch._dynamo.config.capture_scalar_outputs = True

    if torch.cuda.is_available():
        fold_globals.DEVICE = torch.device("cuda")
    elif torch.backends.mps.is_available():
        fold_globals.DEVICE = torch.device("mps")
    else:
        fold_globals.DEVICE = torch.device("cpu")
    print("Using device:", fold_globals.DEVICE)

    use_bible = True
    use_spectra = True
    straight_spectra = False
    use_bert = True
    model_name = "bert-base-multilingual-cased" if use_bert else "FacebookAI/xlm-roberta-base"

    mask_token_id = AutoTokenizer.from_pretrained(
        model_name, clean_up_tokenization_spaces=True
    ).mask_token_id
    # layers = range(1, 12)
    layers = [12]
    num_bands = 1
    if num_bands > 1:
        beginning_freqs = np.linspace(0, 1, num=num_bands, endpoint=False)
        freq_bands = zip(
            beginning_freqs, np.linspace(beginning_freqs[1], 1, num=num_bands)
        )
    else:
        freq_bands = [(0, 1)]

    langs = get_langs(use_bible)
    likelihood_pipeline_components = [
        ("load_bible", BibleTransformer(Path("data/aligned"), langs=langs))
        if use_bible
        else ("load_tsv", TsvToDataFrame(Path("data/XNLI-15way/xnli.15way.orig.tsv"))),
        ("tokenize", TokenTransform(model_name=model_name)),
        ("sample", SampleTokens(num_samples=600, minimum_tokens=20, seed=0)),
    ]
    spectra_component = [
        *(
            # if is_spectra is True, we add just the SpectralTransformer
            [("spectra", SpectralTransformer())]
            if straight_spectra
            # otherwise, we add the two PSD-related transforms
            else [
                ("est_psd", PsdEstimator(nperseg=56 * 2 - 1, axis=1)),
                ("norm_psd", PsdNormalizer(axis=1)),
            ]
        )
    ]

    # metric_funs = [compute_overlaps, kl_divergence_matrix, mae_matrix, coherence_fun]
    # metric_funs = [kl_divergence_matrix]

    f = open(Path("./embedding_output.txt"), "w+", encoding="utf-8")
    for band in freq_bands:
        print(f"{band=}", file=f)

        coherence_fun = partial(coherence_matrix, nperseg=10, freq_band=band)
        coherence_fun.__name__ = "coherence_fun"
        metric_funs = [compute_overlaps, kl_divergence_matrix, coherence_fun]

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

            for layer in layers:
                embed_component = (
                    "embeddings",
                    EmbedTransformer(mask_token_id=mask_token_id, layer=layer, model_name=model_name),
                )

                pipeline = Pipeline(
                    likelihood_pipeline_components
                    + [embed_component]
                    + (
                        spectra_component
                        if use_spectra and fun != coherence_fun
                        else []
                    )
                    + ([band_component] if fun != coherence_fun else [])
                    + [metric_component],
                    memory=pipeline_memory,
                    verbose=False,
                )

                # pass None because TSVToDataFrame ignores X and reads from file_path
                output = pipeline.fit_transform(None)
                print(f"{layer=}", file=f)

                analyze_output(output, langs, f=f)
    f.close()
