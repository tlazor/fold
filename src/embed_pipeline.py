from pathlib import Path

from joblib import Memory
from sklearn.pipeline import Pipeline
import torch

from EmbedTransformer import EmbedTransformer
from MetricTransformer import MetricTransformer, compute_overlaps, kl_divergence_matrix, mae_matrix
from PsdEstimator import PsdEstimator
from PsdNormalizer import PsdNormalizer
from SampleTokens import SampleTokens
from SpectralTransformer import SpectralTransformer
from TokenTransform import TokenTransform
from TsvToDataFrame import TsvToDataFrame
import fold_globals
from pipeline import analyze_output


if __name__ == "__main__":
    cachedir = Path(".cache/joblib")
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

    from transformers import AutoTokenizer

    mask_token_id = AutoTokenizer.from_pretrained(
        "bert-base-multilingual-cased", clean_up_tokenization_spaces=True
    ).mask_token_id

    use_spectra = True
    straight_spectra = True
    likelihood_pipeline_components = [
        ("load_tsv", TsvToDataFrame(Path("data/XNLI-15way/xnli.15way.orig.tsv"))),
        ("tokenize", TokenTransform()),
        ("sample", SampleTokens(num_samples=600, minimum_tokens=20, seed=0)),
        ("embeddings", EmbedTransformer(mask_token_id=mask_token_id)),
        
    ]
    spectra_component = [*(
            # if is_spectra is True, we add just the SpectralTransformer
            [("spectra", SpectralTransformer())]
            if straight_spectra
            # otherwise, we add the two PSD-related transforms
            else [
                ("est_psd", PsdEstimator(nperseg=56*2-1, axis=1)),
                ("norm_psd", PsdNormalizer(axis=1))
            ]
        )]

    metric_funs = [compute_overlaps, kl_divergence_matrix, mae_matrix]
    for fun in metric_funs:
        metric_transformer = MetricTransformer(name=fun.__name__, metric_fun=fun)
        metric_component = (metric_transformer.name, metric_transformer)
        pipeline = Pipeline(
            likelihood_pipeline_components + (spectra_component if use_spectra else []) + [metric_component],
            memory=pipeline_memory,
            verbose=False,
        )

        # pass None because TSVToDataFrame ignores X and reads from file_path
        output = pipeline.fit_transform(None)

        print(metric_transformer.name, output.shape)

        analyze_output(output)