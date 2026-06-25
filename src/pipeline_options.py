"""
Configuration for the FOLD pipeline.

Use PipelineOptions.from_args() to parse CLI arguments at the entry point,
or PipelineOptions(**kwargs) for programmatic / test use.
"""

import argparse
import json
from pathlib import Path


class PipelineOptions:
    def __init__(
        self,
        dataset="xnli",
        model="bert",
        signal_mode="likelihood",
        spectral_mode="welch",
        layers=None,
        num_bands=1,
        use_cache=False,
        analyze_pearson_contrib=False,
        output_dir=".",
    ):
        """
        Parameters
        ----------
        dataset : {"xnli", "bible", "un6"}
        model : {"bert", "xlmr"}
        signal_mode : {"likelihood", "embedding"}
            likelihood — masked token log-probabilities (LikelihoodEstimator)
            embedding  — hidden-layer embeddings (EmbedTransformer), loops over layers
        spectral_mode : {"welch", "fft", "none"}
            welch — Welch PSD (default)
            fft   — circular-averaged FFT power spectrum
            none  — raw token signal, no spectral transform
        layers : list of int
            Hidden layers to extract (embedding mode only).
        num_bands : int
            Number of equal-width frequency sub-bands.
        use_cache : bool
            Load cached pipeline outputs instead of recomputing.
        analyze_pearson_contrib : bool
            Plot per-pair Pearson contribution heatmaps.
        output_dir : str or Path
            Directory for output .txt and .json files.
        """
        if dataset not in {"xnli", "bible", "un6"}:
            raise ValueError(f"dataset must be 'xnli', 'bible', or 'un6', got {dataset!r}")
        if model not in {"bert", "xlmr"}:
            raise ValueError(f"model must be 'bert' or 'xlmr', got {model!r}")
        if signal_mode not in {"likelihood", "embedding"}:
            raise ValueError(f"signal_mode must be 'likelihood' or 'embedding', got {signal_mode!r}")
        if spectral_mode not in {"welch", "fft", "none"}:
            raise ValueError(f"spectral_mode must be 'welch', 'fft', or 'none', got {spectral_mode!r}")

        self.dataset = dataset
        self.model = model
        self.signal_mode = signal_mode
        self.spectral_mode = spectral_mode
        self.layers = layers if layers is not None else [12]
        self.num_bands = num_bands
        self.use_cache = use_cache
        self.analyze_pearson_contrib = analyze_pearson_contrib
        self.output_dir = Path(output_dir)

        if num_bands > 1:
            import numpy as np
            beginning_freqs = np.linspace(0, 1, num=num_bands, endpoint=False)
            self.freq_bands = list(zip(
                beginning_freqs, np.linspace(beginning_freqs[1], 1, num=num_bands)
            ))
        else:
            self.freq_bands = [(0, 1)]

    @classmethod
    def from_args(cls):
        """Parse configuration from sys.argv."""
        parser = argparse.ArgumentParser(
            description="FOLD (Fourier Overlap Linguistic Distance) pipeline"
        )
        parser.add_argument(
            "--dataset",
            choices=["xnli", "bible", "un6"],
            default="xnli",
            help="Parallel corpus to use (default: xnli)",
        )
        parser.add_argument(
            "--model",
            choices=["bert", "xlmr"],
            default="bert",
            help="Multilingual model (default: bert)",
        )
        parser.add_argument(
            "--spectral-mode",
            choices=["welch", "fft", "none"],
            default="welch",
            dest="spectral_mode",
            help=(
                "Spectral transform: welch PSD (default), "
                "circular FFT, or none (raw token signal)"
            ),
        )
        parser.add_argument(
            "--layers",
            type=int,
            nargs="+",
            default=[12],
            help="Hidden layers to extract for the embedding pipeline (default: 12)",
        )
        parser.add_argument(
            "--num-bands",
            type=int,
            default=1,
            dest="num_bands",
            help="Number of frequency sub-bands to analyse (default: 1 = full spectrum)",
        )
        parser.add_argument(
            "--use-cache",
            action="store_true",
            dest="use_cache",
            help="Load cached pipeline outputs instead of recomputing",
        )
        parser.add_argument(
            "--analyze-pearson-contrib",
            action="store_true",
            dest="analyze_pearson_contrib",
            help="Plot per-pair Pearson contribution heatmaps",
        )
        parser.add_argument(
            "--signal-mode",
            choices=["likelihood", "embedding"],
            default="likelihood",
            dest="signal_mode",
            help="Signal extraction mode: likelihood (default) or embedding",
        )
        parser.add_argument(
            "--output-dir",
            default=".",
            dest="output_dir",
            help="Directory for output files (default: current directory)",
        )
        args = parser.parse_args()
        return cls(
            dataset=args.dataset,
            model=args.model,
            signal_mode=args.signal_mode,
            spectral_mode=args.spectral_mode,
            layers=args.layers,
            num_bands=args.num_bands,
            use_cache=args.use_cache,
            analyze_pearson_contrib=args.analyze_pearson_contrib,
            output_dir=args.output_dir,
        )

    # ------------------------------------------------------------------
    # Derived properties
    # ------------------------------------------------------------------

    @property
    def model_name(self):
        return (
            "bert-base-multilingual-cased"
            if self.model == "bert"
            else "FacebookAI/xlm-roberta-base"
        )

    @property
    def use_bert(self):
        return self.model == "bert"

    @property
    def short_model_name(self):
        return self.model

    @property
    def prefix(self):
        return self.dataset

    @property
    def cachedir(self):
        return Path(f".cache/joblib/tmp/{self.model}")

    @property
    def mask_token_id(self):
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained(
            self.model_name, clean_up_tokenization_spaces=True
        ).mask_token_id

    def get_output_filename(self, pipeline_type=None):
        pt = pipeline_type if pipeline_type is not None else self.signal_mode
        stem = f"{self.dataset}_{self.model}_{pt}_output"
        return str(self.output_dir / f"{stem}.txt")

    # ------------------------------------------------------------------
    # Config serialisation
    # ------------------------------------------------------------------

    def to_dict(self):
        return {
            "dataset": self.dataset,
            "model": self.model,
            "model_name": self.model_name,
            "signal_mode": self.signal_mode,
            "spectral_mode": self.spectral_mode,
            "layers": self.layers,
            "num_bands": self.num_bands,
            "freq_bands": [list(b) for b in self.freq_bands],
            "use_cache": self.use_cache,
            "analyze_pearson_contrib": self.analyze_pearson_contrib,
            "output_dir": str(self.output_dir),
        }

    def save(self, path):
        """Write resolved config as JSON alongside an output file."""
        Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    def print_config(self, file=None):
        print(json.dumps(self.to_dict(), indent=2), file=file, flush=True)
