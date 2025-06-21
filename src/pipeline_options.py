"""
Shared configuration options for pipeline and embed_pipeline scripts.
"""

from pathlib import Path
from transformers import AutoTokenizer


class PipelineOptions:
    """Configuration class for pipeline options."""
    
    def __init__(self):
        # Data source options
        self.use_bible = True
        self.use_un6 = False
        
        # Model options
        self.use_bert = True
        self.model_name = (
            "bert-base-multilingual-cased" if self.use_bert else "FacebookAI/xlm-roberta-base"
        )
        
        # Spectra options (mutually exclusive)
        self.no_spectra = False
        self.straight_spectra = False
        
        # Validate mutual exclusion
        self._validate_spectra_options()
        
        # Embedding pipeline specific options
        self.use_cache = False  # Set to False to force recomputation
        self.layers = [12]  # For embedding pipeline
        
        # Analysis options
        self.analyze_pearson_contrib = False  # Control whether to analyze Pearson contribution
        
        # Frequency bands
        self.num_bands = 1
        if self.num_bands > 1:
            import numpy as np
            beginning_freqs = np.linspace(0, 1, num=self.num_bands, endpoint=False)
            self.freq_bands = list(zip(
                beginning_freqs, np.linspace(beginning_freqs[1], 1, num=self.num_bands)
            ))
        else:
            self.freq_bands = [(0, 1)]
    
    def _validate_spectra_options(self):
        """Validate that only one spectra option is True."""
        spectra_options = [self.no_spectra, self.straight_spectra]
        true_count = sum(spectra_options)
        
        # 0 is fine, it means use welch's method
        if true_count > 1:
            raise ValueError(
                f"Exactly one spectra option must be True, but found {true_count} True values: "
                f"no_spectra={self.no_spectra}, "
                f"straight_spectra={self.straight_spectra}"
            )
    
    @property
    def short_model_name(self):
        """Get short model name for file naming."""
        return "bert" if self.use_bert else "xlmr"
    
    @property
    def prefix(self):
        """Get prefix for output files."""
        if self.use_bible:
            return "bible"
        elif self.use_un6:
            return "un6"
        else:
            return "xnli"
    
    @property
    def cachedir(self):
        """Get cache directory path."""
        return Path(f".cache/joblib/tmp/{self.short_model_name}")
    
    @property
    def mask_token_id(self):
        """Get mask token ID for the model."""
        return AutoTokenizer.from_pretrained(
            self.model_name, clean_up_tokenization_spaces=True
        ).mask_token_id
    
    def get_output_filename(self, pipeline_type="likelihood"):
        """Get output filename based on pipeline type."""
        if pipeline_type == "likelihood":
            return f"./{self.prefix}_{self.short_model_name}_likelihood_output.txt"
        elif pipeline_type == "embedding":
            return f"./{self.prefix}_{self.short_model_name}_embedding_output.txt"
        else:
            raise ValueError(f"Unknown pipeline type: {pipeline_type}")
    
    def print_config(self, file=None):
        """Print configuration options."""
        config_str = (
            f"{self.use_bible=}, {self.use_un6=}, "
            f"{self.no_spectra=}, {self.straight_spectra=}, "
            f"{self.use_bert=}, {self.model_name=}, "
            f"{self.layers=}, {self.num_bands=}, {self.use_cache=}, "
            f"{self.analyze_pearson_contrib=}"
        )
        print(config_str, file=file, flush=True)
        return config_str


# Default configuration instance
config = PipelineOptions() 