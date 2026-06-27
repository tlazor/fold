"""Absolute paths anchored to the project root.

Import from this module instead of using bare Path(".") strings so that
scripts run correctly regardless of the working directory.
"""

from pathlib import Path

# src/paths.py → src/ → project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
CACHE_DIR = PROJECT_ROOT / ".cache"


def auto_device():
    """Return the best available torch.device (lazy-imports torch)."""
    import torch  # noqa: PLC0415
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_hf_model(model_cls, model_name: str, device):
    """Load a HuggingFace model, enabling Flash Attention 2 on CUDA when available.

    Sets torch_dtype at load time (required by HF >=4.37 when attn_implementation is
    given) and falls back to standard attention if the architecture or installed
    transformers version does not support FA2.
    """
    import torch  # noqa: PLC0415
    kwargs = {}
    if device.type == "cuda":
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        kwargs["torch_dtype"] = dtype
        try:
            import flash_attn  # noqa: F401, PLC0415
            kwargs["attn_implementation"] = "flash_attention_2"
        except ImportError:
            pass
    try:
        return model_cls.from_pretrained(model_name, **kwargs)
    except (ImportError, ValueError):
        # Architecture doesn't support FA2 (e.g. mBERT on certain transformers versions).
        kwargs.pop("attn_implementation", None)
        return model_cls.from_pretrained(model_name, **kwargs)
