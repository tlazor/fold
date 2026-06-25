import sys
import json
import pytest
from pathlib import Path
from pipeline_options import PipelineOptions


# ===========================================================================
# Default construction
# ===========================================================================

class TestDefaults:
    def test_dataset_default(self):
        assert PipelineOptions().dataset == "xnli"

    def test_model_default(self):
        assert PipelineOptions().model == "bert"

    def test_signal_mode_default(self):
        assert PipelineOptions().signal_mode == "likelihood"

    def test_spectral_mode_default(self):
        assert PipelineOptions().spectral_mode == "welch"

    def test_layers_default(self):
        assert PipelineOptions().layers == [12]

    def test_num_bands_default(self):
        assert PipelineOptions().num_bands == 1

    def test_freq_bands_single_band(self):
        assert PipelineOptions().freq_bands == [(0, 1)]

    def test_use_cache_default(self):
        assert PipelineOptions().use_cache is False

    def test_analyze_pearson_contrib_default(self):
        assert PipelineOptions().analyze_pearson_contrib is False

    def test_output_dir_default(self):
        assert PipelineOptions().output_dir == Path(".")


# ===========================================================================
# Validation
# ===========================================================================

class TestValidation:
    def test_invalid_spectral_mode_raises(self):
        with pytest.raises(ValueError, match="spectral_mode"):
            PipelineOptions(spectral_mode="foobar")

    def test_invalid_dataset_raises(self):
        with pytest.raises(ValueError, match="dataset"):
            PipelineOptions(dataset="gutenberg")

    def test_invalid_model_raises(self):
        with pytest.raises(ValueError, match="model"):
            PipelineOptions(model="gpt4")

    def test_invalid_signal_mode_raises(self):
        with pytest.raises(ValueError, match="signal_mode"):
            PipelineOptions(signal_mode="transformer")


# ===========================================================================
# Derived properties
# ===========================================================================

class TestDerivedProperties:
    def test_model_name_bert(self):
        assert PipelineOptions(model="bert").model_name == "bert-base-multilingual-cased"

    def test_model_name_xlmr(self):
        assert PipelineOptions(model="xlmr").model_name == "FacebookAI/xlm-roberta-base"

    def test_use_bert_true(self):
        assert PipelineOptions(model="bert").use_bert is True

    def test_use_bert_false(self):
        assert PipelineOptions(model="xlmr").use_bert is False

    def test_short_model_name(self):
        assert PipelineOptions(model="bert").short_model_name == "bert"
        assert PipelineOptions(model="xlmr").short_model_name == "xlmr"

    def test_prefix_matches_dataset(self):
        assert PipelineOptions(dataset="bible").prefix == "bible"
        assert PipelineOptions(dataset="un6").prefix == "un6"
        assert PipelineOptions(dataset="xnli").prefix == "xnli"

    def test_cachedir_contains_model(self):
        p = PipelineOptions(model="bert").cachedir
        assert "bert" in str(p)

    def test_cachedir_contains_xlmr(self):
        p = PipelineOptions(model="xlmr").cachedir
        assert "xlmr" in str(p)


# ===========================================================================
# Output filenames
# ===========================================================================

class TestOutputFilenames:
    def test_likelihood_filename(self):
        name = PipelineOptions(dataset="bible", model="bert").get_output_filename("likelihood")
        assert "bible" in name
        assert "bert" in name
        assert "likelihood" in name
        assert name.endswith(".txt")

    def test_embedding_filename(self):
        name = PipelineOptions(dataset="un6", model="xlmr").get_output_filename("embedding")
        assert "un6" in name
        assert "xlmr" in name
        assert "embedding" in name

    def test_output_dir_respected(self, tmp_path):
        cfg = PipelineOptions(dataset="xnli", model="bert", output_dir=str(tmp_path))
        name = cfg.get_output_filename("likelihood")
        assert name.startswith(str(tmp_path))

    def test_output_filename_defaults_to_signal_mode(self):
        cfg = PipelineOptions(signal_mode="embedding")
        assert "embedding" in cfg.get_output_filename()

    def test_output_filename_explicit_type_overrides_signal_mode(self):
        cfg = PipelineOptions(signal_mode="embedding")
        assert "likelihood" in cfg.get_output_filename("likelihood")


# ===========================================================================
# Frequency bands
# ===========================================================================

class TestFreqBands:
    def test_single_band(self):
        bands = PipelineOptions(num_bands=1).freq_bands
        assert bands == [(0, 1)]

    def test_multiple_bands_count(self):
        bands = PipelineOptions(num_bands=4).freq_bands
        assert len(bands) == 4

    def test_multiple_bands_cover_full_range(self):
        import numpy as np
        bands = PipelineOptions(num_bands=4).freq_bands
        assert bands[0][0] == pytest.approx(0.0)
        assert bands[-1][1] == pytest.approx(1.0)

    def test_multiple_bands_non_overlapping(self):
        bands = PipelineOptions(num_bands=3).freq_bands
        for i in range(len(bands) - 1):
            assert bands[i][1] == pytest.approx(bands[i + 1][0], abs=1e-9)


# ===========================================================================
# Serialisation
# ===========================================================================

class TestSerialisation:
    def test_to_dict_has_required_keys(self):
        d = PipelineOptions().to_dict()
        required = {"dataset", "model", "model_name", "signal_mode", "spectral_mode",
                    "layers", "num_bands", "freq_bands", "use_cache",
                    "analyze_pearson_contrib", "output_dir"}
        assert required.issubset(d.keys())

    def test_to_dict_values_match(self):
        cfg = PipelineOptions(dataset="bible", model="xlmr", spectral_mode="fft",
                              layers=[6, 12], num_bands=2)
        d = cfg.to_dict()
        assert d["dataset"] == "bible"
        assert d["model"] == "xlmr"
        assert d["spectral_mode"] == "fft"
        assert d["layers"] == [6, 12]
        assert len(d["freq_bands"]) == 2

    def test_save_writes_valid_json(self, tmp_path):
        cfg = PipelineOptions(dataset="xnli", model="bert")
        out = tmp_path / "config.json"
        cfg.save(out)
        loaded = json.loads(out.read_text())
        assert loaded["dataset"] == "xnli"
        assert loaded["model"] == "bert"

    def test_print_config_produces_valid_json(self, capsys):
        PipelineOptions().print_config()
        captured = capsys.readouterr()
        loaded = json.loads(captured.out)
        assert "dataset" in loaded


# ===========================================================================
# CLI parsing (from_args)
# ===========================================================================

class TestFromArgs:
    def test_defaults(self, monkeypatch):
        monkeypatch.setattr(sys, "argv", ["pipeline"])
        cfg = PipelineOptions.from_args()
        assert cfg.dataset == "xnli"
        assert cfg.model == "bert"
        assert cfg.signal_mode == "likelihood"
        assert cfg.spectral_mode == "welch"

    def test_dataset_flag(self, monkeypatch):
        monkeypatch.setattr(sys, "argv", ["pipeline", "--dataset", "bible"])
        cfg = PipelineOptions.from_args()
        assert cfg.dataset == "bible"

    def test_model_flag(self, monkeypatch):
        monkeypatch.setattr(sys, "argv", ["pipeline", "--model", "xlmr"])
        cfg = PipelineOptions.from_args()
        assert cfg.model == "xlmr"

    def test_signal_mode_flag(self, monkeypatch):
        monkeypatch.setattr(sys, "argv", ["pipeline", "--signal-mode", "embedding"])
        cfg = PipelineOptions.from_args()
        assert cfg.signal_mode == "embedding"

    def test_spectral_mode_flag(self, monkeypatch):
        monkeypatch.setattr(sys, "argv", ["pipeline", "--spectral-mode", "none"])
        cfg = PipelineOptions.from_args()
        assert cfg.spectral_mode == "none"

    def test_layers_flag(self, monkeypatch):
        monkeypatch.setattr(sys, "argv", ["pipeline", "--layers", "6", "9", "12"])
        cfg = PipelineOptions.from_args()
        assert cfg.layers == [6, 9, 12]

    def test_num_bands_flag(self, monkeypatch):
        monkeypatch.setattr(sys, "argv", ["pipeline", "--num-bands", "4"])
        cfg = PipelineOptions.from_args()
        assert cfg.num_bands == 4
        assert len(cfg.freq_bands) == 4

    def test_use_cache_flag(self, monkeypatch):
        monkeypatch.setattr(sys, "argv", ["pipeline", "--use-cache"])
        cfg = PipelineOptions.from_args()
        assert cfg.use_cache is True

    def test_analyze_pearson_contrib_flag(self, monkeypatch):
        monkeypatch.setattr(sys, "argv", ["pipeline", "--analyze-pearson-contrib"])
        cfg = PipelineOptions.from_args()
        assert cfg.analyze_pearson_contrib is True

    def test_output_dir_flag(self, monkeypatch, tmp_path):
        monkeypatch.setattr(sys, "argv", ["pipeline", "--output-dir", str(tmp_path)])
        cfg = PipelineOptions.from_args()
        assert cfg.output_dir == tmp_path

    def test_combined_flags(self, monkeypatch):
        monkeypatch.setattr(sys, "argv", [
            "pipeline",
            "--dataset", "un6",
            "--model", "xlmr",
            "--spectral-mode", "fft",
            "--use-cache",
        ])
        cfg = PipelineOptions.from_args()
        assert cfg.dataset == "un6"
        assert cfg.model == "xlmr"
        assert cfg.spectral_mode == "fft"
        assert cfg.use_cache is True
