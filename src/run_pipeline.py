"""
FOLD unified pipeline entry point.

Replaces the separate pipeline.py / embed_pipeline.py scripts. The
experiment's steps live in experiment.py.

Usage:
    uv run fold-pipeline --signal-mode likelihood [OPTIONS]
    uv run fold-pipeline --signal-mode embedding  [OPTIONS]
    uv run python src/run_pipeline.py --help
"""

from pathlib import Path

from experiment import configure_torch, run_experiment, write_results
from paths import auto_device
from pipeline_options import PipelineOptions


def main():
    configure_torch()

    device = auto_device()
    print("Using device:", device)

    config = PipelineOptions.from_args()
    config.print_config()

    langs, fold = run_experiment(config, device)

    output_path = Path(config.get_output_filename())
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w+", encoding="utf-8") as f:
        write_results(config, langs, fold, f)


if __name__ == "__main__":
    main()
