"""Console entry point for the FOLD pipeline.

With an editable install (`uv sync`) the src/ layout means src/ is on
sys.path, so the flat source modules (run_pipeline, analysis, …) are
importable directly from here.
"""


def main():
    from run_pipeline import main as _main
    _main()
