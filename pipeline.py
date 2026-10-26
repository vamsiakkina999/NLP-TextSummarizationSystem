"""Run the text summarization analysis stages in their original order."""

from pathlib import Path
import runpy


PROJECT_DIR = Path(__file__).resolve().parent
PIPELINE_STAGES = (
    "01_data_exploration.py",
    "02_preprocessing.py",
    "03_scoring_and_evaluation.py",
    "04_weight_optimization.py",
    "05_textrank_and_test.py",
)


def run_pipeline():
    """Execute each stage while sharing the original script's global state."""
    shared_globals = {}
    for stage in PIPELINE_STAGES:
        stage_path = PROJECT_DIR / "pipeline_steps" / stage
        shared_globals.update(
            runpy.run_path(str(stage_path), init_globals=shared_globals)
        )


run_pipeline()
