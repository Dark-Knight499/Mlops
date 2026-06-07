import argparse
from pathlib import Path

from src.config.constants import DEFAULT_PARAMS_PATH
from src.pipeline.pipeline import PipelineRunner
from src.utils.io import load_params


STAGES = [
    "data_ingestion",
    "data_validation",
    "data_preprocessing",
    "feature_engineering",
    "model_training",
    "model_evaluation",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run churn MLOps pipeline stages")
    parser.add_argument("--stage", choices=["all", *STAGES], default="all")
    parser.add_argument("--params", default=str(DEFAULT_PARAMS_PATH))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    params = load_params(Path(args.params))

    runner = PipelineRunner(params)
    if args.stage == "all":
        runner.run_all()
    else:
        runner.run_stage(args.stage)


if __name__ == "__main__":
    main()
