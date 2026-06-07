from collections.abc import Callable

from src.components.data_ingestion import DataIngestion
from src.components.data_preprocessing import DataPreprocessing
from src.components.data_validation import DataValidation
from src.components.feature_engineering import FeatureEngineering
from src.components.model_evaluation import ModelEvaluation
from src.components.model_training import ModelTraining
from src.utils.logger import get_logger


class PipelineRunner:
    def __init__(self, params: dict) -> None:
        self.params = params
        self.logger = get_logger(self.__class__.__name__)
        self.stages: dict[str, Callable[[dict], dict[str, str]]] = {
            "data_ingestion": DataIngestion().run,
            "data_validation": DataValidation().run,
            "data_preprocessing": DataPreprocessing().run,
            "feature_engineering": FeatureEngineering().run,
            "model_training": ModelTraining().run,
            "model_evaluation": ModelEvaluation().run,
        }

    def run_stage(self, stage: str) -> dict[str, str]:
        self.logger.info("Running stage: %s", stage)
        if stage not in self.stages:
            raise ValueError(f"Unknown stage: {stage}")
        output = self.stages[stage](self.params)
        self.logger.info("Completed stage: %s", stage)
        return output

    def run_all(self) -> None:
        for stage in self.stages:
            self.run_stage(stage)
