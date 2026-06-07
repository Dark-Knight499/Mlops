from pathlib import Path

import pandas as pd

from src.utils.logger import get_logger


class FeatureEngineering:
    def __init__(self) -> None:
        self.logger = get_logger(self.__class__.__name__)

    @staticmethod
    def _transform(frame: pd.DataFrame) -> pd.DataFrame:
        transformed = frame.copy()
        safe_tenure = transformed["tenure_months"].clip(lower=1)
        transformed["avg_charge_per_month"] = (transformed["total_charges"] / safe_tenure).round(2)
        transformed["support_to_tenure"] = (transformed["support_tickets"] / safe_tenure).round(4)
        transformed["delay_ticket_interaction"] = transformed["payment_delay_days"] * transformed["support_tickets"]
        return transformed

    def run(self, params: dict) -> dict[str, str]:
        config = params["feature_engineering"]
        train_frame = pd.read_csv(Path(config["train_input_path"]))
        test_frame = pd.read_csv(Path(config["test_input_path"]))

        train_features = self._transform(train_frame)
        test_features = self._transform(test_frame)

        train_path = Path(config["train_output_path"])
        test_path = Path(config["test_output_path"])
        train_path.parent.mkdir(parents=True, exist_ok=True)
        test_path.parent.mkdir(parents=True, exist_ok=True)

        train_features.to_csv(train_path, index=False)
        test_features.to_csv(test_path, index=False)

        self.logger.info("Feature engineering completed")
        return {"train_features_path": str(train_path), "test_features_path": str(test_path)}
