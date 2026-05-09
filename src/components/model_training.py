import pickle
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from src.utils.io import save_json
from src.utils.logger import get_logger


class ModelTraining:
    def __init__(self) -> None:
        self.logger = get_logger(self.__class__.__name__)

    def run(self, params: dict) -> dict[str, str]:
        config = params["model_training"]
        train_features = pd.read_csv(Path(config["input_path"]))

        x_train = train_features.drop(columns=["churn"])
        y_train = train_features["churn"]

        model = RandomForestClassifier(
            n_estimators=config["n_estimators"],
            max_depth=config["max_depth"],
            random_state=params["random_state"],
            class_weight="balanced",
        )
        model.fit(x_train, y_train)

        model_path = Path(config["model_output_path"])
        model_path.parent.mkdir(parents=True, exist_ok=True)
        with model_path.open("wb") as file:
            pickle.dump(model, file)

        feature_columns_path = Path(config["feature_columns_path"])
        save_json(feature_columns_path, {"feature_columns": list(x_train.columns)})

        self.logger.info("Model trained and saved to %s", model_path)
        return {"model_path": str(model_path), "feature_columns_path": str(feature_columns_path)}
