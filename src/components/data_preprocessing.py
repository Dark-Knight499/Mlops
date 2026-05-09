from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src.utils.logger import get_logger


class DataPreprocessing:
    def __init__(self) -> None:
        self.logger = get_logger(self.__class__.__name__)

    def run(self, params: dict) -> dict[str, str]:
        config = params["data_preprocessing"]
        dataset = pd.read_csv(Path(config["input_path"]))

        dataset = dataset.drop_duplicates().reset_index(drop=True)

        train_frame, test_frame = train_test_split(
            dataset,
            test_size=config["test_size"],
            random_state=params["random_state"],
            stratify=dataset["churn"],
        )

        train_path = Path(config["train_output_path"])
        test_path = Path(config["test_output_path"])
        train_path.parent.mkdir(parents=True, exist_ok=True)
        test_path.parent.mkdir(parents=True, exist_ok=True)

        train_frame.to_csv(train_path, index=False)
        test_frame.to_csv(test_path, index=False)

        self.logger.info("Saved train (%s) and test (%s) datasets", len(train_frame), len(test_frame))
        return {"train_path": str(train_path), "test_path": str(test_path)}
