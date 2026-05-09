from pathlib import Path

import pandas as pd

from src.utils.io import save_json
from src.utils.logger import get_logger


class DataValidation:
    def __init__(self) -> None:
        self.logger = get_logger(self.__class__.__name__)

    def run(self, params: dict) -> dict[str, str]:
        config = params["data_validation"]
        dataset = pd.read_csv(Path(config["input_path"]))

        required_columns = config["required_columns"]
        missing_columns = [column for column in required_columns if column not in dataset.columns]
        null_count = int(dataset.isnull().sum().sum())
        invalid_target_values = sorted(set(dataset["churn"]) - {0, 1})

        report = {
            "rows": int(len(dataset)),
            "missing_columns": missing_columns,
            "null_values": null_count,
            "invalid_target_values": invalid_target_values,
            "status": "pass",
        }

        if missing_columns or null_count > 0 or invalid_target_values:
            report["status"] = "fail"

        report_path = Path(config["report_path"])
        save_json(report_path, report)

        if report["status"] == "fail":
            raise ValueError(f"Data validation failed: {report}")

        output_path = Path(config["output_path"])
        output_path.parent.mkdir(parents=True, exist_ok=True)
        dataset.to_csv(output_path, index=False)

        self.logger.info("Validation passed. Report saved to %s", report_path)
        return {"validated_data_path": str(output_path), "report_path": str(report_path)}
