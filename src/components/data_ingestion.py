from pathlib import Path

import pandas as pd
from sklearn.datasets import make_classification

from src.utils.logger import get_logger


class DataIngestion:
    def __init__(self) -> None:
        self.logger = get_logger(self.__class__.__name__)

    def run(self, params: dict) -> dict[str, str]:
        config = params["data_ingestion"]
        random_state = params["random_state"]

        self.logger.info("Generating synthetic telecom churn data")
        features, target = make_classification(
            n_samples=config["n_samples"],
            n_features=8,
            n_informative=6,
            n_redundant=0,
            class_sep=2.0,
            flip_y=0.01,
            random_state=random_state,
            weights=[0.72, 0.28],
        )

        frame = pd.DataFrame(
            features,
            columns=[
                "tenure_months",
                "monthly_charges",
                "total_charges",
                "support_tickets",
                "contract_months",
                "streaming_usage_gb",
                "payment_delay_days",
                "service_calls_last_quarter",
            ],
        )

        frame["tenure_months"] = (frame["tenure_months"].abs() * 18 + 1).round().clip(1, 72)
        frame["monthly_charges"] = (frame["monthly_charges"].abs() * 20 + 30).round(2)
        frame["total_charges"] = (frame["monthly_charges"] * frame["tenure_months"]).round(2)
        frame["support_tickets"] = (frame["support_tickets"].abs() * 2).round().clip(0, 8)
        frame["contract_months"] = (frame["contract_months"].abs() * 6 + 6).round().clip(1, 24)
        frame["streaming_usage_gb"] = (frame["streaming_usage_gb"].abs() * 15 + 2).round(2)
        frame["payment_delay_days"] = (frame["payment_delay_days"].abs() * 4).round().clip(0, 25)
        frame["service_calls_last_quarter"] = (
            frame["service_calls_last_quarter"].abs() * 2
        ).round().clip(0, 10)
        frame["churn"] = target

        output_path = Path(config["output_path"])
        output_path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(output_path, index=False)

        self.logger.info("Saved %s rows to %s", len(frame), output_path)
        return {"data_path": str(output_path)}
