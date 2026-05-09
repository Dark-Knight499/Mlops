import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.utils.io import save_json
from src.utils.logger import get_logger


class ModelEvaluation:
    def __init__(self) -> None:
        self.logger = get_logger(self.__class__.__name__)

    def run(self, params: dict) -> dict[str, str]:
        config = params["model_evaluation"]

        test_features = pd.read_csv(Path(config["input_path"]))
        x_test = test_features.drop(columns=["churn"])
        y_test = test_features["churn"]

        with Path(config["model_path"]).open("rb") as file:
            model = pickle.load(file)

        y_pred = model.predict(x_test)
        y_score = model.predict_proba(x_test)[:, 1]
        threshold = float(config.get("threshold", 0.5))
        y_pred = (y_score >= threshold).astype(int)

        metrics = {
            "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
            "precision": round(float(precision_score(y_test, y_pred)), 4),
            "recall": round(float(recall_score(y_test, y_pred)), 4),
            "f1_score": round(float(f1_score(y_test, y_pred)), 4),
            "roc_auc": round(float(roc_auc_score(y_test, y_score)), 4),
        }

        metrics_path = Path(config["metrics_output_path"])
        save_json(metrics_path, metrics)

        feature_importance_path = Path(config["feature_importance_plot_path"])
        feature_importance_path.parent.mkdir(parents=True, exist_ok=True)

        importance = pd.Series(model.feature_importances_, index=x_test.columns).sort_values(ascending=False)
        plt.figure(figsize=(9, 4.5))
        importance.head(8).plot(kind="bar", color="#4C6FFF")
        plt.title("Top Feature Importances")
        plt.ylabel("Importance")
        plt.tight_layout()
        plt.savefig(feature_importance_path, dpi=180)
        plt.close()

        confusion_matrix_path = Path(config["confusion_matrix_plot_path"])
        confusion_matrix_path.parent.mkdir(parents=True, exist_ok=True)
        disp = ConfusionMatrixDisplay.from_predictions(y_test, y_pred, cmap="Blues")
        disp.ax_.set_title("Churn Prediction Confusion Matrix")
        plt.tight_layout()
        plt.savefig(confusion_matrix_path, dpi=180)
        plt.close()

        self.logger.info("Evaluation complete with metrics: %s", metrics)
        return {
            "metrics_path": str(metrics_path),
            "feature_importance_plot_path": str(feature_importance_path),
            "confusion_matrix_plot_path": str(confusion_matrix_path),
        }
