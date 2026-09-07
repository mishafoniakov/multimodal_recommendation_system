"""Этап 5. Модели: CatBoost, XGBoost, LightGBM, дерево и лес."""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
from sklearn.tree import DecisionTreeClassifier

logger = logging.getLogger(__name__)


class ModelTrainer:
    """Общие метрики и сохранение ROC. Конкретная модель — в наследнике."""

    name = "model"

    def __init__(self) -> None:
        self.model = None

    def fit(self, x_train, y_train, x_val=None, y_val=None, **kwargs):
        raise NotImplementedError

    def predict_proba(self, x) -> np.ndarray:
        return self.model.predict_proba(x)[:, 1]

    def evaluate(self, x_test, y_test, plots_dir: Path) -> dict[str, float]:
        proba = self.predict_proba(x_test)
        pred = (proba >= 0.5).astype(int)
        scores = {
            "accuracy": float(accuracy_score(y_test, pred)),
            "roc_auc": float(roc_auc_score(y_test, proba)),
        }
        plots_dir.mkdir(parents=True, exist_ok=True)
        fpr, tpr, _ = roc_curve(y_test, proba)
        plt.figure(figsize=(8, 5))
        plt.plot(fpr, tpr, label=f"AUC = {scores['roc_auc']:.3f}")
        plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"ROC: {self.name}")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(plots_dir / f"roc_{self.name}.png", dpi=120)
        plt.close()
        logger.info("%s  accuracy=%.4f  roc_auc=%.4f", self.name, scores["accuracy"], scores["roc_auc"])
        return scores

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        if hasattr(self.model, "save_model"):
            self.model.save_model(str(path))
        else:
            import joblib

            joblib.dump(self.model, path)
        logger.info("Модель сохранена: %s", path)


class CatBoostTrainer(ModelTrainer):
    name = "catboost"

    def __init__(self, iterations: int, learning_rate: float, depth: int, scale_pos_weight: float) -> None:
        super().__init__()
        from catboost import CatBoostClassifier

        self.model = CatBoostClassifier(
            iterations=iterations,
            learning_rate=learning_rate,
            max_depth=depth,
            loss_function="Logloss",
            scale_pos_weight=scale_pos_weight,
            early_stopping_rounds=50,
            custom_metric=["AUC", "Accuracy"],
            verbose=100,
        )

    def fit(self, x_train, y_train, x_val=None, y_val=None, q_train=None, q_val=None, **kwargs):
        from catboost import Pool

        train_pool = Pool(x_train, y_train, group_id=q_train)
        eval_set = Pool(x_val, y_val, group_id=q_val) if x_val is not None else None
        self.model.fit(train_pool, eval_set=eval_set, use_best_model=eval_set is not None)
        return self


class XGBoostTrainer(ModelTrainer):
    name = "xgboost"

    def fit(self, x_train, y_train, x_val=None, y_val=None, q_train=None, q_val=None, **kwargs):
        from xgboost import XGBClassifier

        self.model = XGBClassifier(
            n_estimators=2000,
            learning_rate=0.01,
            max_depth=6,
            scale_pos_weight=2.64,
            early_stopping_rounds=20,
            eval_metric="auc",
            n_jobs=-1,
        )
        fit_args = {}
        if x_val is not None:
            fit_args["eval_set"] = [(x_val, y_val)]
        self.model.fit(x_train, y_train, verbose=True, **fit_args)
        return self


class LightGBMTrainer(ModelTrainer):
    name = "lgbm"

    def fit(self, x_train, y_train, x_val=None, y_val=None, q_train=None, q_val=None, **kwargs):
        from lightgbm import LGBMClassifier

        self.model = LGBMClassifier(
            n_estimators=2000,
            learning_rate=0.01,
            max_depth=6,
            scale_pos_weight=2.64,
            n_jobs=-1,
        )
        fit_args = {}
        if x_val is not None:
            fit_args["eval_set"] = [(x_val, y_val)]
        self.model.fit(x_train, y_train, **fit_args)
        return self


class DecisionTreeTrainer(ModelTrainer):
    name = "decision_tree"

    def fit(self, x_train, y_train, x_val=None, y_val=None, q_train=None, q_val=None, **kwargs):
        self.model = DecisionTreeClassifier(random_state=13)
        self.model.fit(x_train, y_train)
        return self


class RandomForestTrainer(ModelTrainer):
    name = "random_forest"

    def fit(self, x_train, y_train, x_val=None, y_val=None, q_train=None, q_val=None, **kwargs):
        self.model = RandomForestClassifier(random_state=0, n_jobs=-1)
        self.model.fit(x_train, y_train)
        return self


TRAINERS = {
    "catboost": CatBoostTrainer,
    "xgboost": XGBoostTrainer,
    "lgbm": LightGBMTrainer,
    "dt": DecisionTreeTrainer,
    "rf": RandomForestTrainer,
}
