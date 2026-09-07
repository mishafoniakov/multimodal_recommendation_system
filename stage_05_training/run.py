"""
О документе
-----------
Источник: `learning_pipeline.csv` (пары и четыре метрики близости).
Цель: отличить связанную пару (`target_y = 1`) от случайной.
Результат: `<model>_model.bin`, ROC-AUC на тесте, `roc_<model>.png`.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from common.logging_setup import setup_logging
from config.settings import Settings
from stage_05_training.dataset import LearningDataset, VECTOR_COLUMNS
from stage_05_training.trainers import TRAINERS, CatBoostTrainer

logger = logging.getLogger(__name__)


def build_trainer(name: str, settings: Settings):
    if name == "catboost":
        return CatBoostTrainer(
            iterations=settings.catboost_iterations,
            learning_rate=settings.catboost_learning_rate,
            depth=settings.catboost_depth,
            scale_pos_weight=settings.scale_pos_weight,
        )
    return TRAINERS[name]()


def run(source: str, model_name: str) -> None:
    settings = Settings(source=source)
    dataset = LearningDataset(random_seed=settings.random_seed)
    frame = dataset.load(settings.path("learning"))
    splits = dataset.split(frame)
    target = splits["target"]

    x_train, y_train = dataset.xy(splits["train"], target, drop_query=True)
    x_val, y_val = dataset.xy(splits["val"], target, drop_query=True)
    x_test, y_test = dataset.xy(splits["test"], target, drop_query=True)

    # sklearn/xgboost/lgbm не едят колонки-списки — раскладываем векторы.
    if model_name != "catboost" and any(col in x_train.columns for col in VECTOR_COLUMNS):
        x_train = dataset.flatten_vector_columns(x_train)
        x_val = dataset.flatten_vector_columns(x_val)
        x_test = dataset.flatten_vector_columns(x_test)

    trainer = build_trainer(model_name, settings)
    q_train = splits["train"]["query"] if "query" in splits["train"].columns else None
    q_val = splits["val"]["query"] if "query" in splits["val"].columns else None
    trainer.fit(x_train, y_train, x_val, y_val, q_train=q_train, q_val=q_val)

    plots_dir = settings.output("plots")
    scores = trainer.evaluate(x_test, y_test, plots_dir)
    trainer.save(settings.output(f"{model_name}_model.bin"))
    settings.output("metrics.json").write_text(json.dumps({model_name: scores}, indent=2), encoding="utf-8")
    logger.info("Этап 5 готов")


if __name__ == "__main__":
    setup_logging()
    parser = argparse.ArgumentParser(description="Этап 5. Обучение")
    parser.add_argument("--source", choices=["yandex", "flickr"], default="yandex")
    parser.add_argument("--model", choices=list(TRAINERS), default="catboost")
    args = parser.parse_args()
    run(args.source, args.model)
