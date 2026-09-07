"""Этап 5. Подготовка таблицы для обучения."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

METRIC_COLUMNS = ["clip_cossim", "clip_l2", "bert_cossim", "bert_l2"]
VECTOR_COLUMNS = ["img_clip_1", "img_clip_2", "img_bert_1", "img_bert_2"]


class LearningDataset:
    """Делит данные на train/val/test и масштабирует 4 метрики близости."""

    def __init__(self, random_seed: int = 13) -> None:
        self.random_seed = random_seed
        self.scaler = StandardScaler()

    def load(self, csv_path) -> pd.DataFrame:
        logger.info("Читаю %s", csv_path)
        return pd.read_csv(csv_path)

    def split(self, frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
        target_col = "target_y" if "target_y" in frame.columns else "target"
        if target_col not in frame.columns:
            raise ValueError("В таблице нет колонки target_y / target")

        work = frame.copy()
        if "query" in work.columns:
            work = work.sort_values("query")

        train, rest = train_test_split(work, test_size=0.4, random_state=self.random_seed)
        val, test = train_test_split(rest, test_size=0.5, random_state=self.random_seed)

        present_metrics = [col for col in METRIC_COLUMNS if col in train.columns]
        if present_metrics:
            train = train.copy()
            val = val.copy()
            test = test.copy()
            train[present_metrics] = self.scaler.fit_transform(train[present_metrics])
            val[present_metrics] = self.scaler.transform(val[present_metrics])
            test[present_metrics] = self.scaler.transform(test[present_metrics])

        return {"train": train, "val": val, "test": test, "target": target_col}

    @staticmethod
    def xy(frame: pd.DataFrame, target_col: str, drop_query: bool = True):
        drop_cols = [target_col]
        if drop_query and "query" in frame.columns:
            drop_cols.append("query")
        y = frame[target_col]
        x = frame.drop(columns=drop_cols)
        return x, y

    @staticmethod
    def flatten_vector_columns(frame: pd.DataFrame) -> pd.DataFrame:
        """Для sklearn: списки CLIP/BERT превращаем в отдельные колонки."""
        parts = []
        leftover = frame.copy()
        for column in VECTOR_COLUMNS:
            if column not in leftover.columns:
                continue
            stacked = np.vstack(leftover[column].map(_to_vector).to_list())
            names = [f"{column}_{i}" for i in range(stacked.shape[1])]
            parts.append(pd.DataFrame(stacked, columns=names, index=leftover.index))
            leftover = leftover.drop(columns=[column])
        if not parts:
            return leftover
        return pd.concat([leftover.reset_index(drop=True), *[part.reset_index(drop=True) for part in parts]], axis=1)


def _to_vector(value) -> np.ndarray:
    if isinstance(value, str):
        value = np.fromstring(value.strip("[]"), sep=",")
    return np.asarray(value, dtype=float).ravel()
