"""Этап 6. Топ-5 кандидатов: модель, CLIP и BERT."""

from __future__ import annotations

import logging

import pandas as pd
from sklearn.preprocessing import StandardScaler

from common.io import JsonlStore

logger = logging.getLogger(__name__)


class CandidateRanker:
    """Считает вероятность «пара связана» и берёт 5 ближайших картинок."""

    def __init__(self, n_candidates: int = 5) -> None:
        self.n_candidates = n_candidates

    def prepare_features(self, pairs: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        ids = pairs[["image", "candidate", "clip_cossim", "bert_cossim"]].copy()
        features = pairs[
            [
                "img_clip_1",
                "img_clip_2",
                "img_bert_1",
                "img_bert_2",
                "clip_cossim",
                "clip_l2",
                "bert_cossim",
                "bert_l2",
            ]
        ].copy()
        metric_cols = ["clip_cossim", "clip_l2", "bert_cossim", "bert_l2"]
        features[metric_cols] = StandardScaler().fit_transform(features[metric_cols])
        return ids, features

    def add_model_scores(self, ids: pd.DataFrame, features: pd.DataFrame, model_path) -> pd.DataFrame:
        from catboost import CatBoostClassifier

        model = CatBoostClassifier()
        model.load_model(str(model_path))
        proba = model.predict_proba(features)
        scored = ids.copy()
        scored["pred_1"] = proba[:, 1]
        return scored

    def top(self, frame: pd.DataFrame, score_col: str, out_json, larger_is_better: bool = True) -> dict:
        work = frame[frame["image"] != frame["candidate"]]
        ordered = work.sort_values(score_col, ascending=not larger_is_better)
        chosen = ordered.groupby("image", sort=False).head(self.n_candidates)
        result = chosen.groupby("image")["candidate"].apply(list).to_dict()
        JsonlStore.write(out_json, [{"img": image, "candidates": cands} for image, cands in result.items()])
        logger.info("Записано %s запросов в %s", len(result), out_json)
        return result
