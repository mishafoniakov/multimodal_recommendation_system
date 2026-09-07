"""Склейка CLIP и BERT по имени картинки (join, не двойной цикл)."""

from __future__ import annotations

import logging

import pandas as pd

from common.io import JsonlStore

logger = logging.getLogger(__name__)


class EmbeddingMerger:
    @staticmethod
    def merge(clip_json, bert_json, out_json) -> None:
        clip_rows = JsonlStore.read(clip_json)
        bert_rows = JsonlStore.read(bert_json)
        clip_frame = pd.DataFrame(clip_rows)
        bert_frame = pd.DataFrame(bert_rows)
        merged = clip_frame.merge(bert_frame, on="img", how="inner")
        JsonlStore.write(
            out_json,
            merged[["img", "clip", "bert"]].to_dict(orient="records"),
        )
        logger.info("Склеено %s CLIP+BERT записей", len(merged))
