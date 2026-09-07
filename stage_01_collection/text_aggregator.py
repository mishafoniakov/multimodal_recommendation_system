"""Этап 1. Сборка текстов: Яндекс JSON и подписи Flickr."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

from common.io import JsonlStore

logger = logging.getLogger(__name__)


class TextAggregator:
    """Приводит тексты к словарю {имя_картинки: текст}."""

    @staticmethod
    def from_yandex_results(result_files: list[Path]) -> dict[str, str]:
        texts = {}
        for path in result_files:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(payload, list) and payload and isinstance(payload[0], list):
                payload = payload[0]
            for item in payload:
                try:
                    pieces = [list(part.values())[0] for part in item["image_text"]]
                    texts[item["image_name"]] = " ".join(pieces)
                except (KeyError, TypeError):
                    continue
        logger.info("Собрано %s текстов Яндекса", len(texts))
        return texts

    @staticmethod
    def from_flickr_csv(csv_path: Path) -> dict[str, str]:
        frame = pd.read_csv(csv_path, delimiter="|")
        frame = frame[["image_name", " comment"]].dropna()
        frame["comment"] = frame[" comment"].map(TextAggregator._clean_caption)
        joined = (
            frame.groupby("image_name")["comment"]
            .agg(lambda values: " ".join(values))
            .to_dict()
        )
        logger.info("Собрано %s подписей Flickr", len(joined))
        return joined

    @staticmethod
    def _clean_caption(text: str) -> str:
        text = text[1:] if text.startswith(" ") else text
        text = text.replace(" .", ".").replace(" ,", ",")
        text = ". ".join(part.strip().lower() for part in text.split(","))
        text = ". ".join(part.strip().capitalize() for part in text.split("."))
        return text.replace("  ", " ")

    @staticmethod
    def save(texts: dict[str, str], out_json: Path) -> None:
        JsonlStore.write(out_json, [{"img": image, "txt": text} for image, text in texts.items()])
