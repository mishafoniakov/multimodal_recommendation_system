"""Запуск этапа 3: BERT-эмбеддинги и склейка с CLIP."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from common.io import JsonlStore
from common.logging_setup import setup_logging
from config.settings import FILE_NAMES, Settings
from stage_01_collection.text_aggregator import TextAggregator
from stage_03_bert.bert_embedder import BertEmbedder
from stage_03_bert.merger import EmbeddingMerger

logger = logging.getLogger(__name__)


def load_texts(settings: Settings) -> dict[str, str]:
    if settings.source == "flickr":
        texts = TextAggregator.from_flickr_csv(settings.captions_csv())
        TextAggregator.save(texts, settings.output("image_big_text_dataset.json"))
        return texts
    return JsonlStore.to_dict(settings.yandex_text_json(), "img", "txt")


def run(source: str) -> None:
    settings = Settings(source=source)
    texts = load_texts(settings)
    embedder = BertEmbedder(
        settings.bert_model_id,
        max_length=settings.bert_max_length,
        batch_size=settings.batch_size,
    )
    bert_path = settings.output(FILE_NAMES[source]["bert"])
    embedder.embed(texts, bert_path)
    EmbeddingMerger.merge(
        settings.path("clip"),
        bert_path,
        settings.output(FILE_NAMES[source]["clip_bert"]),
    )


if __name__ == "__main__":
    setup_logging()
    parser = argparse.ArgumentParser(description="Этап 3. BERT-векторы")
    parser.add_argument("--source", choices=["yandex", "flickr"], default="yandex")
    args = parser.parse_args()
    run(args.source)
    logger.info("Этап 3 готов")
