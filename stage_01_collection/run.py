"""
О документе
-----------
Источник: TSV related images (Яндекс) или CSV подписей Flickr.
Цель: собрать карточку `img` + `txt` без сырых картинок в git.
Результат: `dataset.json`, `image_big_text_dataset.json`.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from common.logging_setup import setup_logging
from config.settings import Settings
from stage_01_collection.image_collector import ImageCollector
from stage_01_collection.text_aggregator import TextAggregator

logger = logging.getLogger(__name__)


def run_images(settings: Settings, tsv_path: Path, start: int, end: int | None) -> None:
    collector = ImageCollector(
        images_dir=settings.images_dir(),
        url_index_path=settings.output("dataset_dict.txt"),
    )
    pairs = ImageCollector.read_pairs(tsv_path)
    collector.collect(pairs, settings.output("dataset.json"), start=start, end=end)


def run_texts(settings: Settings) -> None:
    if settings.source == "flickr":
        texts = TextAggregator.from_flickr_csv(settings.captions_csv())
    else:
        result_dir = settings.data_dir / "Dataset_representation" / "Dataset_texts"
        result_files = sorted(result_dir.glob("result_*.json"))
        if not result_files:
            logger.warning("Файлы result_*.json не найдены, тексты не собраны")
            return
        texts = TextAggregator.from_yandex_results(result_files)
    TextAggregator.save(texts, settings.output("image_big_text_dataset.json"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Этап 1. Сбор картинок и текстов")
    parser.add_argument("--source", choices=["yandex", "flickr"], default="yandex")
    parser.add_argument("--part", choices=["images", "texts", "all"], default="texts")
    parser.add_argument("--tsv", type=Path, default=None, help="related_images.tsv")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    setup_logging()
    args = parse_args()
    settings = Settings(source=args.source)
    if args.part in {"images", "all"}:
        if args.tsv is None:
            raise SystemExit("Для скачивания картинок нужен --tsv")
        run_images(settings, args.tsv, args.start, args.end)
    if args.part in {"texts", "all"}:
        run_texts(settings)
    logger.info("Этап 1 готов")
