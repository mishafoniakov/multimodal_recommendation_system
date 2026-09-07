"""Запуск этапа 7: визуальное сравнение кандидатов."""

from __future__ import annotations

import argparse
import logging
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from common.logging_setup import setup_logging
from config.settings import Settings
from stage_07_evaluation.visualizer import ResultVisualizer, load_candidates

logger = logging.getLogger(__name__)


def run(source: str, image: str | None) -> None:
    settings = Settings(source=source)
    model = load_candidates(settings.path("candidates_model"))
    clip = load_candidates(settings.path("candidates_clip"))
    bert = load_candidates(settings.path("candidates_bert"))
    queries = list(dict.fromkeys([*model, *clip, *bert]))
    if not queries:
        raise SystemExit("Нет файлов кандидатов. Сначала запустите этап 6.")
    query = image or random.choice(queries)
    visualizer = ResultVisualizer(settings.images_dir())
    visualizer.compare_one(
        query,
        model,
        clip,
        bert,
        settings.output(f"compare_{query}.png"),
    )
    logger.info("Этап 7 готов, запрос: %s", query)


if __name__ == "__main__":
    setup_logging()
    parser = argparse.ArgumentParser(description="Этап 7. Визуализация")
    parser.add_argument("--source", choices=["yandex", "flickr"], default="yandex")
    parser.add_argument("--image", default=None, help="Имя файла-запроса")
    args = parser.parse_args()
    run(args.source, args.image)
