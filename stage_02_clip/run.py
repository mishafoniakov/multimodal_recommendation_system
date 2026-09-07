"""Запуск этапа 2: CLIP-эмбеддинги."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from common.logging_setup import setup_logging
from config.settings import FILE_NAMES, Settings
from stage_02_clip.clip_embedder import ClipEmbedder

logger = logging.getLogger(__name__)


def run(source: str) -> None:
    settings = Settings(source=source)
    embedder = ClipEmbedder(settings.clip_model_id, batch_size=settings.batch_size)
    out_path = settings.output(FILE_NAMES[source]["clip"])
    embedder.embed_folder(settings.images_dir(), out_path)


if __name__ == "__main__":
    setup_logging()
    parser = argparse.ArgumentParser(description="Этап 2. CLIP-векторы")
    parser.add_argument("--source", choices=["yandex", "flickr"], default="yandex")
    args = parser.parse_args()
    run(args.source)
    logger.info("Этап 2 готов")
