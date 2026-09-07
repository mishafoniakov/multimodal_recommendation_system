"""Точка входа: python run.py --stage clip --source yandex"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from common.logging_setup import setup_logging

logger = logging.getLogger(__name__)

STAGES = ["collection", "clip", "bert", "cluster", "train", "candidates", "eval"]


def run_stage(stage: str, source: str, model: str) -> None:
    if stage == "collection":
        from stage_01_collection.run import run_texts

        from config.settings import Settings

        run_texts(Settings(source=source))
    elif stage == "clip":
        from stage_02_clip.run import run

        run(source)
    elif stage == "bert":
        from stage_03_bert.run import run

        run(source)
    elif stage == "cluster":
        from stage_04_clustering.run import run

        run(source)
    elif stage == "train":
        from stage_05_training.run import run

        run(source, model)
    elif stage == "candidates":
        from stage_06_candidates.run import run

        run(source)
    elif stage == "eval":
        from stage_07_evaluation.run import run

        run(source, image=None)
    else:
        raise ValueError(stage)


def main() -> None:
    parser = argparse.ArgumentParser(description="Дипломный пайплайн: картинка → похожие картинки")
    parser.add_argument("--stage", choices=STAGES + ["all"], default="all")
    parser.add_argument("--source", choices=["yandex", "flickr"], default="yandex")
    parser.add_argument("--model", default="catboost", help="Модель этапа train")
    args = parser.parse_args()

    stages = STAGES if args.stage == "all" else [args.stage]
    for stage in stages:
        logger.info("=== этап %s / %s ===", stage, args.source)
        run_stage(stage, args.source, args.model)
    logger.info("Готово")


if __name__ == "__main__":
    setup_logging()
    main()
