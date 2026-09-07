"""Запуск этапа 6: кандидаты модели, CLIP и BERT."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from common.logging_setup import setup_logging
from config.settings import FILE_NAMES, Settings
from stage_04_clustering.clusterer import ClusterPipeline
from stage_06_candidates.ranker import CandidateRanker

logger = logging.getLogger(__name__)


def load_pairs(settings: Settings) -> pd.DataFrame:
    metrics_path = settings.output("pairs_with_metrics.csv")
    if metrics_path.exists():
        return pd.read_csv(metrics_path)
    pairs = pd.read_csv(settings.path("pairs"))
    return ClusterPipeline.add_similarity(pairs, settings.path("clip_bert"))


def run(source: str) -> None:
    settings = Settings(source=source)
    pairs = load_pairs(settings)
    ranker = CandidateRanker(n_candidates=settings.n_candidates)
    ids, features = ranker.prepare_features(pairs)
    names = FILE_NAMES[source]

    model_path = settings.path("model")
    if model_path.exists():
        scored = ranker.add_model_scores(ids, features, model_path)
        ranker.top(scored, "pred_1", settings.output(names["candidates_model"]))
    else:
        logger.warning("Модель %s не найдена, кандидаты модели пропущены", model_path)
        scored = ids

    ranker.top(scored, "clip_cossim", settings.output(names["candidates_clip"]))
    ranker.top(scored, "bert_cossim", settings.output(names["candidates_bert"]))
    logger.info("Этап 6 готов")


if __name__ == "__main__":
    setup_logging()
    parser = argparse.ArgumentParser(description="Этап 6. Кандидаты")
    parser.add_argument("--source", choices=["yandex", "flickr"], default="yandex")
    args = parser.parse_args()
    run(args.source)
