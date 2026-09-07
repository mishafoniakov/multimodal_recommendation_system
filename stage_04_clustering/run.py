"""
О документе
-----------
Источник: `clip_bert.json` (CLIP∥BERT, длина 1280).
Цель: сузить кандидатов через MiniBatchKMeans, без полного перебора пар.
Результат: `nearest_clusters.json`, `pairs_with_metrics.csv`.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from common.logging_setup import setup_logging
from config.settings import FILE_NAMES, Settings
from stage_04_clustering.clusterer import ClusterPipeline

logger = logging.getLogger(__name__)


def run(source: str) -> None:
    settings = Settings(source=source)
    pipeline = ClusterPipeline(
        n_clusters=settings.n_clusters,
        n_nearest=settings.n_nearest_clusters,
        random_seed=settings.random_seed,
    )
    pairs = pipeline.fit_pairs(
        clip_bert_json=settings.path("clip_bert"),
        pairs_csv=settings.output(FILE_NAMES[source]["pairs"]),
        nearest_json=settings.output("nearest_clusters.json"),
    )
    with_metrics = pipeline.add_similarity(pairs, settings.path("clip_bert"))
    with_metrics.to_csv(settings.output("pairs_with_metrics.csv"), index=False)
    logger.info("Этап 4 готов, пар: %s", len(with_metrics))


if __name__ == "__main__":
    setup_logging()
    parser = argparse.ArgumentParser(description="Этап 4. Кластеризация")
    parser.add_argument("--source", choices=["yandex", "flickr"], default="yandex")
    args = parser.parse_args()
    run(args.source)
