"""Этап 4. Кластеры по CLIP+BERT и пары картинка–кандидат."""

from __future__ import annotations

import logging
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans

from common.io import JsonlStore
from common.metrics import cosine_similarity, l2_distance

logger = logging.getLogger(__name__)


class ClusterPipeline:
    """KMeans по склеенному вектору, затем ближайшие кластеры → пары изображений."""

    def __init__(self, n_clusters: int = 1000, n_nearest: int = 3, random_seed: int = 13) -> None:
        self.n_clusters = n_clusters
        self.n_nearest = n_nearest
        self.random_seed = random_seed

    def fit_pairs(self, clip_bert_json, pairs_csv, nearest_json) -> pd.DataFrame:
        rows = JsonlStore.read(clip_bert_json)
        names = [row["img"] for row in rows]
        clip = np.asarray([row["clip"] for row in rows], dtype=float)
        bert = np.asarray([row["bert"] for row in rows], dtype=float)
        features = np.hstack([clip, bert])

        n_clusters = min(self.n_clusters, len(names))
        logger.info("KMeans: %s картинок, %s кластеров", len(names), n_clusters)
        model = MiniBatchKMeans(
            n_clusters=n_clusters,
            random_state=self.random_seed,
            batch_size=2048,
            n_init="auto",
        )
        labels = model.fit_predict(features)
        distances = model.transform(features)

        cluster_to_images = defaultdict(list)
        for name, label in zip(names, labels):
            cluster_to_images[int(label)].append(name)

        nearest_rows = []
        pair_rows = []
        for i, name in enumerate(names):
            nearest_ids = np.argsort(distances[i])[: self.n_nearest].tolist()
            nearest_rows.append({"img": name, "clusters": nearest_ids})
            candidates = []
            for cluster_id in nearest_ids:
                candidates.extend(cluster_to_images[int(cluster_id)])
            for candidate in set(candidates):
                if candidate != name:
                    pair_rows.append({"image": name, "candidate": candidate})

        JsonlStore.write(nearest_json, nearest_rows)
        pairs = pd.DataFrame(pair_rows)
        pairs.to_csv(pairs_csv, index=False)
        logger.info("Пар из кластеров: %s", len(pairs))
        return pairs

    @staticmethod
    def add_similarity(pairs: pd.DataFrame, clip_bert_json) -> pd.DataFrame:
        """Для каждой пары считает CLIP/BERT cosine и L2."""
        rows = JsonlStore.read(clip_bert_json)
        clip_map = {row["img"]: row["clip"] for row in rows}
        bert_map = {row["img"]: row["bert"] for row in rows}
        pairs = pairs.copy()
        pairs["img_clip_1"] = pairs["image"].map(clip_map)
        pairs["img_clip_2"] = pairs["candidate"].map(clip_map)
        pairs["img_bert_1"] = pairs["image"].map(bert_map)
        pairs["img_bert_2"] = pairs["candidate"].map(bert_map)
        pairs = pairs.dropna(subset=["img_clip_1", "img_clip_2", "img_bert_1", "img_bert_2"])

        clip_1 = np.vstack(pairs["img_clip_1"].to_list())
        clip_2 = np.vstack(pairs["img_clip_2"].to_list())
        bert_1 = np.vstack(pairs["img_bert_1"].to_list())
        bert_2 = np.vstack(pairs["img_bert_2"].to_list())
        pairs["clip_cossim"] = cosine_similarity(clip_1, clip_2)
        pairs["clip_l2"] = l2_distance(clip_1, clip_2)
        pairs["bert_cossim"] = cosine_similarity(bert_1, bert_2)
        pairs["bert_l2"] = l2_distance(bert_1, bert_2)
        return pairs
