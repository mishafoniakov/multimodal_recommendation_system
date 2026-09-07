"""Метрики близости двух векторов: косинус и L2."""

from __future__ import annotations

import numpy as np


def cosine_similarity(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    numerator = np.sum(left * right, axis=1)
    denominator = np.linalg.norm(left, axis=1) * np.linalg.norm(right, axis=1)
    return numerator / (denominator + 1e-12)


def l2_distance(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    return np.linalg.norm(left - right, axis=1)
