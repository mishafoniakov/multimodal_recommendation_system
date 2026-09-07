"""Этап 7. Картинки запроса и 5 кандидатов в один ряд."""

from __future__ import annotations

import logging
import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from common.io import JsonlStore

logger = logging.getLogger(__name__)


class ResultVisualizer:
    def __init__(self, images_dir: Path) -> None:
        self.images_dir = images_dir

    def grid(self, image_names: list[str], title: str, out_path: Path) -> None:
        n = len(image_names)
        cols = 6
        rows = (n + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(18, 3 * rows))
        axes = axes.ravel() if n > 1 else [axes]
        for i, ax in enumerate(axes):
            ax.axis("off")
            if i >= n:
                continue
            path = self.images_dir / image_names[i]
            if path.exists():
                ax.imshow(Image.open(path))
            ax.set_title(image_names[i], fontsize=8)
        fig.suptitle(title)
        fig.tight_layout()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=120)
        plt.close(fig)
        logger.info("Сохранён коллаж %s", out_path)

    def compare_one(self, query: str, model: dict, clip: dict, bert: dict, out_path: Path) -> None:
        row = [query]
        row += model.get(query, [])[:5]
        row += clip.get(query, [])[:5]
        row += bert.get(query, [])[:5]
        self.grid(row, f"query={query} | model | CLIP | BERT", out_path)


def load_candidates(path: Path) -> dict:
    if not path.exists():
        return {}
    return JsonlStore.to_dict(path, "img", "candidates")
