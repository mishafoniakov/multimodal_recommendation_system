"""Пути и параметры дипломного пайплайна.

Все тяжёлые данные лежат локально. Для GitHub достаточно .env.example.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")

# Имена файлов как в исходных ноутбуках: Yandex без суффикса, Flickr — с _flickr.
FILE_NAMES = {
    "yandex": {
        "clip": "clip.json",
        "bert": "bert.json",
        "clip_bert": "clip_bert.json",
        "pairs": "paired_images_texts_clusters.csv",
        "learning": "learning_pipeline.csv",
        "model": "catboost_model.bin",
        "candidates_model": "image_model_candidates_5.json",
        "candidates_clip": "image_candidates_5_clip_cossim.json",
        "candidates_bert": "image_candidates_5_bert_cossim.json",
    },
    "flickr": {
        "clip": "clip_flickr.json",
        "bert": "bert_flickr.json",
        "clip_bert": "clip_bert_flickr.json",
        "pairs": "nearest_clusters_flickr.csv",
        "learning": "learning_pipeline.csv",
        "model": "catboost_model.bin",
        "candidates_model": "flickr_model_candidates_5.json",
        "candidates_clip": "flickr_candidates_5_clip_cossim.json",
        "candidates_bert": "flickr_candidates_5_bert_cossim.json",
    },
}


class Settings:
    """Настройки одного запуска: источник данных и гиперпараметры."""

    def __init__(self, source: str = "yandex") -> None:
        if source not in FILE_NAMES:
            raise ValueError(f"Неизвестный источник: {source}. Нужен yandex или flickr.")

        self.source = source
        self.root = PROJECT_ROOT
        self.data_dir = Path(os.getenv("THESIS_DATA_DIR", PROJECT_ROOT)).expanduser()
        self.artifacts_dir = self.data_dir / "artifacts" / source
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)

        self.clip_model_id = os.getenv("CLIP_MODEL_ID", "openai/clip-vit-base-patch32")
        self.bert_model_id = os.getenv("BERT_MODEL_ID", "bert-base-multilingual-cased")
        self.bert_max_length = 50
        self.batch_size = int(os.getenv("EMBED_BATCH_SIZE", "32"))

        self.n_clusters = int(os.getenv("CLUSTERS_NUMBER", "1000"))
        self.n_nearest_clusters = int(os.getenv("NEAREST_CLUSTERS_NUMBER", "3"))
        self.n_candidates = int(os.getenv("CANDIDATES_PER_IMAGE", "5"))

        self.catboost_iterations = 2000
        self.catboost_learning_rate = 0.001
        self.catboost_depth = 6
        self.scale_pos_weight = 2.64
        self.random_seed = 13
        self.proxy_url = os.getenv("PROXY_URL", "")

    def images_dir(self) -> Path:
        env_name = "FLICKR_IMAGES_DIR" if self.source == "flickr" else "YANDEX_IMAGES_DIR"
        if os.getenv(env_name):
            return Path(os.getenv(env_name)).expanduser()
        if self.source == "flickr":
            return self.data_dir / "flickr30k_images"
        return self.data_dir / "Dataset"

    def captions_csv(self) -> Path:
        if os.getenv("FLICKR_CAPTIONS_CSV"):
            return Path(os.getenv("FLICKR_CAPTIONS_CSV")).expanduser()
        return self.data_dir / "text_data.csv"

    def yandex_text_json(self) -> Path:
        return self._find(
            "image_big_text_dataset.json",
            extra_dirs=[
                self.data_dir / "Dataset_representation" / "Dataset_texts",
                self.root / "Dataset_representation" / "Dataset_texts",
            ],
        )

    def path(self, key: str) -> Path:
        """Путь к файлу этапа: сначала artifacts, потом старые папки ноутбуков."""
        name = FILE_NAMES[self.source][key]
        return self._find(
            name,
            extra_dirs=[
                self.data_dir / "Dataset_preparation",
                self.data_dir / "Dataset_model",
                self.root / "Dataset_preparation",
                self.root / "Dataset_model",
            ],
        )

    def output(self, name: str) -> Path:
        """Куда писать новый результат текущего запуска."""
        path = self.artifacts_dir / name
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def _find(self, name: str, extra_dirs: list[Path]) -> Path:
        candidates = [self.artifacts_dir / name, *[directory / name for directory in extra_dirs]]
        for path in candidates:
            if path.exists():
                return path
        return self.artifacts_dir / name
