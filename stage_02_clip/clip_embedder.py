"""Этап 2. CLIP-векторы картинок."""

from __future__ import annotations

import logging
from pathlib import Path

import torch
from PIL import Image
from transformers import CLIPImageProcessor, CLIPModel
from tqdm import tqdm

from common.io import JsonlStore

logger = logging.getLogger(__name__)


class ClipEmbedder:
    """Считает CLIP-эмбеддинг для каждой картинки в папке."""

    def __init__(self, model_id: str, batch_size: int = 32) -> None:
        self.batch_size = batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = CLIPModel.from_pretrained(model_id).to(self.device)
        self.model.eval()
        self.processor = CLIPImageProcessor.from_pretrained(model_id)

    def list_images(self, images_dir: Path) -> list[str]:
        names = []
        for path in images_dir.iterdir():
            if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}:
                names.append(path.name)
        return sorted(names)

    def embed_folder(self, images_dir: Path, out_json: Path) -> None:
        files = self.list_images(images_dir)
        logger.info("CLIP: %s картинок, устройство %s", len(files), self.device)
        rows = []
        for start in tqdm(range(0, len(files), self.batch_size), desc="CLIP"):
            batch_names = files[start : start + self.batch_size]
            images = []
            valid_names = []
            for name in batch_names:
                try:
                    images.append(Image.open(images_dir / name).convert("RGB"))
                    valid_names.append(name)
                except Exception:
                    logger.warning("Не открылась картинка %s", name)
            if not images:
                continue
            inputs = self.processor(images=images, return_tensors="pt")
            pixel_values = inputs["pixel_values"].to(self.device)
            with torch.no_grad():
                vectors = self.model.get_image_features(pixel_values)
            for name, vector in zip(valid_names, vectors.cpu().tolist()):
                rows.append({"img": name, "clip": vector})
        JsonlStore.write(out_json, rows)
        logger.info("CLIP записан в %s", out_json)
