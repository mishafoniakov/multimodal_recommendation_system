"""Этап 1. Сбор пар связанных картинок по TSV Яндекса."""

from __future__ import annotations

import csv
import logging
import re
from io import BytesIO
from pathlib import Path

import requests
from PIL import Image

from common.io import JsonlStore

logger = logging.getLogger(__name__)


class ImageCollector:
    """Скачивает пары картинок из related_images.tsv и пишет dataset.json."""

    def __init__(self, images_dir: Path, url_index_path: Path) -> None:
        self.images_dir = images_dir
        self.url_index_path = url_index_path
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.url_to_file = self._load_url_index()

    def _load_url_index(self) -> dict[str, str]:
        if not self.url_index_path.exists():
            return {}
        index = {}
        for line in self.url_index_path.read_text(encoding="utf-8").splitlines():
            parts = line.split(" ", 1)
            if len(parts) == 2:
                file_name, url = parts
                index[url.strip()] = file_name.strip()
        return index

    def _remember(self, file_name: str, url: str) -> None:
        self.url_to_file[url] = file_name
        with self.url_index_path.open("a", encoding="utf-8") as file:
            file.write(f"{file_name} {url}\n")

    def _download(self, url: str, file_name: str) -> str:
        if url in self.url_to_file:
            return self.url_to_file[url]
        response = requests.get(url, timeout=(15, 30))
        response.raise_for_status()
        image = Image.open(BytesIO(response.content))
        image.save(self.images_dir / file_name)
        self._remember(file_name, url)
        return file_name

    @staticmethod
    def read_pairs(tsv_path: Path) -> list[tuple[str, str, str]]:
        """Каждая строка TSV: target и два URL связанных картинок."""
        pattern = re.compile(r"\[.*?\]")
        pairs = []
        with tsv_path.open("r", encoding="utf-8") as file:
            reader = csv.reader(file, delimiter="\t")
            next(reader, None)
            for row in reader:
                urls = re.findall(pattern, row[1])
                if len(urls) >= 2:
                    pairs.append((row[0], urls[0][1:-1], urls[1][1:-1]))
        return pairs

    def collect(
        self,
        pairs: list[tuple[str, str, str]],
        out_json: Path,
        start: int = 0,
        end: int | None = None,
    ) -> None:
        end = len(pairs) if end is None else end
        logger.info("Скачиваю пары %s–%s из %s", start, end, len(pairs))
        for index in range(start, end):
            target, url_1, url_2 = pairs[index]
            try:
                file_1 = self._download(url_1, f"img_{index}_0.jpg")
                file_2 = self._download(url_2, f"img_{index}_1.jpg")
            except Exception:
                logger.warning("Пара %s не скачалась", index)
                continue
            JsonlStore.append(
                out_json,
                {
                    "file_img1": file_1,
                    "file_img2": file_2,
                    "target": target,
                    "url_img_1": url_1,
                    "url_img_2": url_2,
                },
            )
