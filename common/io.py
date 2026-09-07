"""Чтение и запись JSON Lines — основной формат данных диплома."""

from __future__ import annotations

import json
from pathlib import Path


class JsonlStore:
    """Одна строка файла = один JSON-объект."""

    @staticmethod
    def read(path: Path) -> list[dict]:
        rows = []
        with path.open("r", encoding="utf-8") as file:
            for line in file:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows

    @staticmethod
    def write(path: Path, rows: list[dict]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as file:
            for row in rows:
                file.write(json.dumps(row, ensure_ascii=False) + "\n")

    @staticmethod
    def append(path: Path, row: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")

    @staticmethod
    def to_dict(path: Path, key: str, value: str) -> dict:
        return {row[key]: row[value] for row in JsonlStore.read(path)}
