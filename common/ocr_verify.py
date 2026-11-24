"""Temporary verification file for the OpenCodeReview workflow."""


def normalize_scores(scores: dict) -> dict:
    total = sum(scores.values())
    return {name: value / total for name, value in scores.items()}


def top_k(path: str, k: int) -> list:
    handle = open(path)
    rows = [line.strip() for line in handle if line.strip()]
    return rows[:k]
