"""Этап 3. BERT-векторы текстов к картинкам."""

from __future__ import annotations

import logging

import torch
from tqdm import tqdm
from transformers import BertModel, BertTokenizer

from common.io import JsonlStore

logger = logging.getLogger(__name__)


class BertEmbedder:
    """Берёт [CLS]-вектор предложения — как в исходных ноутбуках."""

    def __init__(self, model_id: str, max_length: int = 50, batch_size: int = 32) -> None:
        self.max_length = max_length
        self.batch_size = batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = BertTokenizer.from_pretrained(model_id)
        self.model = BertModel.from_pretrained(model_id, output_hidden_states=True)
        self.model.to(self.device)
        self.model.eval()

    def embed(self, image_to_text: dict[str, str], out_json) -> None:
        items = list(image_to_text.items())
        logger.info("BERT: %s текстов, устройство %s", len(items), self.device)
        rows = []
        for start in tqdm(range(0, len(items), self.batch_size), desc="BERT"):
            batch = items[start : start + self.batch_size]
            names = [name for name, _ in batch]
            texts = [text for _, text in batch]
            encodings = self.tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
                add_special_tokens=True,
            )
            encodings = {key: value.to(self.device) for key, value in encodings.items()}
            with torch.no_grad():
                output = self.model(**encodings)
            cls_vectors = output.last_hidden_state[:, 0, :].cpu().tolist()
            for name, vector in zip(names, cls_vectors):
                rows.append({"img": name, "bert": vector})
        JsonlStore.write(out_json, rows)
        logger.info("BERT записан в %s", out_json)
