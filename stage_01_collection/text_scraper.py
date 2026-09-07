"""Этап 1. Тексты к картинкам через поиск Яндекса по картинке.

Прокси задаётся переменной PROXY_URL, логин и пароль в код не пишем.
"""

from __future__ import annotations

import logging
import time
from urllib.parse import quote

import requests
from bs4 import BeautifulSoup

from common.io import JsonlStore

logger = logging.getLogger(__name__)


class YandexImageScraper:
    """Для каждой картинки достаёт подпись и похожие страницы."""

    def __init__(self, proxy_url: str = "", pause_sec: float = 15.0) -> None:
        self.pause_sec = pause_sec
        self.session = requests.Session()
        self.session.headers["User-Agent"] = (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        )
        if proxy_url:
            self.session.proxies = {"http": proxy_url, "https": proxy_url}

    def extract(self, image_url: str) -> tuple[str, list[str]]:
        search_url = (
            "https://yandex.ru/images/search?rpt=imageview&url=" + quote(image_url, safe="")
        )
        response = self.session.get(search_url, timeout=30)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        titles = soup.find_all("div", class_="CbirSites-ItemTitle")
        text = titles[0].get_text(strip=True) if titles else ""
        urls = []
        for block in titles[:10]:
            link = block.find("a")
            if link and link.get("href"):
                urls.append(link["href"])
        time.sleep(self.pause_sec)
        return text, urls

    def scrape_urls(
        self,
        image_to_url: dict[str, str],
        out_json,
        start: int = 0,
        end: int | None = None,
    ) -> None:
        items = list(image_to_url.items())
        end = len(items) if end is None else end
        for index in range(start, end):
            image_name, image_url = items[index]
            try:
                text, urls = self.extract(image_url)
            except Exception:
                logger.warning("%s не удалось скачать текст", index)
                continue
            JsonlStore.append(
                out_json,
                {
                    "img": image_name,
                    "img_text": text,
                    "url": image_url,
                    "urls": urls,
                },
            )
            logger.info("%s готов", index)
