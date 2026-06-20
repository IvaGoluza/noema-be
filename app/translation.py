import os
from functools import lru_cache

import httpx

GOOGLE_TRANSLATE_URL = "https://translation.googleapis.com/language/translate/v2"


def translate_batch(
    texts: list[str],
    source: str = "hr",
    target: str = "en",
    api_key: str | None = None,
    timeout: float = 5.0,
) -> list[str]:
    """Translate 1–N strings in a single Google Translate v2 call."""
    if api_key is None:
        api_key = os.environ["GOOGLE_TRANSLATE_API_KEY"]

    response = httpx.post(
        GOOGLE_TRANSLATE_URL,
        params={"key": api_key},
        data={"q": texts, "source": source, "target": target, "format": "text"},
        timeout=timeout,
    )
    response.raise_for_status()
    return [t["translatedText"] for t in response.json()["data"]["translations"]]


@lru_cache(maxsize=512)
def _translate_cached(text: str, source: str, target: str) -> str:
    """Single-string cached wrapper. Cache key includes direction."""
    return translate_batch([text], source=source, target=target)[0]


def translate_situation_and_thought(
    situation: str, thought: str
) -> tuple[str, str]:
    """Translate situation and thought from HR to EN, with per-string LRU caching."""
    s_en = _translate_cached(situation, "hr", "en")
    t_en = _translate_cached(thought, "hr", "en")
    return s_en, t_en
