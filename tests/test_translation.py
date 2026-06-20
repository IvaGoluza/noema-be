"""Tests for the Google Translate v2 wrapper."""

import pytest
import respx
import httpx

from app.translation import (
    _translate_cached,
    translate_batch,
    translate_situation_and_thought,
)

TRANSLATE_URL = "https://translation.googleapis.com/language/translate/v2"

MOCK_RESPONSE = {
    "data": {
        "translations": [
            {"translatedText": "He's probably thinking about leaving me."}
        ]
    }
}


@pytest.fixture(autouse=True)
def clear_cache() -> None:
    _translate_cached.cache_clear()


class TestTranslateBatch:
    @respx.mock
    def test_extracts_translated_text(self) -> None:
        respx.post(TRANSLATE_URL).mock(
            return_value=httpx.Response(200, json=MOCK_RESPONSE)
        )
        result = translate_batch(
            ["Sigurno razmišlja o tome da me ostavi."], api_key="test-key"
        )
        assert result == ["He's probably thinking about leaving me."]

    @respx.mock
    def test_raises_on_http_error(self) -> None:
        respx.post(TRANSLATE_URL).mock(return_value=httpx.Response(403))
        with pytest.raises(httpx.HTTPStatusError):
            translate_batch(["some text"], api_key="test-key")

    @respx.mock
    def test_passes_timeout(self) -> None:
        respx.post(TRANSLATE_URL).mock(
            return_value=httpx.Response(200, json=MOCK_RESPONSE)
        )
        result = translate_batch(["text"], api_key="k", timeout=10.0)
        assert len(result) == 1


class TestTranslateCached:
    @respx.mock
    def test_cache_hit_skips_second_http_call(self) -> None:
        route = respx.post(TRANSLATE_URL).mock(
            return_value=httpx.Response(200, json=MOCK_RESPONSE)
        )
        import os
        os.environ.setdefault("GOOGLE_TRANSLATE_API_KEY", "test-key")

        first = _translate_cached("Sigurno razmišlja o tome da me ostavi.", "hr", "en")
        second = _translate_cached("Sigurno razmišlja o tome da me ostavi.", "hr", "en")

        assert first == second == "He's probably thinking about leaving me."
        assert route.call_count == 1


class TestTranslateSituationAndThought:
    @respx.mock
    def test_returns_both_translated_strings(self) -> None:
        import os
        os.environ.setdefault("GOOGLE_TRANSLATE_API_KEY", "test-key")

        sit_response = {
            "data": {"translations": [{"translatedText": "I had an argument with my partner."}]}
        }
        thought_response = {
            "data": {
                "translations": [{"translatedText": "He's definitely going to leave me."}]
            }
        }

        respx.post(TRANSLATE_URL).mock(
            side_effect=[
                httpx.Response(200, json=sit_response),
                httpx.Response(200, json=thought_response),
            ]
        )

        sit_en, thought_en = translate_situation_and_thought(
            "Imala sam svađu s partnerom.",
            "Sigurno će me ostaviti.",
        )
        assert sit_en == "I had an argument with my partner."
        assert thought_en == "He's definitely going to leave me."
