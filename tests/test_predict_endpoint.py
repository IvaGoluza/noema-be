"""Endpoint contract tests for POST /predict.

All classifier inference, Anthropic API calls, and Google Translate calls are
mocked so these tests run without model checkpoints or live API keys.
"""

from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

# --- Shared mock return values ---

BINARY_DISTORTED = {
    "label": "Distorted",
    "prob_distorted": 0.91,
    "category": "Jumping to conclusions: mind reading",
    "category_confidence": 0.78,
    "binary_ms": 30.0,
    "type_ms": 35.0,
}

BINARY_CLEAN = {
    "label": "No Distortion",
    "prob_distorted": 0.07,
    "category": None,
    "category_confidence": None,
    "binary_ms": 28.0,
    "type_ms": 0.0,
}

REFRAME_RESULT: dict[str, Any] = {
    "text": "He may be quiet for many reasons; I'll ask him later.",
    "input_tokens": 300,
    "output_tokens": 20,
    "latency_ms": 1500.0,
}


@pytest_asyncio.fixture
async def client() -> AsyncClient:  # type: ignore[override]
    mock_anthropic = MagicMock()

    # ASGITransport does not send the ASGI lifespan scope, so the lifespan
    # context manager in main.py never runs during tests. We patch the
    # model-loading calls defensively and set app.state directly so the
    # endpoint can find the Anthropic client.
    with (
        patch("app.classification.load_models", return_value=None),
        patch("app.exemplars.validate_exemplars", return_value=None),
    ):
        from app.main import app

        app.state.anthropic_client = mock_anthropic

        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as c:
            yield c


class TestPredictEnglishDistorted:
    async def test_shape_and_fields(self, client: AsyncClient) -> None:
        with patch(
            "app.classification.classify", return_value=BINARY_DISTORTED
        ), patch("app.reframing.call_anthropic", return_value=REFRAME_RESULT):
            resp = await client.post(
                "/predict",
                json={
                    "language": "en",
                    "situation": "My colleague was quiet in the meeting.",
                    "thought": "He must be angry with me.",
                },
            )

        assert resp.status_code == 200
        body = resp.json()
        assert body["language"] == "en"
        assert body["thought_used_for_classification"] == "He must be angry with me."
        assert body["binary"]["label"] == "Distorted"
        assert body["binary"]["prob_distorted"] == pytest.approx(0.91)
        assert body["classification"]["category"] == "Jumping to conclusions: mind reading"
        assert body["classification"]["confidence"] == pytest.approx(0.78)
        assert body["reframe"] == REFRAME_RESULT["text"]
        assert set(body["latency_ms"].keys()) == {
            "translate_in",
            "binary",
            "classification",
            "reframe",
            "total",
        }
        assert body["latency_ms"]["translate_in"] == 0.0


class TestPredictEnglishNonDistorted:
    async def test_nulls_when_no_distortion(self, client: AsyncClient) -> None:
        with patch(
            "app.classification.classify", return_value=BINARY_CLEAN
        ):
            resp = await client.post(
                "/predict",
                json={
                    "language": "en",
                    "situation": "Quiet Saturday morning.",
                    "thought": "I should call my mother this weekend.",
                },
            )

        assert resp.status_code == 200
        body = resp.json()
        assert body["binary"]["label"] == "No Distortion"
        assert body["classification"] is None
        assert body["reframe"] is None
        assert body["latency_ms"]["classification"] == 0.0
        assert body["latency_ms"]["reframe"] == 0.0


class TestPredictCroatianDistorted:
    async def test_translation_called_and_original_thought_returned(
        self, client: AsyncClient
    ) -> None:
        hr_thought = "Sigurno razmišlja o tome da me ostavi."
        hr_situation = "Imala sam svađu s partnerom."
        en_thought = "He's probably thinking about leaving me."
        en_situation = "I had an argument with my partner."

        with (
            patch(
                "app.translation.translate_situation_and_thought",
                return_value=(en_situation, en_thought),
            ),
            patch(
                "app.classification.classify",
                return_value=BINARY_DISTORTED,
            ),
            patch(
                "app.reframing.call_anthropic",
                return_value=REFRAME_RESULT,
            ),
        ):
            resp = await client.post(
                "/predict",
                json={
                    "language": "hr",
                    "situation": hr_situation,
                    "thought": hr_thought,
                },
            )

        assert resp.status_code == 200
        body = resp.json()
        assert body["language"] == "hr"
        assert body["thought_used_for_classification"] == en_thought
        assert body["reframe"] == REFRAME_RESULT["text"]
        assert body["latency_ms"]["translate_in"] >= 0


class TestPredictValidation:
    async def test_empty_thought_rejected(self, client: AsyncClient) -> None:
        resp = await client.post(
            "/predict",
            json={"language": "en", "situation": "Some context.", "thought": "  "},
        )
        assert resp.status_code == 422

    async def test_empty_situation_rejected(self, client: AsyncClient) -> None:
        resp = await client.post(
            "/predict",
            json={"language": "en", "situation": "", "thought": "Some thought."},
        )
        assert resp.status_code == 422

    async def test_invalid_language_rejected(self, client: AsyncClient) -> None:
        resp = await client.post(
            "/predict",
            json={"language": "de", "situation": "Situa.", "thought": "Thought."},
        )
        assert resp.status_code == 422


class TestHealth:
    async def test_health_ok(self, client: AsyncClient) -> None:
        resp = await client.get("/health")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}
