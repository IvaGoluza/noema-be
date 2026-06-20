import time
from contextlib import asynccontextmanager
from typing import AsyncGenerator
from uuid import uuid4

import anthropic
import structlog
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware

from app import classification, exemplars, reframing, translation
from app.config import get_settings
from app.logging_setup import configure_logging
from app.schemas import (
    BinaryResult,
    ClassificationResult,
    LatencyMs,
    PredictRequest,
    PredictResponse,
)

load_dotenv()
settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    configure_logging(settings.log_level)
    log = structlog.get_logger()

    log.info("startup", status="loading models")
    try:
        classification.load_models(settings)
        exemplars.validate_exemplars()
    except Exception as exc:
        log.error("startup_failed", error=str(exc))
        raise RuntimeError(f"Startup aborted: {exc}") from exc

    app.state.anthropic_client = anthropic.Anthropic(
        api_key=settings.anthropic_api_key,
        timeout=settings.anthropic_timeout_s,
    )
    log.info("startup", status="ready")
    yield


app = FastAPI(title="Noema Backend", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_allowed_origins,
    allow_methods=["POST", "GET"],
    allow_headers=["Content-Type"],
)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
async def predict(body: PredictRequest, req: Request) -> PredictResponse:
    req_id = uuid4().hex[:12]
    log = structlog.get_logger().bind(request_id=req_id, language=body.language)
    client = req.app.state.anthropic_client

    t_total_start = time.perf_counter()

    # Stage 1: Translation (HR → EN only; skipped for EN)
    if body.language == "hr":
        try:
            sit_en, thought_en = translation.translate_situation_and_thought(
                body.situation, body.thought
            )
        except Exception as exc:
            log.warning("translation_failed", error=str(exc))
            raise HTTPException(status_code=503, detail="Translation service unavailable.")
        t_translate_ms = (time.perf_counter() - t_total_start) * 1000
    else:
        sit_en = body.situation
        thought_en = body.thought
        t_translate_ms = 0.0

    # Stages 2 & 3: Binary detection + (optional) type classification
    result = classification.classify(thought_en)
    t_binary_ms = result["binary_ms"]
    t_class_ms = result["type_ms"]

    if settings.log_user_text:
        log.debug(
            "predict_input",
            situation=body.situation,
            thought=body.thought,
            thought_en=thought_en,
        )

    log.info(
        "predict",
        binary=result["label"],
        category=result.get("category"),
        translate_ms=round(t_translate_ms),
        binary_ms=round(t_binary_ms),
        class_ms=round(t_class_ms),
    )

    if result["label"] == "No Distortion":
        t_total_ms = (time.perf_counter() - t_total_start) * 1000
        return PredictResponse(
            language=body.language,
            thought_used_for_classification=thought_en,
            binary=BinaryResult(
                label="No Distortion",
                prob_distorted=result["prob_distorted"],
            ),
            classification=None,
            reframe=None,
            latency_ms=LatencyMs(
                translate_in=round(t_translate_ms, 1),
                binary=round(t_binary_ms, 1),
                classification=0.0,
                reframe=0.0,
                total=round(t_total_ms, 1),
            ),
        )

    # Stage 4: Exemplar lookup
    examples = exemplars.get_exemplars(result["category"])

    # Stage 5: Reframe generation — original (untranslated) text passed to LLM
    prompt = reframing.build_prompt_e4(
        thought=body.thought,
        situation=body.situation,
        category=result["category"],
        examples=examples,
        respond_in_croatian=(body.language == "hr"),
    )
    try:
        reframe_result = reframing.call_anthropic(prompt, client, settings)
    except Exception as exc:
        log.warning("reframing_failed", error=str(exc))
        raise HTTPException(status_code=503, detail="Reframing service unavailable.")

    if settings.log_user_text:
        log.debug("predict_output", reframe=reframe_result["text"])

    t_total_ms = (time.perf_counter() - t_total_start) * 1000

    return PredictResponse(
        language=body.language,
        thought_used_for_classification=thought_en,
        binary=BinaryResult(
            label="Distorted",
            prob_distorted=result["prob_distorted"],
        ),
        classification=ClassificationResult(
            category=result["category"],
            confidence=result["category_confidence"],
        ),
        reframe=reframe_result["text"],
        latency_ms=LatencyMs(
            translate_in=round(t_translate_ms, 1),
            binary=round(t_binary_ms, 1),
            classification=round(t_class_ms, 1),
            reframe=round(reframe_result["latency_ms"], 1),
            total=round(t_total_ms, 1),
        ),
    )
