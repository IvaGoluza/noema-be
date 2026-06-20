from typing import Literal

from pydantic import BaseModel, field_validator


class PredictRequest(BaseModel):
    language: Literal["en", "hr"]
    situation: str
    thought: str

    @field_validator("situation", "thought")
    @classmethod
    def must_not_be_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("must not be empty")
        return v


class BinaryResult(BaseModel):
    label: str
    prob_distorted: float


class ClassificationResult(BaseModel):
    category: str
    confidence: float


class LatencyMs(BaseModel):
    translate_in: float
    binary: float
    classification: float
    reframe: float
    total: float


class PredictResponse(BaseModel):
    language: str
    thought_used_for_classification: str
    binary: BinaryResult
    classification: ClassificationResult | None
    reframe: str | None
    latency_ms: LatencyMs
