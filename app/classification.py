import time
from typing import Any

import numpy as np
import torch
from torch.nn.functional import softmax
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from app.config import Settings

MAX_LEN = 512

# Module-level state set once by load_models() during lifespan startup.
device: torch.device
tok_b: Any
mdl_b: Any
tok_t: Any
mdl_t: Any
TYPE_LABELS: list[str]


def load_models(settings: Settings) -> None:
    """Load both checkpoints into module globals. Called once from lifespan."""
    global device, tok_b, mdl_b, tok_t, mdl_t, TYPE_LABELS

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    tok_b = AutoTokenizer.from_pretrained(settings.model_binary_path)
    mdl_b = AutoModelForSequenceClassification.from_pretrained(settings.model_binary_path)
    mdl_b.to(device).eval()
    cfg_b = [mdl_b.config.id2label[i] for i in sorted(mdl_b.config.id2label, key=int)]
    assert cfg_b == ["No Distortion", "Distorted"], f"Unexpected binary labels: {cfg_b}"

    tok_t = AutoTokenizer.from_pretrained(settings.model_type_path)
    mdl_t = AutoModelForSequenceClassification.from_pretrained(settings.model_type_path)
    mdl_t.to(device).eval()
    TYPE_LABELS = [mdl_t.config.id2label[i] for i in sorted(mdl_t.config.id2label, key=int)]
    assert len(TYPE_LABELS) == 10, f"Expected 10 distortion classes, found {len(TYPE_LABELS)}"


def predict_one(text: str, tokenizer: Any, model: Any) -> np.ndarray:
    """Return the full softmax probability distribution for a single input."""
    enc = tokenizer(
        text, truncation=True, max_length=MAX_LEN, return_tensors="pt"
    ).to(device)
    with torch.no_grad():
        logits = model(**enc).logits
    return softmax(logits, dim=-1).cpu().numpy()[0]


def classify(text: str) -> dict[str, Any]:
    """Run the binary + (optional) type pipeline. Returns timing alongside predictions."""
    t0 = time.perf_counter()
    pb = predict_one(text, tok_b, mdl_b)
    binary_ms = (time.perf_counter() - t0) * 1000
    prob_distorted = float(pb[1])  # index 1 is Distorted

    if prob_distorted < 0.5:
        return {
            "label": "No Distortion",
            "prob_distorted": prob_distorted,
            "category": None,
            "category_confidence": None,
            "binary_ms": binary_ms,
            "type_ms": 0.0,
        }

    t0 = time.perf_counter()
    pt = predict_one(text, tok_t, mdl_t)
    type_ms = (time.perf_counter() - t0) * 1000
    type_idx = int(np.argmax(pt))
    return {
        "label": "Distorted",
        "prob_distorted": prob_distorted,
        "category": TYPE_LABELS[type_idx],
        "category_confidence": float(pt[type_idx]),
        "binary_ms": binary_ms,
        "type_ms": type_ms,
    }
