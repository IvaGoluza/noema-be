# Noema Backend

FastAPI backend for detecting cognitive distortions in automatic thoughts and generating CBT-style reframes.

Supports English and Croatian input. The classifier runs in English; the reframer (Claude) responds directly in the user's language.

---

## How it works

Each request to `/predict` runs through a fixed pipeline:

1. **Translation (Croatian only)**: situation and thought are translated HR > EN via Google Cloud Translation v2.
2. **Binary detection**: a fine-tuned model classifies the English thought as `Distorted` or `No Distortion`. If no distortion is detected, the pipeline stops here.
3. **Type classification**: a fine-tuned model assigns one of 10 cognitive distortion categories.
4. **Exemplar lookup**: three labelled examples are retrieved for the predicted category.
5. **Reframe generation**: an E4 few-shot prompt is sent to the Anthropic API (`claude-sonnet-4-6`). The original (untranslated) text is used; Croatian requests include a one-line instruction to respond in Croatian.

---

## Requirements

- Python 3.11+
- Model checkpoints placed in `models/binary_detector/` and `models/category_classifier/` (not committed to git)
- An Anthropic API key
- A Google Cloud Translation v2 API key (only needed for Croatian input)

---

## Setup

**1. Create and activate a virtual environment**

```bash
python -m venv .venv
source .venv/bin/activate
```

**2. Install dependencies**

```bash
pip install -e ".[dev]"
```

**3. Configure environment variables**

```bash
cp .env.example .env
# Fill in ANTHROPIC_API_KEY and GOOGLE_TRANSLATE_API_KEY
```

**4. Place model checkpoints**

Copy your fine-tuned HuggingFace checkpoints into:

```
models/
├── binary_detector/
│   ├── config.json
│   ├── model.safetensors
│   ├── tokenizer.json
│   └── tokenizer_config.json
└── category_classifier/
    ├── config.json
    ├── model.safetensors
    ├── tokenizer.json
    └── tokenizer_config.json
```

**5. Start the server**

```bash
uvicorn app.main:app --reload
```

Model loading takes 5–15 seconds on CPU. The `/health` endpoint returns 200 once both models are ready.

---

## API

### `GET /health`

Liveness probe. Returns `200 OK` once models are loaded.

### `POST /predict`

Run the full pipeline on a single (situation, thought) pair.

**Request**

```json
{
  "language": "hr",
  "situation": "Imala sam svađu s partnerom prije nego što sam pošla na posao.",
  "thought": "Sigurno razmišlja o tome da me ostavi."
}
```

`language` must be `"en"` or `"hr"`. Both `situation` and `thought` are required non-empty strings.

**Response — distorted thought (Croatian)**

```json
{
  "language": "hr",
  "thought_used_for_classification": "He's probably thinking about leaving me.",
  "binary": {
    "label": "Distorted",
    "prob_distorted": 0.91
  },
  "classification": {
    "category": "Jumping to conclusions: mind reading",
    "confidence": 0.78
  },
  "reframe": "Imala sam svađu s partnerom, ali ne mogu znati što razmišlja dok ne razgovaramo.",
  "latency_ms": {
    "translate_in": 124,
    "binary": 31,
    "classification": 38,
    "reframe": 1842,
    "total": 2035
  }
}
```

**Response — no distortion**

```json
{
  "language": "en",
  "thought_used_for_classification": "I should call my mother this weekend.",
  "binary": { "label": "No Distortion", "prob_distorted": 0.07 },
  "classification": null,
  "reframe": null,
  "latency_ms": { "translate_in": 0, "binary": 28, "classification": 0, "reframe": 0, "total": 28 }
}
```

`classification` and `reframe` are `null` when the binary stage returns `No Distortion`. `thought_used_for_classification` exposes the translated text for debugging; for English requests it equals the input thought.

Upstream failures (Google Translate, Anthropic) return `503`. Internal errors return `500`.

---

## Project structure

```
├── app/
│   ├── main.py             FastAPI app and lifespan (model loading)
│   ├── config.py           pydantic-settings, reads .env
│   ├── schemas.py          request / response models
│   ├── classification.py   binary + type pipeline
│   ├── reframing.py        E4 prompt builder + Anthropic call
│   ├── translation.py      Google Translate v2 wrapper with LRU cache
│   ├── exemplars.py        exemplar loader
│   └── logging_setup.py    structlog configuration
├── data/
│   ├── deployment_exemplars.py     few-shot examples + label map (read-only)
│   └── deployment_eval_examples.py 30 test inputs (read-only)
├── models/                 checkpoints (not committed)
├── tests/
│   ├── conftest.py
│   ├── test_classification.py
│   ├── test_reframing_prompt.py
│   ├── test_translation.py
│   └── test_predict_endpoint.py
├── .env.example
└── pyproject.toml
```

---

## Environment variables

| Variable | Default | Description |
|---|---|---|
| `ANTHROPIC_API_KEY` | — | Required. Anthropic API key. |
| `ANTHROPIC_MODEL` | `claude-sonnet-4-6` | Model used for reframing. |
| `ANTHROPIC_MAX_TOKENS` | `200` | Max tokens in the reframe. |
| `ANTHROPIC_TIMEOUT_S` | `30` | Request timeout in seconds. |
| `ANTHROPIC_MAX_RETRIES` | `3` | Retry attempts on transient errors. |
| `GOOGLE_TRANSLATE_API_KEY` | — | Required for Croatian input. |
| `TRANSLATE_TIMEOUT_S` | `5` | Google Translate timeout in seconds. |
| `MODEL_BINARY_PATH` | `./models/binary_detector` | Path to binary detector checkpoint. |
| `MODEL_TYPE_PATH` | `./models/category_classifier` | Path to type classifier checkpoint. |
| `LOG_LEVEL` | `INFO` | Logging level. |
| `LOG_USER_TEXT` | `false` | Log user text at DEBUG level. Never enable in production. |

`ANTHROPIC_MODEL`, `ANTHROPIC_MAX_TOKENS`, and the decoding parameters are fixed to the thesis evaluation values. Changing them invalidates the reported BERTScore / ROUGE-L / BLEU-4 numbers.

---

## Running tests

```bash
pytest
```

The test suite covers:

- Pipeline round-trip on 30 deployment examples (real model inference, no mocking)
- E4 prompt builder determinism against a committed fixture
- Label-mapping coverage (all 10 classifier labels map to valid exemplar keys)
- Google Translate wrapper (mocked via `respx`, LRU cache verified)
- `/predict` endpoint contract for English distorted, English non-distorted, and Croatian distorted inputs (classifiers and Anthropic mocked)
