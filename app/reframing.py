import re
import time
from dataclasses import dataclass
from typing import Any

from app.config import Settings

SYSTEM_MESSAGE = (
    "You are a cognitive behavioral therapy (CBT) assistant that helps people reframe "
    "unhelpful automatic thoughts. Given an unhelpful thought, you produce a single "
    "reframed thought that:\n"
    "- preserves the speaker's situation and the original referent of the thought,\n"
    "- reduces the dysfunctional element (catastrophizing, overgeneralization, mind "
    "reading, and so on) without denying the speaker's experience,\n"
    "- is realistic and balanced rather than dismissive, falsely positive, or preachy,\n"
    "- is written in the first person, in one or two sentences.\n\n"
    "Respond with the reframed thought only. Do not add explanations, labels, "
    "preambles, or quotation marks."
)


@dataclass(frozen=True)
class Example:
    situation: str
    thought: str
    reframe: str
    category: str


def build_prompt_e4(
    thought: str,
    situation: str,
    category: str,
    examples: list[Example],
    respond_in_croatian: bool = False,
) -> str:
    if len(examples) != 3:
        raise ValueError(f"E4 expects exactly 3 examples, got {len(examples)}.")

    header = (
        "Reframe the following unhelpful thought. The thought arises in a "
        "specific situation described below, and reflects a cognitive "
        f'distortion of the type "{category}". Below are three examples of '
        "reframing thoughts that exhibit this same distortion.\n\n"
    )
    example_blocks = "\n\n".join(
        f"Example {i + 1}\n"
        f"Situation: {ex.situation}\n"
        f"Thought: {ex.thought}\n"
        f"Reframed thought: {ex.reframe}"
        for i, ex in enumerate(examples)
    )
    footer = (
        "\n\nNow reframe the following thought, taking the situation into "
        f'account and addressing the "{category}" distortion.\n\n'
        f"Situation: {situation}\n"
        f"Thought: {thought}\n\n"
        "Reframed thought:"
    )
    if respond_in_croatian:
        footer = footer.replace(
            "Reframed thought:",
            "Respond in Croatian.\n\nReframed thought:",
        )
    return header + example_blocks + footer


def _clean_reframe(text: str) -> str:
    text = text.strip()
    for open_q, close_q in [('"', '"'), ("“", "”"), ("'", "'"), ("‘", "’")]:
        if text.startswith(open_q) and text.endswith(close_q) and len(text) > 1:
            text = text[len(open_q) : -len(close_q)].strip()
    text = re.sub(r"(?i)^reframed\s+thought\s*:\s*", "", text).strip()
    return text


def call_anthropic(prompt: str, client: Any, settings: Settings) -> dict[str, Any]:
    """Single call with up to settings.anthropic_max_retries retries on transient errors."""
    last_exc: Exception | None = None
    for attempt in range(settings.anthropic_max_retries):
        try:
            t0 = time.perf_counter()
            response = client.messages.create(
                model=settings.anthropic_model,
                max_tokens=settings.anthropic_max_tokens,
                system=SYSTEM_MESSAGE,
                messages=[{"role": "user", "content": prompt}],
            )
            latency_ms = (time.perf_counter() - t0) * 1000
            return {
                "text": _clean_reframe(response.content[0].text),
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
                "latency_ms": round(latency_ms, 1),
            }
        except Exception as exc:
            last_exc = exc
            if attempt < settings.anthropic_max_retries - 1:
                time.sleep(2**attempt)
    assert last_exc is not None
    raise last_exc
