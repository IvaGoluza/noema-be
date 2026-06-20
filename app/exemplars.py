from data.deployment_exemplars import (
    CLASSIFIER_LABEL_TO_EXEMPLAR_KEY,
    DEPLOYMENT_EXEMPLARS,
)

from app.reframing import Example

LABEL_TO_KEY: dict[str, str] = CLASSIFIER_LABEL_TO_EXEMPLAR_KEY


def validate_exemplars() -> None:
    """Assert every key has exactly 3 entries. Called once at startup."""
    for key, entries in DEPLOYMENT_EXEMPLARS.items():
        assert len(entries) == 3, (
            f"Exemplar key '{key}' has {len(entries)} entries, expected 3."
        )


def get_exemplars(category_label: str) -> list[Example]:
    key = LABEL_TO_KEY[category_label]
    raw = DEPLOYMENT_EXEMPLARS[key]
    return [
        Example(
            situation=e["situation"],
            thought=e["thought"],
            reframe=e["reframe"],
            category=key,
        )
        for e in raw
    ]
