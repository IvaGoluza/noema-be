"""Tests for the E4 prompt builder and the label-to-exemplar mapping coverage."""

from pathlib import Path

import pytest

from data.deployment_exemplars import CLASSIFIER_LABEL_TO_EXEMPLAR_KEY, DEPLOYMENT_EXEMPLARS
from app.exemplars import get_exemplars
from app.reframing import Example, build_prompt_e4

# Fixed inputs that match the content of tests/fixtures/e4_expected.txt.
FIXTURE_THOUGHT = "I'm going to fail this exam."
FIXTURE_SITUATION = "I have an exam tomorrow that I studied for."
FIXTURE_CATEGORY = "Jumping to conclusions: Fortune-telling"

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "e4_expected.txt"


def _fortune_telling_examples() -> list[Example]:
    """Return the three fortune-telling exemplars as Example objects."""
    return get_exemplars("Jumping to conclusions: Fortune-telling")


class TestE4PromptDeterminism:
    def test_matches_fixture(self) -> None:
        examples = _fortune_telling_examples()
        prompt = build_prompt_e4(
            thought=FIXTURE_THOUGHT,
            situation=FIXTURE_SITUATION,
            category=FIXTURE_CATEGORY,
            examples=examples,
        )
        expected = FIXTURE_PATH.read_text(encoding="utf-8")
        assert prompt == expected, (
            "build_prompt_e4 output does not match tests/fixtures/e4_expected.txt.\n"
            "If the prompt template changed intentionally, regenerate the fixture."
        )

    def test_croatian_flag_only_affects_footer(self) -> None:
        examples = _fortune_telling_examples()
        en_prompt = build_prompt_e4(
            thought=FIXTURE_THOUGHT,
            situation=FIXTURE_SITUATION,
            category=FIXTURE_CATEGORY,
            examples=examples,
            respond_in_croatian=False,
        )
        hr_prompt = build_prompt_e4(
            thought=FIXTURE_THOUGHT,
            situation=FIXTURE_SITUATION,
            category=FIXTURE_CATEGORY,
            examples=examples,
            respond_in_croatian=True,
        )
        assert hr_prompt.startswith(en_prompt[: en_prompt.rfind("Reframed thought:")])
        assert "Respond in Croatian." in hr_prompt
        assert "Respond in Croatian." not in en_prompt

    def test_wrong_number_of_examples_raises(self) -> None:
        examples = _fortune_telling_examples()
        with pytest.raises(ValueError, match="E4 expects exactly 3 examples"):
            build_prompt_e4(
                thought="x",
                situation="y",
                category="z",
                examples=examples[:2],
            )


class TestLabelMappingCoverage:
    def test_all_classifier_labels_have_exemplars(self) -> None:
        """Every PatternReframe label must map to a CRD key with exactly 3 entries."""
        for classifier_label, exemplar_key in CLASSIFIER_LABEL_TO_EXEMPLAR_KEY.items():
            assert exemplar_key in DEPLOYMENT_EXEMPLARS, (
                f"Classifier label '{classifier_label}' maps to key '{exemplar_key}', "
                "which is not in DEPLOYMENT_EXEMPLARS."
            )
            entries = DEPLOYMENT_EXEMPLARS[exemplar_key]
            assert len(entries) == 3, (
                f"Key '{exemplar_key}' has {len(entries)} entries, expected 3."
            )

    def test_get_exemplars_returns_three_for_every_label(self) -> None:
        for label in CLASSIFIER_LABEL_TO_EXEMPLAR_KEY:
            exs = get_exemplars(label)
            assert len(exs) == 3, f"get_exemplars('{label}') returned {len(exs)} examples."
            for ex in exs:
                assert isinstance(ex, Example)
                assert ex.situation
                assert ex.thought
                assert ex.reframe
