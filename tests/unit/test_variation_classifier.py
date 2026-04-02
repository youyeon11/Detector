import pytest

from app.domain.variation_types import VariationType
from app.services.variation_classifier import VariationClassifier


pytestmark = pytest.mark.unit


def test_classifier_detects_numeric_substitution() -> None:
    classifier = VariationClassifier()

    result = classifier.classify(canonical="시발", matched_variant="18")

    assert result.variation_type is VariationType.NUMERIC_SUBSTITUTION
    assert "variation_type=numeric_substitution" in result.reasons


def test_classifier_detects_alphabet_substitution() -> None:
    classifier = VariationClassifier()

    result = classifier.classify(canonical="시발", matched_variant="ssibal")

    assert result.variation_type is VariationType.ALPHABET_SUBSTITUTION


def test_classifier_detects_phonetic_variation() -> None:
    classifier = VariationClassifier()

    result = classifier.classify(canonical="병신", matched_variant="븅신")

    assert result.variation_type is VariationType.PHONETIC_VARIATION


def test_classifier_detects_abbreviation() -> None:
    classifier = VariationClassifier()

    result = classifier.classify(canonical="시발", matched_variant="ㅅㅂ")

    assert result.variation_type is VariationType.ABBREVIATION


def test_classifier_detects_spacing_variation_using_collapsed_text() -> None:
    classifier = VariationClassifier()

    result = classifier.classify(
        canonical="시발",
        matched_variant="시 발",
        normalized_variant="시 발",
        collapsed_variant="시발",
    )

    assert result.variation_type is VariationType.SPACING_VARIATION


def test_classifier_falls_back_to_mixed_variation_for_unknown_variant() -> None:
    classifier = VariationClassifier()

    result = classifier.classify(canonical="시발", matched_variant="시발xx")

    assert result.variation_type is VariationType.MIXED_VARIATION
    assert "no explicit variation rule matched" in result.reasons


def test_classifier_falls_back_to_mixed_variation_for_unknown_canonical() -> None:
    classifier = VariationClassifier()

    result = classifier.classify(canonical="없는욕설", matched_variant="없는욕설")

    assert result.variation_type is VariationType.MIXED_VARIATION
    assert "canonical rule not found" in result.reasons
