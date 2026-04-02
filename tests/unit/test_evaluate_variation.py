from dataclasses import dataclass
import uuid
from pathlib import Path

import pytest

from app.domain.enums import DetectionLabel
from app.domain.models import V4DetectionDocument, V4ProfanityHit
from app.domain.variation_types import VariationType
from scripts.evaluate_variation import evaluate_dataset, render_markdown_report, save_report


pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[2]


@dataclass
class FakeVariationDetector:
    results: dict[str, V4DetectionDocument]

    def detect_document(self, text: str) -> V4DetectionDocument:
        return self.results[text]


def make_hit(
    canonical: str,
    variation_type: VariationType,
    *,
    label: DetectionLabel = DetectionLabel.BLOCK,
    score: float = 0.95,
) -> V4ProfanityHit:
    return V4ProfanityHit(
        canonical=canonical,
        matched_variant=canonical,
        variation_type=variation_type,
        severity=5,
        risk_score=0.99,
        label=label,
        score=score,
        reasons=[f"matched={canonical}"],
    )


def make_document(text: str, hits: list[V4ProfanityHit]) -> V4DetectionDocument:
    return V4DetectionDocument(
        message=text,
        message_normalized=text,
        profanity_detected=bool(hits),
        profanity_hits=hits,
    )


def test_evaluate_variation_computes_detection_and_variation_metrics() -> None:
    dataset = [
        {
            "text": "safe text",
            "expected_detected": False,
            "expected_canonical": None,
            "expected_variation_type": None,
            "case_group": "safe",
        },
        {
            "text": "numeric profanity",
            "expected_detected": True,
            "expected_canonical": "시발",
            "expected_variation_type": "numeric_substitution",
            "case_group": "numeric_substitution",
        },
        {
            "text": "wrong variation",
            "expected_detected": True,
            "expected_canonical": "병신",
            "expected_variation_type": "spacing_variation",
            "case_group": "spacing_variation",
        },
        {
            "text": "missed profanity",
            "expected_detected": True,
            "expected_canonical": "개새끼",
            "expected_variation_type": "abbreviation",
            "case_group": "abbreviation",
        },
    ]
    detector = FakeVariationDetector(
        {
            "safe text": make_document("safe text", []),
            "numeric profanity": make_document(
                "numeric profanity",
                [make_hit("시발", VariationType.NUMERIC_SUBSTITUTION)],
            ),
            "wrong variation": make_document(
                "wrong variation",
                [make_hit("병신", VariationType.MIXED_VARIATION)],
            ),
            "missed profanity": make_document("missed profanity", []),
        }
    )

    report = evaluate_dataset(detector, dataset)

    assert report["summary"]["sample_count"] == 4
    assert report["summary"]["tp"] == 2
    assert report["summary"]["tn"] == 1
    assert report["summary"]["fn"] == 1
    assert report["summary"]["fp"] == 0
    assert report["summary"]["precision"] == 1.0
    assert report["summary"]["recall"] == 0.6667
    assert report["summary"]["f1"] == 0.8
    assert report["summary"]["canonical_match_accuracy"] == 0.6667
    assert report["summary"]["variation_classification_accuracy"] == 0.3333
    assert report["summary"]["variation_case_count"] == 3
    assert len(report["false_negatives"]) == 1
    assert len(report["wrong_variation_type"]) == 1
    assert report["variation_type_confusion"] == [
        {
            "expected_variation_type": "abbreviation",
            "predicted_variation_type": "None",
            "count": 1,
        },
        {
            "expected_variation_type": "numeric_substitution",
            "predicted_variation_type": "numeric_substitution",
            "count": 1,
        },
        {
            "expected_variation_type": "spacing_variation",
            "predicted_variation_type": "mixed_variation",
            "count": 1,
        },
    ]


def test_render_markdown_report_contains_variation_sections() -> None:
    report = {
        "summary": {
            "sample_count": 2,
            "tp": 1,
            "fp": 0,
            "tn": 1,
            "fn": 0,
            "precision": 1.0,
            "recall": 1.0,
            "f1": 1.0,
            "false_positive_rate": 0.0,
            "canonical_match_accuracy": 1.0,
            "variation_classification_accuracy": 1.0,
            "variation_case_count": 1,
        },
        "variation_type_confusion": [
            {
                "expected_variation_type": "alphabet_substitution",
                "predicted_variation_type": "alphabet_substitution",
                "count": 1,
            }
        ],
        "false_positives": [],
        "false_negatives": [],
        "wrong_variation_type": [],
        "rows": [],
    }

    content = render_markdown_report(report)

    assert "# Variation Evaluation Report" in content
    assert "Variation Classification Accuracy" in content
    assert "## Variation Type Confusion" in content
    assert "## Wrong Variation Type (0)" in content


def test_save_report_writes_json_output() -> None:
    report = {
        "summary": {
            "sample_count": 1,
            "tp": 0,
            "fp": 0,
            "tn": 1,
            "fn": 0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "false_positive_rate": 0.0,
            "canonical_match_accuracy": 0.0,
            "variation_classification_accuracy": 0.0,
            "variation_case_count": 0,
        },
        "variation_type_confusion": [],
        "false_positives": [],
        "false_negatives": [],
        "wrong_variation_type": [],
        "rows": [],
    }

    path = ROOT / "reports" / f"_test_variation_eval_{uuid.uuid4().hex}.json"
    try:
        save_report(path, report)
        assert path.exists() is True
        assert '"sample_count": 1' in path.read_text(encoding="utf-8")
    finally:
        if path.exists():
            path.unlink()
