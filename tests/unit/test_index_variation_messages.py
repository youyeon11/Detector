import importlib.util
from pathlib import Path

from app.domain.enums import DetectionLabel
from app.domain.models import V4DetectionDocument, V4ProfanityHit
from app.domain.variation_types import VariationType


ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = ROOT / "scripts" / "index_variation_messages.py"

spec = importlib.util.spec_from_file_location("index_variation_messages", MODULE_PATH)
assert spec is not None
assert spec.loader is not None
index_variation_messages = importlib.util.module_from_spec(spec)
spec.loader.exec_module(index_variation_messages)


class FakeDetectorServiceV4:
    def detect_document(self, text: str) -> V4DetectionDocument:
        if "시발" in text:
            return V4DetectionDocument(
                message=text,
                message_normalized=text,
                profanity_detected=True,
                profanity_hits=[
                    V4ProfanityHit(
                        canonical="시발",
                        matched_variant="시발",
                        variation_type=VariationType.PHONETIC_VARIATION,
                        severity=5,
                        risk_score=0.99,
                        label=DetectionLabel.BLOCK,
                        score=0.91,
                        reasons=["matched"],
                    )
                ],
            )
        return V4DetectionDocument(
            message=text,
            message_normalized=text,
            profanity_detected=False,
            profanity_hits=[],
        )


def test_build_actions_serializes_v4_document() -> None:
    rows = [
        {"text": "시발 진짜 짜증나네", "notes": "case-1"},
        {"text": "오늘 점심 뭐 먹지", "notes": "case-2"},
    ]

    actions = index_variation_messages.build_actions(
        "variation_detected_messages",
        rows,
        FakeDetectorServiceV4(),
    )

    assert len(actions) == 2
    assert actions[0]["_id"] == "case-1"
    assert actions[0]["_source"]["profanity_detected"] is True
    assert actions[0]["_source"]["profanity_hits"][0]["variation_type"] == "phonetic_variation"
    assert actions[1]["_source"]["profanity_detected"] is False


def test_build_document_id_uses_fallback_when_note_missing() -> None:
    document_id = index_variation_messages.build_document_id(row={"text": "hello"}, index=7)

    assert document_id == "v4-msg-7"
