from app.domain.enums import DetectionLabel
from app.domain.models import DetectionResult, LexiconHit, NormalizedQuery, RankedCandidate
from app.schemas.request import AnalyzeRequest, DetectRequest
from app.schemas.response import AnalyzeQueryPreview, AnalyzeResponse, DetectResponse


def test_domain_models_hold_expected_values() -> None:
    query = NormalizedQuery(
        raw="씨-발",
        normalized="씨발",
        collapsed="씨발",
        replaced="씨발",
        tokens_for_debug=["씨발"],
    )
    hit = LexiconHit(
        canonical="시발",
        variants=["씨발", "ㅆ발"],
        category="offensive",
        severity=3,
        risk_score=0.9,
        es_score=12.5,
    )
    candidate = RankedCandidate(
        canonical=hit.canonical,
        exact_match=0.0,
        normalized_match=1.0,
        ngram_match_score=0.9,
        severity_weight=0.6,
        context_meta_penalty=0.0,
        final_score=9.8,
        reasons=["normalized match"],
    )
    result = DetectionResult(
        label=DetectionLabel.BLOCK,
        score=0.93,
        matched_term=hit.canonical,
        reasons=candidate.reasons,
        normalized_text=query.normalized,
    )

    assert result.label is DetectionLabel.BLOCK
    assert result.matched_term == "시발"
    assert result.normalized_text == "씨발"


def test_request_and_response_schemas_validate() -> None:
    detect_request = DetectRequest(text="씨-발 진짜")
    analyze_request = AnalyzeRequest(text="씨 발")

    detect_response = DetectResponse(
        label=DetectionLabel.BLOCK,
        score=0.91,
        matched_term="시발",
        reasons=["normalized match", "ngram overlap high"],
        normalized_text="씨발진짜",
    )
    analyze_response = AnalyzeResponse(
        raw=analyze_request.text,
        normalized="씨발",
        collapsed="씨발",
        replaced="씨발",
        tokens_for_debug=["씨발"],
        query_preview=AnalyzeQueryPreview(
            term="씨발",
            norm="씨발",
            ngram="씨발",
        ),
    )

    assert detect_request.text == "씨-발 진짜"
    assert detect_response.label == DetectionLabel.BLOCK
    assert analyze_response.query_preview.norm == "씨발"

