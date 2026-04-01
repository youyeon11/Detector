from __future__ import annotations

from fastapi import APIRouter, Body, Depends

from app.schemas.request import AnalyzeRequest
from app.schemas.response import (
    AnalyzeQueryPreview,
    AnalyzeResponse,
    AnalyzeV4Preview,
    V4ProfanityHitResponse,
)
from app.services.detector_v4 import DetectorServiceV4, get_detector_service_v4


router = APIRouter(tags=["analyze"])


@router.post(
    "/analyze",
    response_model=AnalyzeResponse,
    summary="Inspect normalization and query preview",
)
def analyze(
    request: AnalyzeRequest = Body(
        ...,
        openapi_examples={
            "noisy": {
                "summary": "Noisy offensive variant",
                "value": {"text": "씌발"},
            },
            "meta": {
                "summary": "Meta analysis text",
                "value": {"text": "욕설 예시로 씨발을 분석합니다"},
            },
        },
    ),
    detector: DetectorServiceV4 = Depends(get_detector_service_v4),
) -> AnalyzeResponse:
    normalized, preview = detector.analyze(request.text)
    document = detector.detect_document(request.text)
    lexicon_query = preview["lexicon_query"]
    return AnalyzeResponse(
        raw=normalized.raw,
        normalized=normalized.normalized,
        collapsed=normalized.collapsed,
        replaced=normalized.replaced,
        tokens_for_debug=normalized.tokens_for_debug,
        query_preview=AnalyzeQueryPreview(
            term=normalized.collapsed,
            norm=normalized.collapsed,
            ngram=_extract_ngram_preview(lexicon_query),
        ),
        v4_preview=AnalyzeV4Preview(
            message_normalized=document.message_normalized,
            profanity_detected=document.profanity_detected,
            profanity_hits=[
                V4ProfanityHitResponse(
                    canonical=hit.canonical,
                    matched_variant=hit.matched_variant,
                    variation_type=hit.variation_type,
                    severity=hit.severity,
                    risk_score=hit.risk_score,
                    label=hit.label,
                    score=hit.score,
                    reasons=hit.reasons,
                )
                for hit in document.profanity_hits
            ],
        ),
    )


def _extract_ngram_preview(search_body: dict) -> str:
    should = search_body.get("query", {}).get("bool", {}).get("should", [])
    for clause in should:
        match = clause.get("match", {})
        if "canonical.ngram" in match:
            return str(match["canonical.ngram"].get("query", ""))
    return ""
