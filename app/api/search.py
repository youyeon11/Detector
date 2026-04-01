from __future__ import annotations

from fastapi import APIRouter, Body, Depends

from app.schemas.request import DetectRequest
from app.schemas.response import DetectResponse, V4ProfanityHitResponse
from app.services.detector_v4 import DetectorServiceV4, get_detector_service_v4


router = APIRouter(tags=["detect"])


@router.post(
    "/detect",
    response_model=DetectResponse,
    summary="Detect offensive content",
)
def detect(
    request: DetectRequest = Body(
        ...,
        openapi_examples={
            "offensive": {
                "summary": "Direct offensive text",
                "value": {"text": "씨발 왜 이렇게 늦어"},
            },
            "safe": {
                "summary": "Safe ambiguous text",
                "value": {"text": "시발점에서 출발"},
            },
        },
    ),
    detector: DetectorServiceV4 = Depends(get_detector_service_v4),
) -> DetectResponse:
    result = detector.detect(request.text)
    document = detector.detect_document(request.text)
    return DetectResponse(
        label=result.label,
        score=result.score,
        matched_term=result.matched_term,
        reasons=result.reasons,
        normalized_text=result.normalized_text,
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
    )
