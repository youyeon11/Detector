from __future__ import annotations

from fastapi import APIRouter, Body, Depends

from app.schemas.request import DetectRequest
from app.schemas.response import DetectResponse, VariationHitResponse
from app.services.variation_detection import VariationDetectionService, get_variation_detection_service


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
    detector: VariationDetectionService = Depends(get_variation_detection_service),
) -> DetectResponse:
    execution = detector.detect_with_document(request.text)
    result = execution.detection_result
    document = execution.document_result
    return DetectResponse(
        label=result.label,
        score=result.score,
        matched_term=result.matched_term,
        reasons=result.reasons,
        normalized_text=result.normalized_text,
        profanity_detected=document.profanity_detected,
        profanity_hits=[
            VariationHitResponse(
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
