from pydantic import BaseModel, Field

from app.domain.enums import DetectionLabel
from app.domain.variation_types import VariationType


class VariationHitResponse(BaseModel):
    canonical: str
    matched_variant: str
    variation_type: VariationType
    severity: int = Field(..., ge=0, le=5)
    risk_score: float = Field(..., ge=0.0, le=1.0)
    label: DetectionLabel
    score: float = Field(..., ge=0.0)
    reasons: list[str] = Field(default_factory=list)


class DetectResponse(BaseModel):
    label: DetectionLabel
    score: float = Field(..., ge=0.0)
    matched_term: str | None = None
    reasons: list[str] = Field(default_factory=list)
    normalized_text: str
    profanity_detected: bool = False
    profanity_hits: list[VariationHitResponse] = Field(default_factory=list)


class AnalyzeQueryPreview(BaseModel):
    term: str
    norm: str
    ngram: str


class AnalyzeVariationPreview(BaseModel):
    message_normalized: str
    profanity_detected: bool = False
    profanity_hits: list[VariationHitResponse] = Field(default_factory=list)


class AnalyzeResponse(BaseModel):
    raw: str
    normalized: str
    collapsed: str
    replaced: str
    tokens_for_debug: list[str] = Field(default_factory=list)
    query_preview: AnalyzeQueryPreview
    variation_preview: AnalyzeVariationPreview | None = None


class VariationStatsBucketResponse(BaseModel):
    key: str | float | int
    doc_count: int = Field(..., ge=0)


class VariationStatsSummaryResponse(BaseModel):
    value: int | None = None
    relation: str | None = None


class VariationStatsResponse(BaseModel):
    index: str
    detected_hits_total: VariationStatsSummaryResponse
    top_canonical_buckets: list[VariationStatsBucketResponse] = Field(default_factory=list)
    variation_type_buckets: list[VariationStatsBucketResponse] = Field(default_factory=list)
    severity_histogram_buckets: list[VariationStatsBucketResponse] = Field(default_factory=list)


# Backward-compatible aliases during the Variation Detection rename.
V4ProfanityHitResponse = VariationHitResponse
AnalyzeV4Preview = AnalyzeVariationPreview
V4StatsBucketResponse = VariationStatsBucketResponse
V4StatsSummaryResponse = VariationStatsSummaryResponse
V4StatsResponse = VariationStatsResponse
