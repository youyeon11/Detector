from dataclasses import dataclass, field

from app.domain.enums import DetectionLabel
from app.domain.variation_types import VariationType

"""
전처리 결과
"""
@dataclass(slots=True)
class NormalizedQuery:
    raw: str
    normalized: str
    collapsed: str
    replaced: str
    tokens_for_debug: list[str] = field(default_factory=list)


"""
해당되는 후보군 사전 매칭 결과
"""
@dataclass(slots=True)
class LexiconHit:
    canonical: str
    variants: list[str]
    category: str
    severity: int
    risk_score: float
    es_score: float


"""
문장/문서 단위 검색 결과
"""
@dataclass(slots=True)
class DocumentHit:
    raw_text: str
    normalized_text: str
    collapsed_text: str
    replaced_text: str
    expected_label: str
    source: str
    notes: str
    tokens_for_debug: list[str]
    es_score: float


"""
스코어링 기반 결과물 산출
여러 매칭 방식(Exact, Norm, Ngram)과 비즈니스 로직을 결합하여 최종 점수 계산
"""
@dataclass(slots=True)
class RankedCandidate:
    canonical: str
    exact_match: float
    normalized_match: float
    ngram_match_score: float
    severity_weight: float
    context_meta_penalty: float
    final_score: float
    reasons: list[str] = field(default_factory=list)


"""
최종 결과
"""
@dataclass(slots=True)
class DetectionResult:
    label: DetectionLabel
    score: float
    matched_term: str | None
    reasons: list[str]
    normalized_text: str


@dataclass(slots=True)
class VariationClassificationResult:
    canonical: str
    matched_variant: str
    variation_type: VariationType
    reasons: list[str] = field(default_factory=list)


@dataclass(slots=True)
class V4ProfanityHit:
    canonical: str
    matched_variant: str
    variation_type: VariationType
    severity: int
    risk_score: float
    label: DetectionLabel
    score: float
    reasons: list[str] = field(default_factory=list)


@dataclass(slots=True)
class V4DetectionDocument:
    message: str
    message_normalized: str
    profanity_detected: bool
    profanity_hits: list[V4ProfanityHit] = field(default_factory=list)
