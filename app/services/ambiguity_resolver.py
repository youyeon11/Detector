from __future__ import annotations

from dataclasses import replace
from typing import Protocol

from app.config import Settings, get_settings
from app.domain.enums import DetectionLabel
from app.domain.models import NormalizedQuery, RankedCandidate


class MorphologyPenaltyProvider(Protocol):
    def penalty_for(
        self,
        *,
        normalized_query: NormalizedQuery,
        candidate: RankedCandidate,
    ) -> float: ...


class AmbiguityResolver:
    SAFE_PATTERNS = (
        "시발점",
        "시발역",
        "개시",
        "씨앗",
        "발가락",
    )
    META_PATTERNS = (
        "사용 금지",
        "예시",
        "분석",
        "탐지",
        "욕설",
        "설명",
    )

    def __init__(
        self,
        *,
        settings: Settings | None = None,
        morphology_penalty_provider: MorphologyPenaltyProvider | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.morphology_penalty_provider = morphology_penalty_provider

    def resolve(
        self,
        *,
        normalized_query: NormalizedQuery,
        candidate: RankedCandidate,
    ) -> tuple[RankedCandidate, DetectionLabel]:
        whitelist_penalty = self.compute_whitelist_penalty(normalized_query)
        meta_penalty = self.compute_meta_penalty(normalized_query)
        morphology_penalty = self.compute_morphology_penalty(
            normalized_query=normalized_query,
            candidate=candidate,
        )
        total_penalty = round(whitelist_penalty + meta_penalty + morphology_penalty, 3)
        adjusted_score = round(max(candidate.final_score - total_penalty, 0.0), 3)

        reasons = list(candidate.reasons)
        if whitelist_penalty > 0.0:
            reasons.append(f"whitelist penalty applied ({whitelist_penalty})")
        if meta_penalty > 0.0:
            reasons.append(f"meta-context penalty applied ({meta_penalty})")
        if morphology_penalty > 0.0:
            reasons.append(f"morphology penalty applied ({morphology_penalty})")
        reasons.append(f"resolved_score={adjusted_score}")

        resolved_candidate = replace(
            candidate,
            context_meta_penalty=round(candidate.context_meta_penalty + total_penalty, 3),
            final_score=adjusted_score,
            reasons=reasons,
        )
        label = self.classify(
            normalized_query=normalized_query,
            candidate=resolved_candidate,
            whitelist_penalty=whitelist_penalty,
            meta_penalty=meta_penalty,
        )
        return resolved_candidate, label

    def compute_whitelist_penalty(self, normalized_query: NormalizedQuery) -> float:
        haystacks = {
            normalized_query.raw,
            normalized_query.normalized,
            normalized_query.replaced,
            normalized_query.collapsed,
        }
        if any(pattern in haystack for pattern in self.SAFE_PATTERNS for haystack in haystacks):
            return 3.0
        return 0.0

    def compute_meta_penalty(self, normalized_query: NormalizedQuery) -> float:
        if any(pattern in normalized_query.raw for pattern in self.META_PATTERNS):
            return 1.5
        return 0.0

    def compute_morphology_penalty(
        self,
        *,
        normalized_query: NormalizedQuery,
        candidate: RankedCandidate,
    ) -> float:
        if self.morphology_penalty_provider is None:
            return 0.0
        return round(
            max(
                self.morphology_penalty_provider.penalty_for(
                    normalized_query=normalized_query,
                    candidate=candidate,
                ),
                0.0,
            ),
            3,
        )

    def classify(
        self,
        *,
        normalized_query: NormalizedQuery,
        candidate: RankedCandidate,
        whitelist_penalty: float,
        meta_penalty: float,
    ) -> DetectionLabel:
        confidence = min(candidate.final_score / 10.0, 1.0)

        if whitelist_penalty > 0.0 and confidence < self.settings.block_threshold:
            return DetectionLabel.PASS
        if meta_penalty > 0.0:
            if confidence >= self.settings.review_threshold:
                return DetectionLabel.REVIEW
            return DetectionLabel.PASS
        if confidence >= self.settings.block_threshold:
            return DetectionLabel.BLOCK
        if confidence >= self.settings.review_threshold:
            return DetectionLabel.REVIEW
        return DetectionLabel.PASS
