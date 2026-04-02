from __future__ import annotations

from dataclasses import dataclass

from app.domain.enums import DetectionLabel
from app.domain.models import (
    DetectionResult,
    LexiconHit,
    NormalizedQuery,
    RankedCandidate,
    VariationDetectionDocument,
    VariationHit,
)
from app.services.detector import DetectorService
from app.services.variation_classifier import VariationClassifier


POSITIVE_LABELS = {DetectionLabel.BLOCK, DetectionLabel.REVIEW}


@dataclass(slots=True)
class VariationDetectionExecutionResult:
    detection_result: DetectionResult
    document_result: VariationDetectionDocument


class VariationDetectionService(DetectorService):
    def __init__(
        self,
        *,
        variation_classifier: VariationClassifier | None = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.variation_classifier = variation_classifier or VariationClassifier()

    def detect(self, text: str) -> DetectionResult:
        return self.detect_with_document(text).detection_result

    def detect_document(self, text: str) -> VariationDetectionDocument:
        return self.detect_with_document(text).document_result

    def detect_with_document(self, text: str) -> VariationDetectionExecutionResult:
        normalized_query = self.normalizer.normalize(text)
        lexicon_query = self.query_builder.build_lexicon_query(normalized_query)
        hits = self.lexicon_retriever.retrieve_lexicon_hits(search_body=lexicon_query)
        if not hits:
            hits = self._retrieve_local_fallback_hits(normalized_query)

        if not hits:
            return VariationDetectionExecutionResult(
                detection_result=DetectionResult(
                    label=DetectionLabel.PASS,
                    score=0.0,
                    matched_term=None,
                    reasons=["no lexicon candidates matched"],
                    normalized_text=normalized_query.normalized,
                ),
                document_result=VariationDetectionDocument(
                    message=text,
                    message_normalized=normalized_query.normalized,
                    profanity_detected=False,
                    profanity_hits=[],
                ),
            )

        ranked_candidates = self.ranker.rank(normalized_query=normalized_query, hits=hits)
        hits_by_canonical = {hit.canonical: hit for hit in hits}
        positive_hits: list[VariationHit] = []
        seen_canonicals: set[str] = set()
        resolved_top_candidate: RankedCandidate | None = None
        resolved_top_label: DetectionLabel = DetectionLabel.PASS

        for index, candidate in enumerate(ranked_candidates):
            if candidate.canonical in seen_canonicals:
                continue
            hit = hits_by_canonical.get(candidate.canonical)
            if hit is None:
                continue

            resolved_candidate, label = self.resolver.resolve(
                normalized_query=normalized_query,
                candidate=candidate,
            )
            if index == 0:
                resolved_top_candidate = resolved_candidate
                resolved_top_label = label
            seen_canonicals.add(candidate.canonical)

            if label not in POSITIVE_LABELS:
                continue

            matched_variant = self._find_matched_variant(normalized_query, hit)
            classification = self.variation_classifier.classify(
                canonical=hit.canonical,
                matched_variant=matched_variant,
                normalized_variant=normalized_query.normalized,
                collapsed_variant=normalized_query.collapsed,
            )
            positive_hits.append(
                VariationHit(
                    canonical=hit.canonical,
                    matched_variant=matched_variant,
                    variation_type=classification.variation_type,
                    severity=hit.severity,
                    risk_score=hit.risk_score,
                    label=label,
                    score=self._to_confidence(resolved_candidate),
                    reasons=[*resolved_candidate.reasons, *classification.reasons],
                )
            )

        if resolved_top_candidate is None:
            return VariationDetectionExecutionResult(
                detection_result=DetectionResult(
                    label=DetectionLabel.PASS,
                    score=0.0,
                    matched_term=None,
                    reasons=["no ranked candidates resolved"],
                    normalized_text=normalized_query.normalized,
                ),
                document_result=VariationDetectionDocument(
                    message=text,
                    message_normalized=normalized_query.normalized,
                    profanity_detected=False,
                    profanity_hits=[],
                ),
            )

        return VariationDetectionExecutionResult(
            detection_result=DetectionResult(
                label=resolved_top_label,
                score=self._to_confidence(resolved_top_candidate),
                matched_term=resolved_top_candidate.canonical,
                reasons=resolved_top_candidate.reasons,
                normalized_text=normalized_query.normalized,
            ),
            document_result=VariationDetectionDocument(
                message=text,
                message_normalized=normalized_query.normalized,
                profanity_detected=bool(positive_hits),
                profanity_hits=positive_hits,
            ),
        )

    def _find_matched_variant(self, normalized_query: NormalizedQuery, hit: LexiconHit) -> str:
        candidates = [hit.canonical, *hit.variants]
        raw_lower = normalized_query.raw.lower()
        normalized_lower = normalized_query.normalized.lower()
        collapsed_lower = normalized_query.collapsed.lower()

        for candidate in sorted(candidates, key=len, reverse=True):
            lowered = candidate.lower()
            collapsed_candidate = self._collapse(candidate).lower()
            if lowered and (lowered in raw_lower or lowered in normalized_lower):
                return candidate
            if collapsed_candidate and collapsed_candidate in collapsed_lower:
                return candidate
        return hit.canonical


def build_ranked_candidate(
    *,
    hit: LexiconHit,
    final_score: float,
    reasons: list[str] | None = None,
) -> RankedCandidate:
    return RankedCandidate(
        canonical=hit.canonical,
        exact_match=1.0,
        normalized_match=1.0,
        ngram_match_score=0.8,
        severity_weight=min(max(hit.severity, 0), 5) / 5,
        context_meta_penalty=0.0,
        final_score=final_score,
        reasons=reasons or [f"candidate={hit.canonical}"],
    )


def get_variation_detection_service() -> VariationDetectionService:
    return VariationDetectionService()


# Backward-compatible aliases during the Variation Detection rename.
DetectorExecutionResult = VariationDetectionExecutionResult
DetectorServiceV4 = VariationDetectionService
get_detector_service_v4 = get_variation_detection_service
