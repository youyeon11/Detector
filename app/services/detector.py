from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

from app.config import Settings, get_settings
from app.domain.enums import DetectionLabel
from app.domain.models import DetectionResult, LexiconHit, NormalizedQuery, RankedCandidate
from app.services.ambiguity_resolver import AmbiguityResolver
from app.services.normalizer import QueryNormalizer
from app.services.query_builder import QueryBuilder
from app.services.ranker import CandidateRanker
from app.services.retriever import LexiconRetriever

ROOT = Path(__file__).resolve().parents[2]
LOCAL_LEXICON_PATH = ROOT / "dataset" / "profanity_lexicon.jsonl"


class DetectorService:
    def __init__(
        self,
        *,
        settings: Settings | None = None,
        normalizer: QueryNormalizer | None = None,
        query_builder: QueryBuilder | None = None,
        lexicon_retriever: LexiconRetriever | None = None,
        ranker: CandidateRanker | None = None,
        resolver: AmbiguityResolver | None = None,
        fallback_lexicon_hits: list[LexiconHit] | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.normalizer = normalizer or QueryNormalizer()
        self.query_builder = query_builder or QueryBuilder(settings=self.settings)
        self.lexicon_retriever = lexicon_retriever or LexiconRetriever(settings=self.settings)
        self.ranker = ranker or CandidateRanker()
        self.resolver = resolver or AmbiguityResolver(settings=self.settings)
        self.fallback_lexicon_hits = (
            fallback_lexicon_hits
            if fallback_lexicon_hits is not None
            else load_local_lexicon_hits()
        )

    def detect(self, text: str) -> DetectionResult:
        normalized_query = self.normalizer.normalize(text)
        lexicon_query = self.query_builder.build_lexicon_query(normalized_query)
        hits = self.lexicon_retriever.retrieve_lexicon_hits(search_body=lexicon_query)
        if not hits:
            hits = self._retrieve_local_fallback_hits(normalized_query)

        if not hits:
            return DetectionResult(
                label=DetectionLabel.PASS,
                score=0.0,
                matched_term=None,
                reasons=["no lexicon candidates matched"],
                normalized_text=normalized_query.normalized,
            )

        ranked_candidates = self.ranker.rank(normalized_query=normalized_query, hits=hits)
        top_candidate = ranked_candidates[0]
        resolved_candidate, label = self.resolver.resolve(
            normalized_query=normalized_query,
            candidate=top_candidate,
        )

        return DetectionResult(
            label=label,
            score=self._to_confidence(resolved_candidate),
            matched_term=resolved_candidate.canonical,
            reasons=resolved_candidate.reasons,
            normalized_text=normalized_query.normalized,
        )

    def analyze(self, text: str) -> tuple[NormalizedQuery, dict]:
        normalized_query = self.normalizer.normalize(text)
        lexicon_query = self.query_builder.build_lexicon_query(normalized_query)
        document_query = self.query_builder.build_document_query(normalized_query)
        return normalized_query, {
            "lexicon_query": lexicon_query,
            "document_query": document_query,
        }

    def _to_confidence(self, candidate: RankedCandidate) -> float:
        return round(min(candidate.final_score / 10.0, 1.0), 3)

    def _retrieve_local_fallback_hits(
        self,
        normalized_query: NormalizedQuery,
    ) -> list[LexiconHit]:
        query_value = normalized_query.collapsed
        if not query_value:
            return []

        fallback_hits: list[LexiconHit] = []
        for hit in self.fallback_lexicon_hits:
            candidates = {self._collapse(hit.canonical)}
            candidates.update(self._collapse(variant) for variant in hit.variants)
            if any(candidate and candidate in query_value for candidate in candidates):
                fallback_hits.append(
                    LexiconHit(
                        canonical=hit.canonical,
                        variants=hit.variants,
                        category=hit.category,
                        severity=hit.severity,
                        risk_score=hit.risk_score,
                        es_score=self._estimate_local_fallback_score(
                            query_value=query_value,
                            candidates=candidates,
                            hit=hit,
                        ),
                    )
                )
        return fallback_hits

    def _collapse(self, text: str) -> str:
        return "".join(text.split())

    def _estimate_local_fallback_score(
        self,
        *,
        query_value: str,
        candidates: set[str],
        hit: LexiconHit,
    ) -> float:
        exact_match = 1.0 if any(candidate and candidate in query_value for candidate in candidates) else 0.0
        normalized_match = 1.0 if query_value in candidates else (0.7 if exact_match else 0.0)
        ngram_match_score = self._compute_ngram_overlap(query_value, candidates)
        severity_score = min(max(hit.severity, 0), 5) / 5
        severity_weight = (severity_score * 0.7) + (max(hit.risk_score, 0.0) * 0.3)

        estimated_score = (
            (exact_match * 4.0)
            + (normalized_match * 3.0)
            + (ngram_match_score * 2.0)
            + (severity_weight * 1.5)
        )
        return round(estimated_score, 3)

    def _compute_ngram_overlap(self, query_value: str, candidates: set[str], n: int = 2) -> float:
        query_ngrams = self._build_ngrams(query_value, n)
        if not query_ngrams:
            return 0.0

        best_score = 0.0
        for candidate in candidates:
            if not candidate:
                continue
            candidate_ngrams = self._build_ngrams(candidate, n)
            if not candidate_ngrams:
                continue
            union = query_ngrams | candidate_ngrams
            if not union:
                continue
            overlap = query_ngrams & candidate_ngrams
            best_score = max(best_score, len(overlap) / len(union))
        return round(best_score, 3)

    def _build_ngrams(self, text: str, n: int) -> set[str]:
        if len(text) < n:
            return set()
        return {text[index : index + n] for index in range(len(text) - n + 1)}


@lru_cache(maxsize=1)
def load_local_lexicon_hits(path: Path = LOCAL_LEXICON_PATH) -> list[LexiconHit]:
    if not path.exists():
        return []

    hits: list[LexiconHit] = []
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                row = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid local lexicon JSONL at line {line_number}: {path}") from exc

            canonical = row.get("canonical")
            variants = row.get("variants", [])
            if not isinstance(canonical, str) or not canonical.strip():
                continue
            if not isinstance(variants, list):
                variants = []

            hits.append(
                LexiconHit(
                    canonical=canonical,
                    variants=[item for item in variants if isinstance(item, str) and item.strip()],
                    category=row.get("category", "unknown") if isinstance(row.get("category"), str) else "unknown",
                    severity=int(row.get("severity", 0)) if isinstance(row.get("severity"), (int, float)) else 0,
                    risk_score=(
                        float(row.get("risk_score", 0.0))
                        if isinstance(row.get("risk_score"), (int, float))
                        else 0.0
                    ),
                    es_score=0.0,
                )
            )
    return hits


def get_detector_service() -> DetectorService:
    return DetectorService()
