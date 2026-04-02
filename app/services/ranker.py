from __future__ import annotations

from app.domain.models import LexiconHit, NormalizedQuery, RankedCandidate


class CandidateRanker:
    def rank(
        self,
        *,
        normalized_query: NormalizedQuery,
        hits: list[LexiconHit],
    ) -> list[RankedCandidate]:
        del normalized_query
        ranked = [self._rank_hit(hit=hit) for hit in hits]
        return sorted(ranked, key=lambda item: item.final_score, reverse=True)

    def _rank_hit(
        self,
        *,
        hit: LexiconHit,
    ) -> RankedCandidate:
        final_score = round(max(hit.es_score, 0.0), 3)
        reasons = self.build_reasons(hit=hit, final_score=final_score)

        return RankedCandidate(
            canonical=hit.canonical,
            exact_match=0.0,
            normalized_match=0.0,
            ngram_match_score=0.0,
            severity_weight=0.0,
            context_meta_penalty=0.0,
            final_score=final_score,
            reasons=reasons,
        )

    def build_reasons(
        self,
        *,
        hit: LexiconHit,
        final_score: float,
    ) -> list[str]:
        return [
            f"es_function_score={final_score}",
            f"candidate={hit.canonical}",
        ]
