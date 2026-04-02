from __future__ import annotations

from typing import Any

from app.config import Settings, get_settings
from app.domain.models import NormalizedQuery


class QueryBuilder:
    def __init__(self, *, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()

    def build_lexicon_query(
        self,
        normalized_query: NormalizedQuery,
        *,
        size: int = 10,
        min_score: float | None = None,
        minimum_should_match: str | int | None = None,
        prefix_length: int | None = None,
        include_edge: bool = True,
        include_nori: bool = False,
        include_fuzzy: bool = True,
    ) -> dict[str, Any]:
        should = []
        should.extend(self.build_exact_clauses(normalized_query, scope="lexicon"))
        should.extend(self.build_normalized_clauses(normalized_query, scope="lexicon"))
        should.extend(self.build_ngram_clauses(normalized_query, scope="lexicon"))
        if include_edge:
            should.extend(self.build_edge_clauses(normalized_query, scope="lexicon"))
        if include_nori:
            should.extend(self.build_nori_clauses(normalized_query, scope="lexicon"))
        if include_fuzzy:
            should.extend(
                self.build_fuzzy_clauses(
                    normalized_query,
                    scope="lexicon",
                    prefix_length=prefix_length,
                )
            )

        return self._build_search_body(
            normalized_query=normalized_query,
            scope="lexicon",
            should=should,
            size=size,
            min_score=min_score,
            source=[
                "canonical",
                "variants",
                "category",
                "severity",
                "risk_score",
            ],
            minimum_should_match=1 if minimum_should_match is None else minimum_should_match,
        )

    def build_document_query(
        self,
        normalized_query: NormalizedQuery,
        *,
        size: int = 10,
        min_score: float | None = None,
        minimum_should_match: str | int | None = None,
        prefix_length: int | None = None,
        include_edge: bool = True,
        include_nori: bool = False,
        include_fuzzy: bool = False,
    ) -> dict[str, Any]:
        should = []
        should.extend(self.build_exact_clauses(normalized_query, scope="docs"))
        should.extend(self.build_ngram_clauses(normalized_query, scope="docs"))
        if include_edge:
            should.extend(self.build_edge_clauses(normalized_query, scope="docs"))
        if include_nori:
            should.extend(self.build_nori_clauses(normalized_query, scope="docs"))
        if include_fuzzy:
            should.extend(
                self.build_fuzzy_clauses(
                    normalized_query,
                    scope="docs",
                    prefix_length=prefix_length,
                )
            )

        return self._build_search_body(
            normalized_query=normalized_query,
            scope="docs",
            should=should,
            size=size,
            min_score=min_score,
            source=[
                "raw_text",
                "normalized_text",
                "collapsed_text",
                "replaced_text",
                "expected_label",
                "source",
                "notes",
                "tokens_for_debug",
            ],
            minimum_should_match=minimum_should_match,
        )

    def build_exact_clauses(
        self,
        normalized_query: NormalizedQuery,
        *,
        scope: str,
    ) -> list[dict[str, Any]]:
        collapsed = normalized_query.collapsed
        if scope == "lexicon":
            return [
                {"term": {"canonical.keyword": {"value": collapsed, "boost": self.settings.term_boost}}},
                {"term": {"variants.keyword": {"value": collapsed, "boost": self.settings.term_boost}}},
            ]
        return [
            {"term": {"collapsed_text": {"value": collapsed, "boost": self.settings.term_boost}}},
            {"term": {"replaced_text": {"value": normalized_query.replaced, "boost": self.settings.term_boost}}},
        ]

    def build_normalized_clauses(
        self,
        normalized_query: NormalizedQuery,
        *,
        scope: str,
    ) -> list[dict[str, Any]]:
        text = normalized_query.collapsed
        if scope == "lexicon":
            return [
                {"match": {"canonical.norm": {"query": text, "boost": self.settings.norm_boost}}},
                {"match": {"variants.norm": {"query": text, "boost": self.settings.norm_boost - 1}}},
            ]
        return [
            {"match": {"normalized_text": {"query": text, "boost": self.settings.norm_boost}}},
            {"match": {"raw_text.norm": {"query": text, "boost": self.settings.norm_boost - 1}}},
        ]

    def build_ngram_clauses(
        self,
        normalized_query: NormalizedQuery,
        *,
        scope: str,
    ) -> list[dict[str, Any]]:
        text = normalized_query.collapsed
        if scope == "lexicon":
            return [
                {"match": {"canonical.ngram": {"query": text, "boost": self.settings.ngram_boost}}},
                {"match": {"variants.ngram": {"query": text, "boost": self.settings.ngram_boost}}},
            ]
        return [
            {"match": {"raw_text.ngram": {"query": text, "boost": self.settings.ngram_boost}}},
            {"match": {"normalized_text.ngram": {"query": text, "boost": self.settings.ngram_boost}}},
        ]

    def build_edge_clauses(
        self,
        normalized_query: NormalizedQuery,
        *,
        scope: str,
    ) -> list[dict[str, Any]]:
        text = normalized_query.collapsed
        if scope == "lexicon":
            return [
                {"match": {"canonical.edge": {"query": text, "boost": self.settings.edge_boost}}},
                {"match": {"variants.edge": {"query": text, "boost": self.settings.edge_boost}}},
            ]
        return [
            {"match": {"raw_text.edge": {"query": text, "boost": self.settings.edge_boost}}},
            {"match": {"normalized_text.edge": {"query": text, "boost": self.settings.edge_boost}}},
        ]

    def build_nori_clauses(
        self,
        normalized_query: NormalizedQuery,
        *,
        scope: str,
    ) -> list[dict[str, Any]]:
        text = normalized_query.raw
        if scope == "lexicon":
            return [
                {"match": {"canonical.nori": {"query": text, "boost": self.settings.nori_boost}}},
                {"match": {"variants.nori": {"query": text, "boost": self.settings.nori_boost}}},
            ]
        return [
            {"match": {"raw_text.nori": {"query": text, "boost": self.settings.nori_boost}}},
        ]

    def build_fuzzy_clauses(
        self,
        normalized_query: NormalizedQuery,
        *,
        scope: str,
        prefix_length: int | None = None,
    ) -> list[dict[str, Any]]:
        final_prefix_length = (
            prefix_length if prefix_length is not None else self.settings.prefix_length
        )
        text = normalized_query.collapsed
        fields = (
            ("canonical.norm", "variants.norm")
            if scope == "lexicon"
            else ("normalized_text", "raw_text.norm")
        )
        return [
            {
                "match": {
                    field: {
                        "query": text,
                        "boost": self.settings.fuzzy_boost,
                        "fuzziness": "AUTO",
                        "prefix_length": final_prefix_length,
                    }
                }
            }
            for field in fields
        ]

    def _build_search_body(
        self,
        *,
        normalized_query: NormalizedQuery,
        scope: str,
        should: list[dict[str, Any]],
        size: int,
        min_score: float | None,
        source: list[str],
        minimum_should_match: str | int | None,
    ) -> dict[str, Any]:
        final_min_score = self.settings.min_score if min_score is None else min_score
        final_msm = (
            self.settings.minimum_should_match
            if minimum_should_match is None
            else minimum_should_match
        )
        return {
            "size": size,
            "min_score": final_min_score,
            "_source": source,
            "query": {
                "function_score": {
                    "query": {
                        "bool": {
                            "should": should,
                            "minimum_should_match": final_msm,
                            "filter": [],
                        }
                    },
                    "score_mode": "sum",
                    "boost_mode": "sum",
                    "functions": [
                        {
                            "script_score": {
                                "script": self._build_rerank_script(
                                    normalized_query=normalized_query,
                                    scope=scope,
                                )
                            }
                        }
                    ],
                }
            },
        }

    def _build_rerank_script(
        self,
        *,
        normalized_query: NormalizedQuery,
        scope: str,
    ) -> dict[str, Any]:
        if scope == "lexicon":
            primary_field = "canonical.keyword"
            variant_field = "variants.keyword"
        else:
            primary_field = "collapsed_text"
            variant_field = "replaced_text"

        return {
            "source": """
double exact = 0.0;
double normalized = 0.0;
double ngram = 0.0;
double severityWeight = 0.0;
String queryCollapsed = params.query_collapsed;
List candidates = new ArrayList();
if (doc.containsKey(params.primary_field)) {
    for (def value : doc[params.primary_field]) {
        String candidateValue = value.toString().replace(' ', '').toLowerCase();
        candidates.add(candidateValue);
    }
}
if (doc.containsKey(params.variant_field)) {
    for (def value : doc[params.variant_field]) {
        String candidateValue = value.toString().replace(' ', '').toLowerCase();
        candidates.add(candidateValue);
    }
}

for (def candidate : candidates) {
    if (candidate == null || candidate.length() == 0) {
        continue;
    }
    if (candidate.equals(queryCollapsed)) {
        exact = 1.0;
    }
    if (queryCollapsed.equals(candidate)) {
        normalized = Math.max(normalized, 1.0);
    } else if (queryCollapsed.contains(candidate) || candidate.contains(queryCollapsed)) {
        normalized = Math.max(normalized, 0.7);
    }

    if (queryCollapsed.length() >= 2 && candidate.length() >= 2) {
        Set leftSet = new HashSet();
        for (int i = 0; i < queryCollapsed.length() - 1; ++i) {
            leftSet.add(queryCollapsed.substring(i, i + 2));
        }
        Set rightSet = new HashSet();
        for (int i = 0; i < candidate.length() - 1; ++i) {
            rightSet.add(candidate.substring(i, i + 2));
        }
        Set union = new HashSet(leftSet);
        union.addAll(rightSet);
        if (!union.isEmpty()) {
            Set overlap = new HashSet(leftSet);
            overlap.retainAll(rightSet);
            double currentNgram = overlap.size() / (double) union.size();
            if (currentNgram > ngram) {
                ngram = currentNgram;
            }
        }
    }
}

if (doc.containsKey('severity') && !doc['severity'].empty && doc.containsKey('risk_score') && !doc['risk_score'].empty) {
    double severity = Math.min(Math.max(doc['severity'].value, 0), 5) / 5.0;
    double risk = Math.max(doc['risk_score'].value, 0.0);
    severityWeight = (severity * 0.7) + (risk * 0.3);
}

return (exact * params.exact_weight)
    + (normalized * params.normalized_weight)
    + (ngram * params.ngram_weight)
    + (severityWeight * params.severity_weight)
    + (_score * params.bm25_weight);
""",
            "params": {
                "primary_field": primary_field,
                "variant_field": variant_field,
                "query_collapsed": normalized_query.collapsed.lower(),
                "exact_weight": 4.0,
                "normalized_weight": 3.0,
                "ngram_weight": 2.0,
                "severity_weight": 1.5,
                "bm25_weight": 0.1,
            },
        }
