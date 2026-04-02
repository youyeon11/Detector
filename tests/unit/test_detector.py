from app.config import Settings
from app.domain.enums import DetectionLabel
from app.domain.models import DetectionResult, LexiconHit, NormalizedQuery, RankedCandidate
from app.services.detector import DetectorService
from app.services.normalizer import QueryNormalizer


class FakeNormalizer:
    def __init__(self, normalized_query: NormalizedQuery) -> None:
        self.normalized_query = normalized_query
        self.last_text: str | None = None

    def normalize(self, text: str) -> NormalizedQuery:
        self.last_text = text
        return self.normalized_query


class FakeQueryBuilder:
    def __init__(self) -> None:
        self.last_lexicon_query_input: NormalizedQuery | None = None
        self.last_document_query_input: NormalizedQuery | None = None

    def build_lexicon_query(self, normalized_query: NormalizedQuery) -> dict:
        self.last_lexicon_query_input = normalized_query
        return {"query": {"match_all": {}}, "_source": ["canonical"]}

    def build_document_query(self, normalized_query: NormalizedQuery) -> dict:
        self.last_document_query_input = normalized_query
        return {"query": {"match_all": {}}, "_source": ["raw_text"]}


class FakeLexiconRetriever:
    def __init__(self, hits: list[LexiconHit]) -> None:
        self.hits = hits
        self.last_search_body: dict | None = None

    def retrieve_lexicon_hits(self, *, search_body: dict) -> list[LexiconHit]:
        self.last_search_body = search_body
        return self.hits


class FakeRanker:
    def __init__(self, ranked_candidates: list[RankedCandidate]) -> None:
        self.ranked_candidates = ranked_candidates
        self.last_query: NormalizedQuery | None = None
        self.last_hits: list[LexiconHit] | None = None

    def rank(
        self,
        *,
        normalized_query: NormalizedQuery,
        hits: list[LexiconHit],
    ) -> list[RankedCandidate]:
        self.last_query = normalized_query
        self.last_hits = hits
        return self.ranked_candidates


class FakeResolver:
    def __init__(self, resolved_candidate: RankedCandidate, label: DetectionLabel) -> None:
        self.resolved_candidate = resolved_candidate
        self.label = label
        self.last_query: NormalizedQuery | None = None
        self.last_candidate: RankedCandidate | None = None

    def resolve(
        self,
        *,
        normalized_query: NormalizedQuery,
        candidate: RankedCandidate,
    ) -> tuple[RankedCandidate, DetectionLabel]:
        self.last_query = normalized_query
        self.last_candidate = candidate
        return self.resolved_candidate, self.label


def build_settings() -> Settings:
    return Settings(
        review_threshold=0.5,
        block_threshold=0.85,
    )


def make_normalized_query(
    *,
    raw: str = "씨발",
    normalized: str = "씨발",
    collapsed: str = "씨발",
    replaced: str = "씨발",
) -> NormalizedQuery:
    return NormalizedQuery(
        raw=raw,
        normalized=normalized,
        collapsed=collapsed,
        replaced=replaced,
        tokens_for_debug=["raw=씨발"],
    )


def make_hit() -> LexiconHit:
    return LexiconHit(
        canonical="시발",
        variants=["씨발"],
        category="profanity",
        severity=5,
        risk_score=0.99,
        es_score=5.0,
    )


def make_candidate(
    *,
    canonical: str = "시발",
    final_score: float = 9.1,
    reasons: list[str] | None = None,
) -> RankedCandidate:
    return RankedCandidate(
        canonical=canonical,
        exact_match=1.0,
        normalized_match=1.0,
        ngram_match_score=0.8,
        severity_weight=0.95,
        context_meta_penalty=0.0,
        final_score=final_score,
        reasons=reasons or ["exact collapsed match"],
    )


def test_detector_returns_pass_when_no_hits() -> None:
    normalized_query = make_normalized_query(raw="개시 합니다", normalized="개시 합니다", collapsed="개시합니다")
    detector = DetectorService(
        settings=build_settings(),
        normalizer=FakeNormalizer(normalized_query),
        query_builder=FakeQueryBuilder(),
        lexicon_retriever=FakeLexiconRetriever([]),
        ranker=FakeRanker([]),
        resolver=FakeResolver(make_candidate(final_score=0.0), DetectionLabel.PASS),
    )

    result = detector.detect("개시 합니다")

    assert result == DetectionResult(
        label=DetectionLabel.PASS,
        score=0.0,
        matched_term=None,
        reasons=["no lexicon candidates matched"],
        normalized_text="개시 합니다",
    )


def test_detector_runs_full_pipeline_for_block() -> None:
    normalized_query = make_normalized_query()
    hit = make_hit()
    ranked = make_candidate(final_score=9.2)
    resolved = make_candidate(final_score=9.2, reasons=["exact collapsed match", "resolved_score=9.2"])

    builder = FakeQueryBuilder()
    retriever = FakeLexiconRetriever([hit])
    ranker = FakeRanker([ranked])
    resolver = FakeResolver(resolved, DetectionLabel.BLOCK)
    detector = DetectorService(
        settings=build_settings(),
        normalizer=FakeNormalizer(normalized_query),
        query_builder=builder,
        lexicon_retriever=retriever,
        ranker=ranker,
        resolver=resolver,
    )

    result = detector.detect("씨발")

    assert result.label is DetectionLabel.BLOCK
    assert result.matched_term == "시발"
    assert result.score == 0.92
    assert builder.last_lexicon_query_input == normalized_query
    assert retriever.last_search_body == {"query": {"match_all": {}}, "_source": ["canonical"]}
    assert ranker.last_hits == [hit]
    assert resolver.last_candidate == ranked


def test_detector_can_downgrade_to_review_from_resolver() -> None:
    normalized_query = make_normalized_query(
        raw="욕설 예시로 씨발을 분석합니다",
        normalized="욕설 예시로 씨발을 분석합니다",
        collapsed="욕설예시로씨발을분석합니다",
        replaced="욕설 예시로 씨발을 분석합니다",
    )
    hit = make_hit()
    ranked = make_candidate(final_score=8.6)
    resolved = make_candidate(
        final_score=6.4,
        reasons=["context penalty applied (0.15)", "resolved_score=6.4"],
    )
    detector = DetectorService(
        settings=build_settings(),
        normalizer=FakeNormalizer(normalized_query),
        query_builder=FakeQueryBuilder(),
        lexicon_retriever=FakeLexiconRetriever([hit]),
        ranker=FakeRanker([ranked]),
        resolver=FakeResolver(resolved, DetectionLabel.REVIEW),
    )

    result = detector.detect("욕설 예시로 씨발을 분석합니다")

    assert result.label is DetectionLabel.REVIEW
    assert result.score == 0.64
    assert "resolved_score=6.4" in result.reasons


def test_detector_analyze_returns_query_previews() -> None:
    normalized_query = make_normalized_query(raw="씌발", normalized="씨발", collapsed="씨발", replaced="씨발")
    builder = FakeQueryBuilder()
    detector = DetectorService(
        settings=build_settings(),
        normalizer=FakeNormalizer(normalized_query),
        query_builder=builder,
        lexicon_retriever=FakeLexiconRetriever([]),
        ranker=FakeRanker([]),
        resolver=FakeResolver(make_candidate(final_score=0.0), DetectionLabel.PASS),
    )

    analyzed, preview = detector.analyze("씌발")

    assert analyzed == normalized_query
    assert "lexicon_query" in preview
    assert "document_query" in preview
    assert builder.last_lexicon_query_input == normalized_query
    assert builder.last_document_query_input == normalized_query


def test_detector_uses_local_fallback_hits_when_search_returns_none() -> None:
    normalized_query = make_normalized_query(
        raw="badword in sentence",
        normalized="badword in sentence",
        collapsed="badwordinsentence",
        replaced="badword in sentence",
    )
    fallback_hit = LexiconHit(
        canonical="badword",
        variants=["bad word"],
        category="profanity",
        severity=5,
        risk_score=0.99,
        es_score=0.0,
    )
    ranked = make_candidate(canonical="badword", final_score=8.8)
    resolved = make_candidate(canonical="badword", final_score=8.8, reasons=["resolved_score=8.8"])

    ranker = FakeRanker([ranked])
    resolver = FakeResolver(resolved, DetectionLabel.BLOCK)
    detector = DetectorService(
        settings=build_settings(),
        normalizer=FakeNormalizer(normalized_query),
        query_builder=FakeQueryBuilder(),
        lexicon_retriever=FakeLexiconRetriever([]),
        ranker=ranker,
        resolver=resolver,
        fallback_lexicon_hits=[fallback_hit],
    )

    result = detector.detect("badword in sentence")

    assert result.label is DetectionLabel.BLOCK
    assert result.matched_term == "badword"
    assert ranker.last_hits is not None
    assert ranker.last_hits[0].canonical == "badword"
    assert ranker.last_hits[0].es_score > 0.0


def test_detector_blocks_swiibal_variant_via_normalizer_and_fallback() -> None:
    fallback_hit = LexiconHit(
        canonical="시발",
        variants=["쉬이발", "시빨"],
        category="profanity",
        severity=5,
        risk_score=0.99,
        es_score=0.0,
    )
    detector = DetectorService(
        settings=build_settings(),
        normalizer=QueryNormalizer(),
        query_builder=FakeQueryBuilder(),
        lexicon_retriever=FakeLexiconRetriever([]),
        fallback_lexicon_hits=[fallback_hit],
    )

    result = detector.detect("쉬이발 왜 이래")

    assert result.label is not DetectionLabel.PASS
    assert result.matched_term == "시발"


def test_detector_blocks_sibbal_variant_via_normalizer_and_fallback() -> None:
    fallback_hit = LexiconHit(
        canonical="시발",
        variants=["쉬이발", "시빨"],
        category="profanity",
        severity=5,
        risk_score=0.99,
        es_score=0.0,
    )
    detector = DetectorService(
        settings=build_settings(),
        normalizer=QueryNormalizer(),
        query_builder=FakeQueryBuilder(),
        lexicon_retriever=FakeLexiconRetriever([]),
        fallback_lexicon_hits=[fallback_hit],
    )

    result = detector.detect("시빨 뭐하냐")

    assert result.label is not DetectionLabel.PASS
    assert result.matched_term == "시발"
