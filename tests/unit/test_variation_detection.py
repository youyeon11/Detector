import pytest

from app.domain.enums import DetectionLabel
from app.domain.models import LexiconHit, NormalizedQuery, RankedCandidate
from app.domain.variation_types import VariationType
from app.services.variation_detection import VariationDetectionService, build_ranked_candidate


pytestmark = pytest.mark.unit


class FakeNormalizer:
    def __init__(self, normalized_query: NormalizedQuery) -> None:
        self.normalized_query = normalized_query

    def normalize(self, text: str) -> NormalizedQuery:
        assert isinstance(text, str)
        return self.normalized_query


class FakeQueryBuilder:
    def build_lexicon_query(self, normalized_query: NormalizedQuery) -> dict:
        return {"query": {"match_all": {}}, "_source": ["canonical"]}


class FakeLexiconRetriever:
    def __init__(self, hits: list[LexiconHit]) -> None:
        self.hits = hits
        self.call_count = 0

    def retrieve_lexicon_hits(self, *, search_body: dict) -> list[LexiconHit]:
        assert "query" in search_body
        self.call_count += 1
        return self.hits


class FakeRanker:
    def __init__(self, ranked_candidates: list[RankedCandidate]) -> None:
        self.ranked_candidates = ranked_candidates

    def rank(self, *, normalized_query: NormalizedQuery, hits: list[LexiconHit]) -> list[RankedCandidate]:
        return self.ranked_candidates


class FakeResolver:
    def __init__(self, labels_by_canonical: dict[str, DetectionLabel]) -> None:
        self.labels_by_canonical = labels_by_canonical

    def resolve(
        self,
        *,
        normalized_query: NormalizedQuery,
        candidate: RankedCandidate,
    ) -> tuple[RankedCandidate, DetectionLabel]:
        return candidate, self.labels_by_canonical[candidate.canonical]


def make_query(
    *,
    raw: str,
    normalized: str | None = None,
    collapsed: str | None = None,
    replaced: str | None = None,
) -> NormalizedQuery:
    normalized_value = normalized if normalized is not None else raw
    collapsed_value = collapsed if collapsed is not None else normalized_value.replace(" ", "")
    replaced_value = replaced if replaced is not None else raw
    return NormalizedQuery(
        raw=raw,
        normalized=normalized_value,
        collapsed=collapsed_value,
        replaced=replaced_value,
        tokens_for_debug=[],
    )


def make_hit(
    *,
    canonical: str,
    variants: list[str],
    severity: int = 5,
    risk_score: float = 0.99,
) -> LexiconHit:
    return LexiconHit(
        canonical=canonical,
        variants=variants,
        category="profanity",
        severity=severity,
        risk_score=risk_score,
        es_score=5.0,
    )


def test_detector_v4_returns_empty_document_when_no_hits() -> None:
    query = make_query(raw="오늘은 평화로운 문장입니다")
    detector = VariationDetectionService(
        normalizer=FakeNormalizer(query),
        query_builder=FakeQueryBuilder(),
        lexicon_retriever=FakeLexiconRetriever([]),
        ranker=FakeRanker([]),
        resolver=FakeResolver({}),
        fallback_lexicon_hits=[],
    )

    result = detector.detect_document("오늘은 평화로운 문장입니다")

    assert result.profanity_detected is False
    assert result.profanity_hits == []
    assert result.message_normalized == "오늘은 평화로운 문장입니다"


def test_detector_v4_returns_single_classified_hit() -> None:
    query = make_query(raw="ssibal 진짜 짜증나네")
    hit = make_hit(canonical="시발", variants=["ssibal", "씨발"])
    detector = VariationDetectionService(
        normalizer=FakeNormalizer(query),
        query_builder=FakeQueryBuilder(),
        lexicon_retriever=FakeLexiconRetriever([hit]),
        ranker=FakeRanker([build_ranked_candidate(hit=hit, final_score=9.2)]),
        resolver=FakeResolver({"시발": DetectionLabel.BLOCK}),
        fallback_lexicon_hits=[],
    )

    result = detector.detect_document("ssibal 진짜 짜증나네")

    assert result.profanity_detected is True
    assert len(result.profanity_hits) == 1
    assert result.profanity_hits[0].canonical == "시발"
    assert result.profanity_hits[0].matched_variant == "ssibal"
    assert result.profanity_hits[0].variation_type is VariationType.ALPHABET_SUBSTITUTION
    assert result.profanity_hits[0].label is DetectionLabel.BLOCK


def test_detector_v4_can_return_detection_and_document_from_single_execution() -> None:
    query = make_query(raw="ssibal in sentence")
    hit = make_hit(canonical="시발", variants=["ssibal", "씨발"])
    retriever = FakeLexiconRetriever([hit])
    detector = VariationDetectionService(
        normalizer=FakeNormalizer(query),
        query_builder=FakeQueryBuilder(),
        lexicon_retriever=retriever,
        ranker=FakeRanker([build_ranked_candidate(hit=hit, final_score=9.2)]),
        resolver=FakeResolver({"시발": DetectionLabel.BLOCK}),
        fallback_lexicon_hits=[],
    )

    execution = detector.detect_with_document("ssibal in sentence")

    assert execution.detection_result.label is DetectionLabel.BLOCK
    assert execution.detection_result.matched_term == "시발"
    assert execution.document_result.profanity_detected is True
    assert execution.document_result.profanity_hits[0].canonical == "시발"
    assert retriever.call_count == 1


def test_detector_v4_supports_multi_hit_and_skips_pass() -> None:
    query = make_query(raw="ㅅㅂ 개새키")
    hit_1 = make_hit(canonical="시발", variants=["ㅅㅂ", "씨발"])
    hit_2 = make_hit(canonical="개새끼", variants=["개새키", "개색기"])
    hit_3 = make_hit(canonical="병신", variants=["ㅂㅅ"])
    detector = VariationDetectionService(
        normalizer=FakeNormalizer(query),
        query_builder=FakeQueryBuilder(),
        lexicon_retriever=FakeLexiconRetriever([hit_1, hit_2, hit_3]),
        ranker=FakeRanker(
            [
                build_ranked_candidate(hit=hit_1, final_score=9.0),
                build_ranked_candidate(hit=hit_2, final_score=8.7),
                build_ranked_candidate(hit=hit_3, final_score=6.0),
            ]
        ),
        resolver=FakeResolver(
            {
                "시발": DetectionLabel.BLOCK,
                "개새끼": DetectionLabel.REVIEW,
                "병신": DetectionLabel.PASS,
            }
        ),
        fallback_lexicon_hits=[],
    )

    result = detector.detect_document("ㅅㅂ 개새키")

    assert result.profanity_detected is True
    assert [item.canonical for item in result.profanity_hits] == ["시발", "개새끼"]
    assert result.profanity_hits[0].variation_type is VariationType.ABBREVIATION
    assert result.profanity_hits[1].variation_type is VariationType.PHONETIC_VARIATION
    assert all(item.label is not DetectionLabel.PASS for item in result.profanity_hits)
