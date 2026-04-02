from app.domain.models import LexiconHit, NormalizedQuery
from app.services.ranker import CandidateRanker


def make_query(
    *,
    raw: str,
    normalized: str,
    collapsed: str,
    replaced: str,
) -> NormalizedQuery:
    return NormalizedQuery(
        raw=raw,
        normalized=normalized,
        collapsed=collapsed,
        replaced=replaced,
        tokens_for_debug=[],
    )


def test_ranker_prefers_higher_es_function_score() -> None:
    ranker = CandidateRanker()
    query = make_query(
        raw="시발",
        normalized="시발",
        collapsed="시발",
        replaced="시발",
    )
    hits = [
        LexiconHit(
            canonical="시발",
            variants=["씨발"],
            category="profanity",
            severity=5,
            risk_score=0.99,
            es_score=131.837,
        ),
        LexiconHit(
            canonical="병신",
            variants=["븅신"],
            category="abuse",
            severity=5,
            risk_score=0.99,
            es_score=77.125,
        ),
    ]

    ranked = ranker.rank(normalized_query=query, hits=hits)

    assert ranked[0].canonical == "시발"
    assert ranked[0].final_score == 131.837
    assert ranked[1].final_score == 77.125
    assert ranked[0].reasons == [
        "es_function_score=131.837",
        "candidate=시발",
    ]


def test_ranker_clamps_negative_es_score_to_zero() -> None:
    ranker = CandidateRanker()
    query = make_query(
        raw="badword in sentence",
        normalized="badword in sentence",
        collapsed="badwordinsentence",
        replaced="badword in sentence",
    )
    hit = LexiconHit(
        canonical="badword",
        variants=["bad word"],
        category="profanity",
        severity=5,
        risk_score=0.99,
        es_score=-3.2,
    )

    ranked = ranker.rank(normalized_query=query, hits=[hit])

    assert ranked[0].final_score == 0.0
    assert ranked[0].exact_match == 0.0
    assert ranked[0].normalized_match == 0.0
    assert ranked[0].ngram_match_score == 0.0
    assert ranked[0].severity_weight == 0.0
    assert ranked[0].context_meta_penalty == 0.0


def test_ranker_handles_empty_hits() -> None:
    ranker = CandidateRanker()
    query = make_query(
        raw="무해한 문장",
        normalized="무해한 문장",
        collapsed="무해한문장",
        replaced="무해한 문장",
    )

    ranked = ranker.rank(normalized_query=query, hits=[])

    assert ranked == []
