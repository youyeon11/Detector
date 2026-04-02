from app.config import Settings
from app.domain.enums import DetectionLabel
from app.domain.models import NormalizedQuery, RankedCandidate
from app.services.ambiguity_resolver import AmbiguityResolver


def make_query(
    *,
    raw: str,
    normalized: str | None = None,
    collapsed: str | None = None,
    replaced: str | None = None,
) -> NormalizedQuery:
    return NormalizedQuery(
        raw=raw,
        normalized=normalized if normalized is not None else raw,
        collapsed=collapsed if collapsed is not None else raw.replace(" ", ""),
        replaced=replaced if replaced is not None else raw,
        tokens_for_debug=[],
    )


def make_candidate(
    *,
    canonical: str = "시발",
    final_score: float = 9.2,
    context_meta_penalty: float = 0.0,
) -> RankedCandidate:
    return RankedCandidate(
        canonical=canonical,
        exact_match=1.0,
        normalized_match=1.0,
        ngram_match_score=0.8,
        severity_weight=0.95,
        context_meta_penalty=context_meta_penalty,
        final_score=final_score,
        reasons=["exact collapsed match"],
    )


def build_settings() -> Settings:
    return Settings(
        review_threshold=0.5,
        block_threshold=0.85,
    )


def test_resolver_keeps_block_for_clear_offensive_text() -> None:
    resolver = AmbiguityResolver(settings=build_settings())
    query = make_query(raw="씨발 왜 이렇게 늦어", collapsed="씨발왜이렇게늦어")
    candidate = make_candidate(final_score=9.1)

    resolved, label = resolver.resolve(normalized_query=query, candidate=candidate)

    assert resolved.final_score == 9.1
    assert label is DetectionLabel.BLOCK


def test_resolver_downgrades_meta_context_to_review() -> None:
    resolver = AmbiguityResolver(settings=build_settings())
    query = make_query(
        raw="욕설 예시로 씨발을 분석합니다",
        collapsed="욕설예시로씨발을분석합니다",
    )
    candidate = make_candidate(final_score=8.8)

    resolved, label = resolver.resolve(normalized_query=query, candidate=candidate)

    assert resolved.final_score < candidate.final_score
    assert any("meta-context penalty applied" in reason for reason in resolved.reasons)
    assert label is DetectionLabel.REVIEW


def test_resolver_whitelist_can_prevent_block_for_safe_term() -> None:
    resolver = AmbiguityResolver(settings=build_settings())
    query = make_query(raw="시발점에서 출발", collapsed="시발점에서출발")
    candidate = make_candidate(final_score=7.0)

    resolved, label = resolver.resolve(normalized_query=query, candidate=candidate)

    assert resolved.final_score == 4.0
    assert any("whitelist penalty applied" in reason for reason in resolved.reasons)
    assert label is DetectionLabel.PASS


def test_resolver_marks_safe_normal_word_as_pass() -> None:
    resolver = AmbiguityResolver(settings=build_settings())
    query = make_query(raw="개시 합니다", collapsed="개시합니다")
    candidate = make_candidate(canonical="개새끼", final_score=6.5)

    resolved, label = resolver.resolve(normalized_query=query, candidate=candidate)

    assert resolved.final_score == 3.5
    assert label is DetectionLabel.PASS
