from app.config import Settings
from app.domain.models import NormalizedQuery
from app.services.query_builder import QueryBuilder


def make_query(
    *,
    raw: str = "씨발놈아",
    normalized: str = "씨발놈아",
    collapsed: str = "씨발놈아",
    replaced: str = "씨발놈아",
) -> NormalizedQuery:
    return NormalizedQuery(
        raw=raw,
        normalized=normalized,
        collapsed=collapsed,
        replaced=replaced,
        tokens_for_debug=[],
    )


def build_settings() -> Settings:
    return Settings(
        term_boost=10.0,
        norm_boost=8.0,
        ngram_boost=4.0,
        edge_boost=2.0,
        nori_boost=0.5,
        fuzzy_boost=3.0,
        min_score=0.5,
        minimum_should_match="70%",
        prefix_length=1,
    )


def extract_should(body: dict) -> list[dict]:
    return body["query"]["function_score"]["query"]["bool"]["should"]


def extract_script_params(body: dict) -> dict:
    return body["query"]["function_score"]["functions"][0]["script_score"]["script"]["params"]


def test_build_lexicon_query_returns_function_score_search_body() -> None:
    builder = QueryBuilder(settings=build_settings())

    body = builder.build_lexicon_query(make_query(), include_nori=False, include_fuzzy=True)

    assert body["size"] == 10
    assert body["min_score"] == 0.5
    assert body["_source"] == [
        "canonical",
        "variants",
        "category",
        "severity",
        "risk_score",
    ]
    function_score = body["query"]["function_score"]
    assert function_score["score_mode"] == "sum"
    assert function_score["boost_mode"] == "sum"
    assert function_score["query"]["bool"]["minimum_should_match"] == 1
    should = extract_should(body)
    assert any("term" in clause and "canonical.keyword" in clause["term"] for clause in should)
    assert any("match" in clause and "canonical.norm" in clause["match"] for clause in should)
    assert any("match" in clause and "canonical.ngram" in clause["match"] for clause in should)
    assert any("match" in clause and "canonical.edge" in clause["match"] for clause in should)
    assert any(
        "match" in clause
        and "canonical.norm" in clause["match"]
        and clause["match"]["canonical.norm"].get("fuzziness") == "AUTO"
        for clause in should
    )


def test_build_document_query_uses_function_score_defaults() -> None:
    builder = QueryBuilder(settings=build_settings())

    body = builder.build_document_query(make_query(raw="진짜씨발이야"), include_nori=False)

    should = extract_should(body)
    assert body["_source"] == [
        "raw_text",
        "normalized_text",
        "collapsed_text",
        "replaced_text",
        "expected_label",
        "source",
        "notes",
        "tokens_for_debug",
    ]
    assert any("term" in clause and "collapsed_text" in clause["term"] for clause in should)
    assert any("match" in clause and "raw_text.ngram" in clause["match"] for clause in should)
    assert any("match" in clause and "normalized_text.ngram" in clause["match"] for clause in should)
    assert any("match" in clause and "raw_text.edge" in clause["match"] for clause in should)
    assert not any("raw_text.nori" in clause.get("match", {}) for clause in should)
    assert not any(
        clause.get("match", {}).get("normalized_text", {}).get("fuzziness") == "AUTO"
        for clause in should
    )


def test_build_document_query_can_enable_nori_and_fuzzy() -> None:
    builder = QueryBuilder(settings=build_settings())

    body = builder.build_document_query(
        make_query(raw="시발점에서 출발", normalized="시발점에서 출발", collapsed="시발점에서출발"),
        include_nori=True,
        include_fuzzy=True,
        prefix_length=2,
        min_score=0.3,
    )

    should = extract_should(body)
    assert body["min_score"] == 0.3
    assert any("raw_text.nori" in clause.get("match", {}) for clause in should)
    fuzzy_clauses = [
        clause for clause in should if clause.get("match", {}).get("normalized_text", {}).get("fuzziness") == "AUTO"
    ]
    assert fuzzy_clauses
    assert fuzzy_clauses[0]["match"]["normalized_text"]["prefix_length"] == 2


def test_clause_builders_use_configured_boosts_and_function_score_params() -> None:
    builder = QueryBuilder(settings=build_settings())
    normalized_query = make_query(raw="씌발", normalized="씨발", collapsed="씨발", replaced="씨발")

    exact = builder.build_exact_clauses(normalized_query, scope="lexicon")
    norm = builder.build_normalized_clauses(normalized_query, scope="lexicon")
    ngram = builder.build_ngram_clauses(normalized_query, scope="lexicon")
    edge = builder.build_edge_clauses(normalized_query, scope="lexicon")
    nori = builder.build_nori_clauses(normalized_query, scope="lexicon")
    fuzzy = builder.build_fuzzy_clauses(normalized_query, scope="lexicon")
    body = builder.build_lexicon_query(normalized_query)
    params = extract_script_params(body)

    assert exact[0]["term"]["canonical.keyword"]["boost"] == 10.0
    assert norm[0]["match"]["canonical.norm"]["boost"] == 8.0
    assert ngram[0]["match"]["canonical.ngram"]["boost"] == 4.0
    assert edge[0]["match"]["canonical.edge"]["boost"] == 2.0
    assert nori[0]["match"]["canonical.nori"]["boost"] == 0.5
    assert fuzzy[0]["match"]["canonical.norm"]["prefix_length"] == 1
    assert params["query_collapsed"] == "씨발"
    assert params["exact_weight"] == 4.0
    assert params["normalized_weight"] == 3.0
    assert params["ngram_weight"] == 2.0
    assert params["severity_weight"] == 1.5
    assert params["bm25_weight"] == 0.1


def test_builder_allows_minimum_should_match_override() -> None:
    builder = QueryBuilder(settings=build_settings())

    body = builder.build_lexicon_query(make_query(raw="씨발", normalized="씨발", collapsed="씨발"), minimum_should_match="100%")

    assert body["query"]["function_score"]["query"]["bool"]["minimum_should_match"] == "100%"


def test_build_document_query_keeps_configured_minimum_should_match() -> None:
    builder = QueryBuilder(settings=build_settings())

    body = builder.build_document_query(make_query())

    assert body["query"]["function_score"]["query"]["bool"]["minimum_should_match"] == "70%"
