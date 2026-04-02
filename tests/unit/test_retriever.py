import pytest

from app.config import Settings
from app.services.elastic_client import ElasticClientError
from app.services.retriever import DocumentRetriever, LexiconRetriever, RetrieverError


class FakeElasticClient:
    def __init__(self) -> None:
        self.search_response: dict = {"hits": {"hits": []}}
        self.raise_on_search = False
        self.raise_on_ping = False
        self.raise_on_health = False
        self.last_search_kwargs: dict = {}

    def search(self, **kwargs):  # type: ignore[no-untyped-def]
        self.last_search_kwargs = kwargs
        if self.raise_on_search:
            raise ElasticClientError("search failed")
        return self.search_response

    def ping(self) -> bool:
        if self.raise_on_ping:
            raise ElasticClientError("ping failed")
        return True

    def health(self) -> dict:
        if self.raise_on_health:
            raise ElasticClientError("health failed")
        return {"status": "green"}


@pytest.fixture
def settings() -> Settings:
    return Settings(
        profanity_lexicon_index="profanity_lexicon",
        elasticsearch_request_timeout=1.5,
        min_score=0.42,
    )


@pytest.fixture
def fake_client() -> FakeElasticClient:
    return FakeElasticClient()


def test_retrieve_lexicon_hits_maps_response(settings: Settings, fake_client: FakeElasticClient) -> None:
    fake_client.search_response = {
        "hits": {
            "hits": [
                {
                    "_score": 12.3,
                    "_source": {
                        "canonical": "badword",
                        "variants": ["b@dword", "badword"],
                        "category": "offensive",
                        "severity": 3,
                        "risk_score": 0.9,
                    },
                }
            ]
        }
    }
    retriever = LexiconRetriever(settings=settings, elastic_client=fake_client)

    hits = retriever.retrieve_lexicon_hits(query={"match_all": {}})

    assert len(hits) == 1
    hit = hits[0]
    assert hit.canonical == "badword"
    assert hit.variants == ["b@dword", "badword"]
    assert hit.category == "offensive"
    assert hit.severity == 3
    assert hit.risk_score == 0.9
    assert hit.es_score == 12.3

    assert fake_client.last_search_kwargs["index"] == "profanity_lexicon"
    assert fake_client.last_search_kwargs["size"] == 20
    assert fake_client.last_search_kwargs["timeout"] == 1.5
    assert fake_client.last_search_kwargs["min_score"] == 0.42
    assert fake_client.last_search_kwargs["source_includes"] == [
        "canonical",
        "variants",
        "category",
        "severity",
        "risk_score",
    ]


def test_retrieve_lexicon_hits_skips_invalid_source(settings: Settings, fake_client: FakeElasticClient) -> None:
    fake_client.search_response = {
        "hits": {
            "hits": [
                {"_score": 2.0, "_source": None},
                {"_score": 1.0, "_source": {"canonical": ""}},
                {
                    "_score": 5.0,
                    "_source": {
                        "canonical": "badword",
                        "variants": "b@dword",
                    },
                },
            ]
        }
    }
    retriever = LexiconRetriever(settings=settings, elastic_client=fake_client)

    hits = retriever.retrieve_lexicon_hits(query={"match_all": {}})

    assert len(hits) == 1
    assert hits[0].canonical == "badword"
    assert hits[0].variants == ["b@dword"]
    assert hits[0].category == "unknown"
    assert hits[0].severity == 0
    assert hits[0].risk_score == 0.0


def test_retrieve_lexicon_hits_raises_retriever_error_on_es_error(
    settings: Settings,
    fake_client: FakeElasticClient,
) -> None:
    fake_client.raise_on_search = True
    retriever = LexiconRetriever(settings=settings, elastic_client=fake_client)

    with pytest.raises(RetrieverError):
        retriever.retrieve_lexicon_hits(query={"match_all": {}})


def test_retrieve_lexicon_hits_accepts_search_body(settings: Settings, fake_client: FakeElasticClient) -> None:
    fake_client.search_response = {
        "hits": {
            "hits": [
                {
                    "_score": 8.0,
                    "_source": {
                        "canonical": "badword",
                        "variants": ["badword"],
                        "category": "offensive",
                        "severity": 2,
                        "risk_score": 0.7,
                    },
                }
            ]
        }
    }
    retriever = LexiconRetriever(settings=settings, elastic_client=fake_client)

    hits = retriever.retrieve_lexicon_hits(search_body={"query": {"match_all": {}}})

    assert len(hits) == 1
    assert fake_client.last_search_kwargs["body"]["query"] == {"match_all": {}}
    assert fake_client.last_search_kwargs["body"]["min_score"] == 0.42
    assert fake_client.last_search_kwargs["body"]["_source"] == [
        "canonical",
        "variants",
        "category",
        "severity",
        "risk_score",
    ]


def test_document_retriever_maps_response(settings: Settings, fake_client: FakeElasticClient) -> None:
    fake_client.search_response = {
        "hits": {
            "hits": [
                {
                    "_score": 4.2,
                    "_source": {
                        "raw_text": "씨발 왜 이렇게 늦어",
                        "normalized_text": "씨발 왜 이렇게 늦어",
                        "collapsed_text": "씨발왜이렇게늦어",
                        "replaced_text": "씨발 왜 이렇게 늦어",
                        "expected_label": "BLOCK",
                        "source": "seed_eval",
                        "notes": "대표 변형",
                        "tokens_for_debug": ["raw=씨발 왜 이렇게 늦어"],
                    },
                }
            ]
        }
    }
    retriever = DocumentRetriever(settings=settings, elastic_client=fake_client)

    hits = retriever.retrieve_document_hits(query={"match_all": {}})

    assert len(hits) == 1
    hit = hits[0]
    assert hit.raw_text == "씨발 왜 이렇게 늦어"
    assert hit.expected_label == "BLOCK"
    assert hit.source == "seed_eval"
    assert hit.es_score == 4.2
    assert fake_client.last_search_kwargs["index"] == "noisy_text_docs"


def test_document_retriever_fills_defaults_for_partial_source(
    settings: Settings,
    fake_client: FakeElasticClient,
) -> None:
    fake_client.search_response = {
        "hits": {
            "hits": [
                {
                    "_score": 1.1,
                    "_source": {
                        "raw_text": "일반 문장",
                    },
                }
            ]
        }
    }
    retriever = DocumentRetriever(settings=settings, elastic_client=fake_client)

    hits = retriever.retrieve_document_hits(query={"match_all": {}})

    assert len(hits) == 1
    hit = hits[0]
    assert hit.normalized_text == "일반 문장"
    assert hit.collapsed_text == "일반문장"
    assert hit.expected_label == "UNKNOWN"
    assert hit.source == "unknown"
    assert hit.tokens_for_debug == []


def test_document_retriever_accepts_search_body(settings: Settings, fake_client: FakeElasticClient) -> None:
    fake_client.search_response = {
        "hits": {
            "hits": [
                {
                    "_score": 1.0,
                    "_source": {
                        "raw_text": "문장",
                    },
                }
            ]
        }
    }
    retriever = DocumentRetriever(settings=settings, elastic_client=fake_client)

    retriever.retrieve_document_hits(search_body={"query": {"match_all": {}}})

    assert fake_client.last_search_kwargs["body"]["query"] == {"match_all": {}}
    assert fake_client.last_search_kwargs["body"]["min_score"] == 0.42
    assert fake_client.last_search_kwargs["body"]["_source"] == [
        "raw_text",
        "normalized_text",
        "collapsed_text",
        "replaced_text",
        "expected_label",
        "source",
        "notes",
        "tokens_for_debug",
    ]


def test_retriever_ping_and_health(settings: Settings, fake_client: FakeElasticClient) -> None:
    retriever = LexiconRetriever(settings=settings, elastic_client=fake_client)

    assert retriever.ping() is True
    assert retriever.health() == {"status": "green"}


def test_retriever_ping_wraps_elastic_error(settings: Settings, fake_client: FakeElasticClient) -> None:
    fake_client.raise_on_ping = True
    retriever = LexiconRetriever(settings=settings, elastic_client=fake_client)

    with pytest.raises(RetrieverError):
        retriever.ping()


def test_retriever_health_wraps_elastic_error(settings: Settings, fake_client: FakeElasticClient) -> None:
    fake_client.raise_on_health = True
    retriever = LexiconRetriever(settings=settings, elastic_client=fake_client)

    with pytest.raises(RetrieverError):
        retriever.health()
