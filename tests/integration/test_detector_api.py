from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.domain.models import LexiconHit
from app.main import create_app
from app.services.variation_detection import (
    VariationDetectionService,
    get_variation_detection_service,
)


pytestmark = pytest.mark.integration


class StubLexiconRetriever:
    def __init__(self, hits: list[LexiconHit], *, ping_result: bool = True) -> None:
        self.hits = hits
        self.ping_result = ping_result

    def retrieve_lexicon_hits(self, *, search_body: dict) -> list[LexiconHit]:
        assert "query" in search_body
        return self.hits

    def ping(self) -> bool:
        return self.ping_result


def build_integration_client(retriever: StubLexiconRetriever) -> TestClient:
    app = create_app()
    detector = VariationDetectionService(lexicon_retriever=retriever)
    app.dependency_overrides[get_variation_detection_service] = lambda: detector
    return TestClient(app)


def test_detect_endpoint_runs_real_detector_pipeline(detector_cases: dict) -> None:
    case = detector_cases["detect"]["offensive"]
    client = build_integration_client(
        StubLexiconRetriever(
            [
                LexiconHit(
                    canonical="시발",
                    variants=["씨발", "ㅅㅂ"],
                    category="profanity",
                    severity=5,
                    risk_score=0.99,
                    es_score=9.2,
                )
            ]
        )
    )

    response = client.post("/detect", json={"text": case["text"]})

    assert response.status_code == 200
    body = response.json()
    assert body["label"] == case["expected_label"]
    assert body["matched_term"] == case["expected_term"]
    assert body["score"] >= 0.85


def test_detect_endpoint_returns_review_for_meta_case(detector_cases: dict) -> None:
    case = detector_cases["detect"]["boundary_review"]
    client = build_integration_client(
        StubLexiconRetriever(
            [
                LexiconHit(
                    canonical="시발",
                    variants=["씨발", "ㅅㅂ"],
                    category="profanity",
                    severity=5,
                    risk_score=0.99,
                    es_score=6.4,
                )
            ]
        )
    )

    response = client.post("/detect", json={"text": case["text"]})

    assert response.status_code == 200
    body = response.json()
    assert body["label"] == case["expected_label"]
    assert body["matched_term"] == case["expected_term"]


def test_analyze_endpoint_returns_real_normalization_preview(detector_cases: dict) -> None:
    case = detector_cases["analyze"]["noisy"]
    client = build_integration_client(StubLexiconRetriever([]))

    response = client.post("/analyze", json={"text": case["text"]})

    assert response.status_code == 200
    body = response.json()
    assert body["collapsed"] == case["expected_collapsed"]
    assert body["query_preview"]["term"] == case["expected_collapsed"]
    assert body["query_preview"]["ngram"] == case["expected_collapsed"]
    assert "variation_preview" in body


def test_health_endpoint_reports_ok_without_es(monkeypatch: pytest.MonkeyPatch) -> None:
    retriever = StubLexiconRetriever([], ping_result=True)
    monkeypatch.setattr(
        "app.main.DetectorService",
        lambda: VariationDetectionService(lexicon_retriever=retriever),
    )
    client = TestClient(create_app())

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_variation_stats_endpoint_returns_aggregates(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeElasticClient:
        def search(self, *, index: str, body: dict) -> dict:
            body_str = str(body)
            if "top_canonical" in body_str:
                return {
                    "aggregations": {
                        "hits_nested": {
                            "top_canonical": {"buckets": [{"key": "term", "doc_count": 4}]}
                        }
                    }
                }
            if "variation_types" in body_str:
                return {
                    "aggregations": {
                        "hits_nested": {
                            "variation_types": {"buckets": [{"key": "abbreviation", "doc_count": 2}]}
                        }
                    }
                }
            if "severity_histogram" in body_str:
                return {
                    "aggregations": {
                        "hits_nested": {
                            "severity_histogram": {"buckets": [{"key": 5.0, "doc_count": 4}]}
                        }
                    }
                }
            return {"hits": {"total": {"value": 4, "relation": "eq"}, "hits": []}}

    monkeypatch.setattr("app.api.stats.ElasticClient", lambda settings: FakeElasticClient())
    client = TestClient(create_app())

    response = client.get("/stats/variation")

    assert response.status_code == 200
    body = response.json()
    assert body["detected_hits_total"]["value"] == 4
    assert body["top_canonical_buckets"][0]["key"] == "term"
