import pytest
from fastapi.testclient import TestClient


pytestmark = pytest.mark.unit


def test_detect_endpoint_returns_detection_result(unit_client: TestClient) -> None:
    response = unit_client.post("/detect", json={"text": "sample offensive input"})

    assert response.status_code == 200
    body = response.json()
    assert body["label"] == "BLOCK"
    assert body["matched_term"] is not None
    assert body["normalized_text"]
    assert body["profanity_detected"] is True
    assert body["profanity_hits"][0]["variation_type"] == "phonetic_variation"


def test_analyze_endpoint_returns_query_preview(unit_client: TestClient) -> None:
    response = unit_client.post("/analyze", json={"text": "sample noisy input"})

    assert response.status_code == 200
    body = response.json()
    assert body["raw"]
    assert body["normalized"]
    assert body["query_preview"]["term"]
    assert body["query_preview"]["ngram"]
    assert body["variation_preview"]["message_normalized"]
    assert body["variation_preview"]["profanity_hits"][0]["canonical"]


def test_health_endpoint_returns_ok(unit_client: TestClient) -> None:
    response = unit_client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_openapi_contains_detect_analyze_and_stats_paths(unit_client: TestClient) -> None:
    response = unit_client.get("/openapi.json")

    assert response.status_code == 200
    body = response.json()
    assert body["info"]["title"] == "Detector API"
    assert "/detect" in body["paths"]
    assert "/analyze" in body["paths"]
    assert "/stats/variation" in body["paths"]


def test_swagger_ui_and_redoc_are_exposed(unit_client: TestClient) -> None:
    docs_response = unit_client.get("/docs")
    redoc_response = unit_client.get("/redoc")

    assert docs_response.status_code == 200
    assert "swagger" in docs_response.text.lower()
    assert redoc_response.status_code == 200
    assert "redoc" in redoc_response.text.lower()


def test_variation_stats_endpoint_returns_aggregated_buckets(
    unit_client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeElasticClient:
        def search(self, *, index: str, body: dict) -> dict:
            body_str = str(body)
            if "top_canonical" in body_str:
                return {
                    "aggregations": {
                        "hits_nested": {
                            "top_canonical": {"buckets": [{"key": "term", "doc_count": 3}]}
                        }
                    }
                }
            if "variation_types" in body_str:
                return {
                    "aggregations": {
                        "hits_nested": {
                            "variation_types": {
                                "buckets": [{"key": "phonetic_variation", "doc_count": 2}]
                            }
                        }
                    }
                }
            if "severity_histogram" in body_str:
                return {
                    "aggregations": {
                        "hits_nested": {
                            "severity_histogram": {"buckets": [{"key": 5.0, "doc_count": 3}]}
                        }
                    }
                }
            return {"hits": {"total": {"value": 3, "relation": "eq"}, "hits": []}}

    monkeypatch.setattr("app.api.stats.ElasticClient", lambda settings: FakeElasticClient())

    response = unit_client.get("/stats/variation")

    assert response.status_code == 200
    body = response.json()
    assert body["detected_hits_total"]["value"] == 3
    assert body["top_canonical_buckets"][0]["key"] == "term"
    assert body["variation_type_buckets"][0]["key"] == "phonetic_variation"
