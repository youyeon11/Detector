import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = ROOT / "scripts" / "variation_query_examples.py"

spec = importlib.util.spec_from_file_location("variation_query_examples", MODULE_PATH)
assert spec is not None
assert spec.loader is not None
variation_query_examples = importlib.util.module_from_spec(spec)
spec.loader.exec_module(variation_query_examples)


class FakeElasticClient:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def search(self, *, index: str, body: dict) -> dict:
        self.calls.append({"index": index, "body": body})
        if "top_canonical" in str(body):
            return {
                "aggregations": {
                    "hits_nested": {
                        "top_canonical": {
                            "buckets": [{"key": "시발", "doc_count": 10}]
                        }
                    }
                }
            }
        if "variation_types" in str(body):
            return {
                "aggregations": {
                    "hits_nested": {
                        "variation_types": {
                            "buckets": [{"key": "alphabet_substitution", "doc_count": 5}]
                        }
                    }
                }
            }
        if "severity_histogram" in str(body):
            return {
                "aggregations": {
                    "hits_nested": {
                        "severity_histogram": {
                            "buckets": [{"key": 5.0, "doc_count": 12}]
                        }
                    }
                }
            }
        return {
            "hits": {
                "total": {"value": 30, "relation": "eq"},
                "hits": [{"_source": {"message": "시발 진짜 짜증나네"}}],
            }
        }


def test_detected_query_filters_on_profanity_detected() -> None:
    query = variation_query_examples.build_detected_query()

    assert query["query"]["term"]["profanity_detected"] is True
    assert "message" in query["_source"]


def test_aggregation_queries_use_nested_hits_path() -> None:
    top_canonical = variation_query_examples.build_top_canonical_agg()
    variation_types = variation_query_examples.build_variation_type_agg()
    severity_histogram = variation_query_examples.build_severity_histogram_agg()

    assert top_canonical["aggs"]["hits_nested"]["nested"]["path"] == "profanity_hits"
    assert variation_types["aggs"]["hits_nested"]["nested"]["path"] == "profanity_hits"
    assert severity_histogram["aggs"]["hits_nested"]["nested"]["path"] == "profanity_hits"


def test_build_report_collects_all_query_outputs() -> None:
    client = FakeElasticClient()

    report = variation_query_examples.build_report(client, "variation_detected_messages")

    assert report["index"] == "variation_detected_messages"
    assert report["results"]["detected_hits_total"]["value"] == 30
    assert report["results"]["top_canonical_buckets"][0]["key"] == "시발"
    assert report["results"]["variation_type_buckets"][0]["key"] == "alphabet_substitution"
    assert report["results"]["severity_histogram_buckets"][0]["key"] == 5.0
    assert len(client.calls) == 4
