import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def load_json(relative_path: str) -> dict:
    return json.loads((ROOT / relative_path).read_text(encoding="utf-8"))


def test_variation_index_uses_nested_hits_schema() -> None:
    body = load_json("elastic/variation_detected_messages_index.json")
    properties = body["mappings"]["properties"]

    assert properties["message"]["type"] == "text"
    assert properties["message_normalized"]["type"] == "text"
    assert properties["profanity_detected"]["type"] == "boolean"
    assert properties["profanity_hits"]["type"] == "nested"


def test_variation_hit_fields_use_analytics_friendly_types() -> None:
    body = load_json("elastic/variation_detected_messages_index.json")
    hit_fields = body["mappings"]["properties"]["profanity_hits"]["properties"]

    assert hit_fields["canonical"]["type"] == "keyword"
    assert hit_fields["matched_variant"]["type"] == "keyword"
    assert hit_fields["variation_type"]["type"] == "keyword"
    assert hit_fields["severity"]["type"] == "integer"
    assert hit_fields["risk_score"]["type"] == "float"
    assert hit_fields["label"]["type"] == "keyword"
    assert hit_fields["score"]["type"] == "float"
    assert hit_fields["reasons"]["type"] == "keyword"
