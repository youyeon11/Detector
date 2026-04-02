from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from elasticsearch import Elasticsearch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.config import get_settings
from scripts.collect_es_search_logs import load_input_set


COMPARE_FIELDS = {
    "ngram_docs": ("raw_text.ngram", "ko_ngram_search_analyzer", "ko_ngram_index_analyzer"),
    "edge_docs": ("raw_text.edge", "ko_edge_search_analyzer", "ko_edge_index_analyzer"),
    "ngram_lexicon": ("variants.ngram", "ko_ngram_search_analyzer", "ko_ngram_index_analyzer"),
    "edge_lexicon": ("variants.edge", "ko_edge_search_analyzer", "ko_edge_index_analyzer"),
}


def build_client() -> Elasticsearch:
    settings = get_settings()
    kwargs: dict[str, object] = {
        "hosts": [settings.elasticsearch_url],
        "request_timeout": settings.elasticsearch_request_timeout,
    }
    if settings.elasticsearch_username and settings.elasticsearch_password:
        kwargs["basic_auth"] = (
            settings.elasticsearch_username,
            settings.elasticsearch_password,
        )
    return Elasticsearch(**kwargs)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect Elasticsearch search index/search analyzer split comparison."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=ROOT / "dataset" / "es_search_input_set.json",
        help="ES search input set JSON path.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "logs" / "es_search_index_vs_search_compare.json",
        help="Output path for analyzer split comparison log.",
    )
    return parser.parse_args()


def build_match_query(field: str, text: str, analyzer: str | None) -> dict:
    field_query: dict[str, object] = {"query": text}
    if analyzer is not None:
        field_query["analyzer"] = analyzer
    return {
        "size": 3,
        "query": {
            "match": {
                field: field_query,
            }
        },
    }


def analyze_tokens(client: Elasticsearch, *, index: str, analyzer: str, text: str) -> list[str]:
    response = client.indices.analyze(
        index=index,
        body={"analyzer": analyzer, "text": text},
    )
    return [token["token"] for token in response.body.get("tokens", [])]


def run_search(client: Elasticsearch, *, index: str, field: str, text: str, analyzer: str | None) -> list[dict]:
    response = client.search(
        index=index,
        body=build_match_query(field, text, analyzer),
    )
    hits = response.body["hits"]["hits"]
    rows: list[dict] = []
    for hit in hits:
        source = hit["_source"]
        rows.append(
            {
                "id": hit["_id"],
                "score": hit["_score"],
                "canonical": source.get("canonical"),
                "raw_text": source.get("raw_text"),
                "expected_label": source.get("expected_label"),
            }
        )
    return rows


def summarize_rows(rows: list[dict]) -> dict:
    summary: dict[str, dict] = {}
    for key in COMPARE_FIELDS:
        total_gain = 0
        allow_fp_gain = 0
        detect_gain = 0
        review_gain = 0
        for row in rows:
            compared = row["comparisons"][key]
            default_hits = len(compared["default_search_hits"])
            index_hits = len(compared["index_override_hits"])
            delta = index_hits - default_hits
            total_gain += delta

            if delta > 0:
                expected = row["expected_detection"]
                if expected == "allow":
                    allow_fp_gain += 1
                elif expected == "detect":
                    detect_gain += 1
                elif expected == "review":
                    review_gain += 1

        summary[key] = {
            "total_hit_gain_when_using_index_analyzer": total_gain,
            "allow_case_gain_count": allow_fp_gain,
            "detect_case_gain_count": detect_gain,
            "review_case_gain_count": review_gain,
        }
    return summary


def main() -> int:
    args = parse_args()
    client = build_client()
    settings = get_settings()

    if not client.ping():
        raise RuntimeError("Elasticsearch is not reachable. Start ES and retry.")

    samples = load_input_set(args.input)
    rows: list[dict] = []

    for sample in samples:
        text = sample["text"]
        comparisons: dict[str, dict] = {}

        for key, (field, search_analyzer, index_analyzer) in COMPARE_FIELDS.items():
            index_name = (
                settings.noisy_text_docs_index
                if key.endswith("_docs")
                else settings.profanity_lexicon_index
            )
            comparisons[key] = {
                "field": field,
                "default_search_analyzer": search_analyzer,
                "index_analyzer": index_analyzer,
                "default_search_tokens": analyze_tokens(
                    client,
                    index=index_name,
                    analyzer=search_analyzer,
                    text=text,
                ),
                "index_override_tokens": analyze_tokens(
                    client,
                    index=index_name,
                    analyzer=index_analyzer,
                    text=text,
                ),
                "default_search_hits": run_search(
                    client,
                    index=index_name,
                    field=field,
                    text=text,
                    analyzer=None,
                ),
                "index_override_hits": run_search(
                    client,
                    index=index_name,
                    field=field,
                    text=text,
                    analyzer=index_analyzer,
                ),
            }

        rows.append(
            {
                "id": sample["id"],
                "text": text,
                "expected_detection": sample["expected_detection"],
                "notes": sample["notes"],
                "comparisons": comparisons,
            }
        )

    payload = {
        "summary": summarize_rows(rows),
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote analyzer split comparison log: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
