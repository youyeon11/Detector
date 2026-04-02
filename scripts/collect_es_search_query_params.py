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


LEXICON_FIELDS = [
    "canonical.norm^3",
    "variants.norm^2",
    "canonical.ngram^1",
    "variants.ngram^1",
]
DOCS_FIELDS = [
    "raw_text.ngram^1",
    "normalized_text.ngram^1",
]
MINIMUM_SHOULD_MATCH_VALUES = ["1", "70%", "2", "100%"]
MIN_SCORE_VALUES = [None, 0.1, 0.5, 1.0, 2.0]
PREFIX_LENGTH_VALUES = [0, 1, 2]


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
        description="Collect Elasticsearch search query parameter sweep logs."
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
        default=ROOT / "logs" / "es_search_query_params.json",
        help="Output path for query parameter log.",
    )
    return parser.parse_args()


def evaluate_hits(rows: list[dict]) -> dict:
    detect_targets = sum(1 for row in rows if row["expected_detection"] == "detect")
    review_targets = sum(1 for row in rows if row["expected_detection"] == "review")
    allow_targets = sum(1 for row in rows if row["expected_detection"] == "allow")

    detect_hits = sum(
        1 for row in rows if row["expected_detection"] == "detect" and row["hit_count"] > 0
    )
    review_hits = sum(
        1 for row in rows if row["expected_detection"] == "review" and row["hit_count"] > 0
    )
    allow_false_positives = sum(
        1 for row in rows if row["expected_detection"] == "allow" and row["hit_count"] > 0
    )

    avg_hit_count = round(sum(row["hit_count"] for row in rows) / max(len(rows), 1), 3)
    return {
        "detect_hits": detect_hits,
        "detect_targets": detect_targets,
        "review_hits": review_hits,
        "review_targets": review_targets,
        "allow_false_positives": allow_false_positives,
        "allow_targets": allow_targets,
        "detect_recall": round(detect_hits / max(detect_targets, 1), 3),
        "review_capture_rate": round(review_hits / max(review_targets, 1), 3),
        "allow_fp_rate": round(allow_false_positives / max(allow_targets, 1), 3),
        "avg_hit_count": avg_hit_count,
    }


def collect_multi_match_sweep(
    client: Elasticsearch,
    *,
    index: str,
    fields: list[str],
    samples: list[dict],
    scope: str,
) -> list[dict]:
    rows: list[dict] = []

    for minimum_should_match in MINIMUM_SHOULD_MATCH_VALUES:
        for min_score in MIN_SCORE_VALUES:
            case_rows: list[dict] = []
            for sample in samples:
                body: dict[str, object] = {
                    "size": 3,
                    "query": {
                        "multi_match": {
                            "query": sample["text"],
                            "fields": fields,
                            "minimum_should_match": minimum_should_match,
                        }
                    },
                }
                if min_score is not None:
                    body["min_score"] = min_score

                response = client.search(index=index, body=body)
                hits = response.body["hits"]["hits"]
                case_rows.append(
                    {
                        "id": sample["id"],
                        "text": sample["text"],
                        "expected_detection": sample["expected_detection"],
                        "hit_count": len(hits),
                        "top_hit": (
                            hits[0]["_source"].get("canonical")
                            or hits[0]["_source"].get("raw_text")
                        )
                        if hits
                        else None,
                    }
                )

            rows.append(
                {
                    "scope": scope,
                    "minimum_should_match": minimum_should_match,
                    "min_score": min_score,
                    "fields": fields,
                    "summary": evaluate_hits(case_rows),
                    "cases": case_rows,
                }
            )
    return rows


def collect_fuzzy_sweep(
    client: Elasticsearch,
    *,
    index: str,
    fields: list[str],
    samples: list[dict],
) -> list[dict]:
    rows: list[dict] = []
    for prefix_length in PREFIX_LENGTH_VALUES:
        case_rows: list[dict] = []
        for sample in samples:
            should = [
                {
                    "match": {
                        field: {
                            "query": sample["text"],
                            "fuzziness": "AUTO",
                            "prefix_length": prefix_length,
                        }
                    }
                }
                for field in fields
            ]
            body = {
                "size": 3,
                "query": {
                    "bool": {
                        "should": should,
                        "minimum_should_match": 1,
                    }
                },
            }
            response = client.search(index=index, body=body)
            hits = response.body["hits"]["hits"]
            case_rows.append(
                {
                    "id": sample["id"],
                    "text": sample["text"],
                    "expected_detection": sample["expected_detection"],
                    "hit_count": len(hits),
                    "top_hit": hits[0]["_source"].get("canonical") if hits else None,
                }
            )

        rows.append(
            {
                "prefix_length": prefix_length,
                "fields": fields,
                "summary": evaluate_hits(case_rows),
                "cases": case_rows,
            }
        )
    return rows


def main() -> int:
    args = parse_args()
    client = build_client()
    settings = get_settings()

    if not client.ping():
        raise RuntimeError("Elasticsearch is not reachable. Start ES and retry.")

    samples = load_input_set(args.input)

    payload = {
        "input_path": str(args.input),
        "current_defaults": {
            "minimum_should_match": settings.minimum_should_match,
            "min_score": settings.min_score,
            "term_boost": settings.term_boost,
            "norm_boost": settings.norm_boost,
            "ngram_boost": settings.ngram_boost,
            "fuzzy_boost": settings.fuzzy_boost,
        },
        "lexicon_multi_match": collect_multi_match_sweep(
            client,
            index=settings.profanity_lexicon_index,
            fields=LEXICON_FIELDS,
            samples=samples,
            scope="lexicon",
        ),
        "docs_multi_match": collect_multi_match_sweep(
            client,
            index=settings.noisy_text_docs_index,
            fields=DOCS_FIELDS,
            samples=samples,
            scope="docs",
        ),
        "lexicon_fuzzy_prefix": collect_fuzzy_sweep(
            client,
            index=settings.profanity_lexicon_index,
            fields=["canonical.norm", "variants.norm"],
            samples=samples,
        ),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote query parameter sweep log: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
