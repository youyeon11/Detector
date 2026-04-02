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
from app.services.normalizer import normalize_text
from scripts.validate_phase4 import ensure_indices_exist


ANALYZERS = (
    "ko_norm_index_analyzer",
    "ko_ngram_index_analyzer",
    "ko_edge_index_analyzer",
    "ko_nori_analyzer",
)


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
    parser = argparse.ArgumentParser(description="Collect Elasticsearch search analyze/search logs.")
    parser.add_argument(
        "--input",
        type=Path,
        default=ROOT / "dataset" / "es_search_input_set.json",
        help="Input set JSON path.",
    )
    parser.add_argument(
        "--input-log",
        type=Path,
        default=ROOT / "logs" / "es_search_input_set.json",
        help="Output path for frozen input set copy.",
    )
    parser.add_argument(
        "--analyze-output",
        type=Path,
        default=ROOT / "logs" / "es_search_analyze.json",
        help="Output path for _analyze logs.",
    )
    parser.add_argument(
        "--search-output",
        type=Path,
        default=ROOT / "logs" / "es_search_results.json",
        help="Output path for _search logs.",
    )
    return parser.parse_args()


def load_input_set(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    if not isinstance(payload, list):
        raise ValueError("ES search input set must be a JSON array.")
    return payload


def collect_analyze_log(client: Elasticsearch, samples: list[dict]) -> dict:
    settings = get_settings()
    rows: list[dict] = []

    for sample in samples:
        text = sample["text"]
        normalized = normalize_text(text)

        analyzer_results: dict[str, list[str]] = {}
        for analyzer in ANALYZERS:
            response = client.indices.analyze(
                index=settings.profanity_lexicon_index,
                body={"analyzer": analyzer, "text": text},
            )
            analyzer_results[analyzer] = [
                token["token"] for token in response.get("tokens", [])
            ]

        rows.append(
            {
                **sample,
                "python_normalized": normalized.normalized,
                "python_collapsed": normalized.collapsed,
                "python_replaced": normalized.replaced,
                "python_tokens_for_debug": normalized.tokens_for_debug,
                "es_analyzers": analyzer_results,
                "norm_matches_expected": normalized.collapsed
                == sample["expected_normalized"].replace(" ", ""),
                "es_norm_matches_python": " ".join(
                    analyzer_results["ko_norm_index_analyzer"]
                )
                == normalized.collapsed,
            }
        )

    return {
        "summary": {
            "sample_count": len(samples),
            "analyzers": list(ANALYZERS),
        },
        "rows": rows,
    }


def build_multi_match_query(text: str, fields: list[str], min_score: float | None = None) -> dict:
    body: dict[str, object] = {
        "size": 3,
        "query": {
            "multi_match": {
                "query": text,
                "fields": fields,
            }
        },
    }
    if min_score is not None:
        body["min_score"] = min_score
    return body


def collect_search_log(client: Elasticsearch, samples: list[dict]) -> dict:
    settings = get_settings()
    lexicon_strategies = {
        "norm_only": ["canonical.norm^3", "variants.norm^2"],
        "ngram_only": ["canonical.ngram^1", "variants.ngram^1"],
        "edge_only": ["canonical.edge^1", "variants.edge^1"],
        "nori_only": ["canonical.nori^1", "variants.nori^1"],
        "norm_ngram": [
            "canonical.norm^3",
            "variants.norm^2",
            "canonical.ngram^1",
            "variants.ngram^1",
        ],
        "full_multi": [
            "canonical.norm^3",
            "variants.norm^2",
            "canonical.ngram^1",
            "variants.ngram^1",
            "canonical.edge^1",
            "variants.edge^1",
            "canonical.nori^0.5",
            "variants.nori^0.5",
        ],
    }
    docs_strategies = {
        "norm_only": ["raw_text.norm^2", "normalized_text^3"],
        "ngram_only": ["raw_text.ngram^1", "normalized_text.ngram^1"],
        "edge_only": ["raw_text.edge^1", "normalized_text.edge^1"],
        "nori_only": ["raw_text.nori^1"],
        "norm_ngram": [
            "raw_text.norm^2",
            "normalized_text^3",
            "raw_text.ngram^1",
            "normalized_text.ngram^1",
        ],
        "full_multi": [
            "raw_text.norm^2",
            "normalized_text^3",
            "raw_text.ngram^1",
            "normalized_text.ngram^1",
            "raw_text.edge^1",
            "normalized_text.edge^1",
            "raw_text.nori^0.5",
        ],
    }

    rows: list[dict] = []
    for sample in samples:
        text = sample["text"]
        row = {
            "id": sample["id"],
            "text": text,
            "expected_detection": sample["expected_detection"],
            "lexicon": {},
            "docs": {},
        }

        for strategy_name, fields in lexicon_strategies.items():
            response = client.search(
                index=settings.profanity_lexicon_index,
                body=build_multi_match_query(text, fields, min_score=0.1),
            )
            hits = response.get("hits", {}).get("hits", [])
            row["lexicon"][strategy_name] = {
                "fields": fields,
                "hit_count": len(hits),
                "top_hits": [
                    {
                        "id": hit.get("_id"),
                        "score": hit.get("_score"),
                        "canonical": hit.get("_source", {}).get("canonical"),
                    }
                    for hit in hits
                ],
            }

        for strategy_name, fields in docs_strategies.items():
            response = client.search(
                index=settings.noisy_text_docs_index,
                body=build_multi_match_query(text, fields, min_score=0.1),
            )
            hits = response.get("hits", {}).get("hits", [])
            row["docs"][strategy_name] = {
                "fields": fields,
                "hit_count": len(hits),
                "top_hits": [
                    {
                        "id": hit.get("_id"),
                        "score": hit.get("_score"),
                        "raw_text": hit.get("_source", {}).get("raw_text"),
                        "expected_label": hit.get("_source", {}).get("expected_label"),
                    }
                    for hit in hits
                ],
            }

        rows.append(row)

    return {
        "summary": {
            "sample_count": len(samples),
            "lexicon_strategies": list(lexicon_strategies),
            "docs_strategies": list(docs_strategies),
        },
        "rows": rows,
    }


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> int:
    args = parse_args()
    client = build_client()

    if not client.ping():
        raise RuntimeError("Elasticsearch is not reachable. Start ES and retry.")

    ensure_indices_exist(client)
    samples = load_input_set(args.input)

    analyze_log = collect_analyze_log(client, samples)
    search_log = collect_search_log(client, samples)

    write_json(args.input_log, samples)
    write_json(args.analyze_output, analyze_log)
    write_json(args.search_output, search_log)

    print(f"wrote input log: {args.input_log}")
    print(f"wrote analyze log: {args.analyze_output}")
    print(f"wrote search log: {args.search_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
