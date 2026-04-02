from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.config import get_settings
from scripts.collect_es_search_logs import load_input_set
from scripts.load_documents import build_actions as build_doc_actions
from scripts.load_documents import iter_jsonl as iter_doc_jsonl
from scripts.load_lexicon import build_actions as build_lexicon_actions
from scripts.load_lexicon import iter_jsonl as iter_lexicon_jsonl


GRAM_VALUES = (1, 2, 3, 4)


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
        description="Collect Elasticsearch search n-gram comparison logs."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=ROOT / "dataset" / "es_search_input_set.json",
        help="ES search input set JSON path.",
    )
    parser.add_argument(
        "--lexicon-path",
        type=Path,
        default=ROOT / "dataset" / "profanity_lexicon.jsonl",
        help="Lexicon dataset path.",
    )
    parser.add_argument(
        "--docs-path",
        type=Path,
        default=ROOT / "dataset" / "eval_sentences.jsonl",
        help="Docs dataset path.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "logs" / "es_search_ngram_compare.json",
        help="Output path for n-gram comparison log.",
    )
    parser.add_argument(
        "--keep-indices",
        action="store_true",
        help="Keep temporary indices instead of deleting them after collection.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def build_index_body(template_path: Path, *, min_gram: int) -> dict:
    body = copy.deepcopy(load_json(template_path))
    analysis = body["settings"]["analysis"]
    max_gram = min_gram
    analysis["tokenizer"]["ko_ngram_tokenizer"]["min_gram"] = min_gram
    analysis["tokenizer"]["ko_ngram_tokenizer"]["max_gram"] = max_gram
    analysis["filter"]["ko_length_2_30"]["min"] = min_gram
    return body


def recreate_index(client: Elasticsearch, index_name: str, body: dict) -> None:
    if client.indices.exists(index=index_name):
        client.indices.delete(index=index_name)
    client.indices.create(index=index_name, **body)


def load_temp_data(
    client: Elasticsearch,
    *,
    lexicon_index: str,
    docs_index: str,
    lexicon_path: Path,
    docs_path: Path,
) -> None:
    lexicon_rows = iter_lexicon_jsonl(lexicon_path)
    docs_rows = iter_doc_jsonl(docs_path)
    lexicon_actions = build_lexicon_actions(lexicon_index, lexicon_rows)
    docs_actions = build_doc_actions(docs_index, docs_rows)
    bulk(client, lexicon_actions, refresh="wait_for")
    bulk(client, docs_actions, refresh="wait_for")


def build_multi_match_query(text: str, fields: list[str], *, size: int = 3) -> dict:
    return {
        "size": size,
        "query": {
            "multi_match": {
                "query": text,
                "fields": fields,
            }
        },
    }


def collect_case_results(
    client: Elasticsearch,
    *,
    lexicon_index: str,
    docs_index: str,
    samples: list[dict],
) -> list[dict]:
    rows: list[dict] = []
    lexicon_fields = ["canonical.ngram^1", "variants.ngram^1"]
    docs_fields = ["raw_text.ngram^1", "normalized_text.ngram^1"]

    for sample in samples:
        text = sample["text"]
        lexicon_response = client.search(
            index=lexicon_index,
            body=build_multi_match_query(text, lexicon_fields),
        )
        docs_response = client.search(
            index=docs_index,
            body=build_multi_match_query(text, docs_fields),
        )
        lexicon_hits = lexicon_response.body["hits"]["hits"]
        docs_hits = docs_response.body["hits"]["hits"]

        rows.append(
            {
                "id": sample["id"],
                "text": text,
                "expected_detection": sample["expected_detection"],
                "lexicon_hit_count": len(lexicon_hits),
                "lexicon_top_hits": [
                    {
                        "id": hit["_id"],
                        "score": hit["_score"],
                        "canonical": hit["_source"].get("canonical"),
                    }
                    for hit in lexicon_hits
                ],
                "docs_hit_count": len(docs_hits),
                "docs_top_hits": [
                    {
                        "id": hit["_id"],
                        "score": hit["_score"],
                        "raw_text": hit["_source"].get("raw_text"),
                        "expected_label": hit["_source"].get("expected_label"),
                    }
                    for hit in docs_hits
                ],
                "detected": bool(lexicon_hits or docs_hits),
            }
        )

    return rows


def summarize_case_results(rows: list[dict]) -> dict:
    detect_targets = sum(1 for row in rows if row["expected_detection"] == "detect")
    allow_targets = sum(1 for row in rows if row["expected_detection"] == "allow")
    review_targets = sum(1 for row in rows if row["expected_detection"] == "review")

    detect_hits = sum(
        1
        for row in rows
        if row["expected_detection"] == "detect" and row["detected"]
    )
    allow_false_positives = sum(
        1
        for row in rows
        if row["expected_detection"] == "allow" and row["detected"]
    )
    review_hits = sum(
        1
        for row in rows
        if row["expected_detection"] == "review" and row["detected"]
    )
    avg_lexicon_hits = round(
        sum(row["lexicon_hit_count"] for row in rows) / max(len(rows), 1),
        3,
    )
    avg_docs_hits = round(
        sum(row["docs_hit_count"] for row in rows) / max(len(rows), 1),
        3,
    )

    return {
        "detect_targets": detect_targets,
        "review_targets": review_targets,
        "allow_targets": allow_targets,
        "detect_hits": detect_hits,
        "allow_false_positives": allow_false_positives,
        "review_hits": review_hits,
        "recall_detect_only": round(detect_hits / max(detect_targets, 1), 3),
        "avg_lexicon_hits": avg_lexicon_hits,
        "avg_docs_hits": avg_docs_hits,
    }


def get_index_stats(client: Elasticsearch, index_name: str) -> dict:
    stats = client.indices.stats(index=index_name, metric=["store", "docs"]).body
    index_stats = stats["indices"][index_name]["total"]
    return {
        "docs_count": index_stats["docs"]["count"],
        "store_size_in_bytes": index_stats["store"]["size_in_bytes"],
    }


def collect_token_examples(
    client: Elasticsearch,
    *,
    lexicon_index: str,
    texts: list[str],
) -> list[dict]:
    rows: list[dict] = []
    for text in texts:
        response = client.indices.analyze(
            index=lexicon_index,
            body={"analyzer": "ko_ngram_index_analyzer", "text": text},
        )
        tokens = [token["token"] for token in response.body.get("tokens", [])]
        rows.append(
            {
                "text": text,
                "token_count": len(tokens),
                "tokens": tokens,
            }
        )
    return rows


def delete_index_if_exists(client: Elasticsearch, index_name: str) -> None:
    if client.indices.exists(index=index_name):
        client.indices.delete(index=index_name)


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> int:
    args = parse_args()
    client = build_client()

    if not client.ping():
        raise RuntimeError("Elasticsearch is not reachable. Start ES and retry.")

    samples = load_input_set(args.input)
    template_lexicon = ROOT / "elastic" / "profanity_lexicon_index.json"
    template_docs = ROOT / "elastic" / "noisy_text_docs_index.json"
    token_examples = ["씨발", "씨발놈아", "진짜씨발이야", "발표 잘했어"]

    runs: list[dict] = []

    for min_gram in GRAM_VALUES:
        lexicon_index = f"profanity_lexicon_g{min_gram}"
        docs_index = f"noisy_text_docs_g{min_gram}"

        lexicon_body = build_index_body(template_lexicon, min_gram=min_gram)
        docs_body = build_index_body(template_docs, min_gram=min_gram)

        recreate_index(client, lexicon_index, lexicon_body)
        recreate_index(client, docs_index, docs_body)
        load_temp_data(
            client,
            lexicon_index=lexicon_index,
            docs_index=docs_index,
            lexicon_path=args.lexicon_path,
            docs_path=args.docs_path,
        )

        case_rows = collect_case_results(
            client,
            lexicon_index=lexicon_index,
            docs_index=docs_index,
            samples=samples,
        )
        summary = summarize_case_results(case_rows)
        lexicon_stats = get_index_stats(client, lexicon_index)
        docs_stats = get_index_stats(client, docs_index)
        examples = collect_token_examples(
            client,
            lexicon_index=lexicon_index,
            texts=token_examples,
        )

        runs.append(
            {
                "min_gram": min_gram,
                "max_gram": min_gram,
                "indices": {
                    "lexicon": lexicon_index,
                    "docs": docs_index,
                },
                "summary": summary,
                "lexicon_index_stats": lexicon_stats,
                "docs_index_stats": docs_stats,
                "token_examples": examples,
                "case_rows": case_rows,
            }
        )

        if not args.keep_indices:
            delete_index_if_exists(client, lexicon_index)
            delete_index_if_exists(client, docs_index)

    payload = {
        "input_path": str(args.input),
        "compared_min_grams": list(GRAM_VALUES),
        "runs": runs,
    }
    write_json(args.output, payload)
    print(f"wrote ngram comparison log: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
