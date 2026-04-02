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


def load_index_body(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def create_index(client: Elasticsearch, index_name: str, body: dict, *, recreate: bool) -> None:
    if client.indices.exists(index=index_name):
        if not recreate:
            print(f"skip existing index: {index_name}")
            return
        client.indices.delete(index=index_name)
        print(f"deleted index: {index_name}")

    client.indices.create(index=index_name, **body)
    print(f"created index: {index_name}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create Elasticsearch indices for the variation detection pipeline.")
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="Delete the index first when it already exists.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    settings = get_settings()
    client = build_client()

    index_specs = [
        (
            settings.variation_detected_messages_index,
            ROOT / "elastic" / "variation_detected_messages_index.json",
        ),
    ]

    for index_name, path in index_specs:
        body = load_index_body(path)
        create_index(client, index_name, body, recreate=args.recreate)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
