from __future__ import annotations

from fastapi import APIRouter, Depends

from app.config import Settings, get_settings
from app.schemas.response import (
    V4StatsBucketResponse,
    V4StatsResponse,
    V4StatsSummaryResponse,
)
from app.services.elastic_client import ElasticClient
from scripts.v4_query_examples import build_report


router = APIRouter(tags=["stats"])


def get_elastic_client(settings: Settings = Depends(get_settings)) -> ElasticClient:
    return ElasticClient(settings=settings)


@router.get(
    "/stats/v4",
    response_model=V4StatsResponse,
    summary="Read V4 profanity statistics",
)
def read_v4_stats(
    client: ElasticClient = Depends(get_elastic_client),
    settings: Settings = Depends(get_settings),
) -> V4StatsResponse:
    report = build_report(client, settings.v4_detected_messages_index)
    results = report["results"]
    detected_total = results["detected_hits_total"]
    return V4StatsResponse(
        index=report["index"],
        detected_hits_total=V4StatsSummaryResponse(
            value=detected_total.get("value"),
            relation=detected_total.get("relation"),
        ),
        top_canonical_buckets=[
            V4StatsBucketResponse(key=bucket["key"], doc_count=bucket["doc_count"])
            for bucket in results["top_canonical_buckets"]
        ],
        variation_type_buckets=[
            V4StatsBucketResponse(key=bucket["key"], doc_count=bucket["doc_count"])
            for bucket in results["variation_type_buckets"]
        ],
        severity_histogram_buckets=[
            V4StatsBucketResponse(key=bucket["key"], doc_count=bucket["doc_count"])
            for bucket in results["severity_histogram_buckets"]
        ],
    )
