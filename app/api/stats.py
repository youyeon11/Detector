from __future__ import annotations

from fastapi import APIRouter, Depends

from app.config import Settings, get_settings
from app.schemas.response import (
    VariationStatsBucketResponse,
    VariationStatsResponse,
    VariationStatsSummaryResponse,
)
from app.services.elastic_client import ElasticClient
from scripts.variation_query_examples import build_report


router = APIRouter(tags=["stats"])


def get_elastic_client(settings: Settings = Depends(get_settings)) -> ElasticClient:
    return ElasticClient(settings=settings)


@router.get("/stats/v4", include_in_schema=False)
@router.get(
    "/stats/variation",
    response_model=VariationStatsResponse,
    summary="Read variation detection statistics",
)
def read_variation_stats(
    client: ElasticClient = Depends(get_elastic_client),
    settings: Settings = Depends(get_settings),
) -> VariationStatsResponse:
    report = build_report(client, settings.variation_detected_messages_index)
    results = report["results"]
    detected_total = results["detected_hits_total"]
    return VariationStatsResponse(
        index=report["index"],
        detected_hits_total=VariationStatsSummaryResponse(
            value=detected_total.get("value"),
            relation=detected_total.get("relation"),
        ),
        top_canonical_buckets=[
            VariationStatsBucketResponse(key=bucket["key"], doc_count=bucket["doc_count"])
            for bucket in results["top_canonical_buckets"]
        ],
        variation_type_buckets=[
            VariationStatsBucketResponse(key=bucket["key"], doc_count=bucket["doc_count"])
            for bucket in results["variation_type_buckets"]
        ],
        severity_histogram_buckets=[
            VariationStatsBucketResponse(key=bucket["key"], doc_count=bucket["doc_count"])
            for bucket in results["severity_histogram_buckets"]
        ],
    )
