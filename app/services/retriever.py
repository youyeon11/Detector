from __future__ import annotations

from typing import Any

from app.config import Settings, get_settings
from app.domain.models import DocumentHit, LexiconHit
from app.services.elastic_client import ElasticClient, ElasticClientError


class RetrieverError(RuntimeError):
    """Raised when retrieval cannot be completed."""


class LexiconRetriever:
    def __init__(
        self,
        *,
        elastic_client: ElasticClient | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.elastic_client = elastic_client or ElasticClient(settings=self.settings)
        self.default_size = 20
        self.default_timeout = self.settings.elasticsearch_request_timeout
        self.default_min_score = self.settings.min_score
        self.default_source_includes = [
            "canonical",
            "variants",
            "category",
            "severity",
            "risk_score",
        ]

    def retrieve_lexicon_hits(
        self,
        *,
        query: dict[str, Any] | None = None,
        search_body: dict[str, Any] | None = None,
        size: int | None = None,
        timeout: float | None = None,
        min_score: float | None = None,
        source_includes: list[str] | None = None,
    ) -> list[LexiconHit]:
        final_size = size if size is not None else self.default_size
        final_timeout = timeout if timeout is not None else self.default_timeout
        final_min_score = min_score if min_score is not None else self.default_min_score
        final_source = source_includes if source_includes is not None else self.default_source_includes

        try:
            response = self._search(
                query=query,
                search_body=search_body,
                size=final_size,
                timeout=final_timeout,
                min_score=final_min_score,
                source_includes=final_source,
            )
        except ElasticClientError as exc:
            raise RetrieverError("Failed to retrieve lexicon hits from Elasticsearch.") from exc

        raw_hits = response.get("hits", {}).get("hits", [])
        if not isinstance(raw_hits, list):
            return []

        mapped_hits: list[LexiconHit] = []
        for raw_hit in raw_hits:
            mapped = self._map_raw_hit(raw_hit)
            if mapped is not None:
                mapped_hits.append(mapped)
        return mapped_hits

    def _search(
        self,
        *,
        query: dict[str, Any] | None,
        search_body: dict[str, Any] | None,
        size: int,
        timeout: float,
        min_score: float,
        source_includes: list[str],
    ) -> dict[str, Any]:
        if search_body is not None:
            body = dict(search_body)
            body.setdefault("size", size)
            body.setdefault("min_score", min_score)
            if "_source" not in body and "_source_includes" not in body:
                body["_source"] = source_includes
            return self.elastic_client.search(
                index=self.settings.profanity_lexicon_index,
                body=body,
                timeout=timeout,
            )

        if query is None:
            raise ValueError("Either `query` or `search_body` must be provided.")

        return self.elastic_client.search(
            index=self.settings.profanity_lexicon_index,
            query=query,
            size=size,
            timeout=timeout,
            min_score=min_score,
            source_includes=source_includes,
        )

    def ping(self) -> bool:
        try:
            return self.elastic_client.ping()
        except ElasticClientError as exc:
            raise RetrieverError("Elasticsearch ping failed during retrieval health check.") from exc

    def health(self) -> dict[str, Any]:
        try:
            return self.elastic_client.health()
        except ElasticClientError as exc:
            raise RetrieverError("Elasticsearch health check failed during retrieval.") from exc

    def _map_raw_hit(self, raw_hit: Any) -> LexiconHit | None:
        if not isinstance(raw_hit, dict):
            return None

        source = raw_hit.get("_source")
        if not isinstance(source, dict):
            return None

        canonical = source.get("canonical")
        if not isinstance(canonical, str) or not canonical.strip():
            return None

        variants_value = source.get("variants", [])
        variants = self._coerce_variants(variants_value, canonical)
        category = source.get("category")
        severity = source.get("severity")
        risk_score = source.get("risk_score")

        return LexiconHit(
            canonical=canonical,
            variants=variants,
            category=category if isinstance(category, str) else "unknown",
            severity=int(severity) if isinstance(severity, (int, float)) else 0,
            risk_score=float(risk_score) if isinstance(risk_score, (int, float)) else 0.0,
            es_score=float(raw_hit.get("_score", 0.0)),
        )

    def _coerce_variants(self, value: Any, canonical: str) -> list[str]:
        if isinstance(value, list):
            variants = [item for item in value if isinstance(item, str) and item.strip()]
            if variants:
                return variants
        if isinstance(value, str) and value.strip():
            return [value]
        return [canonical]


class DocumentRetriever:
    def __init__(
        self,
        *,
        elastic_client: ElasticClient | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.elastic_client = elastic_client or ElasticClient(settings=self.settings)
        self.default_size = 20
        self.default_timeout = self.settings.elasticsearch_request_timeout
        self.default_min_score = self.settings.min_score
        self.default_source_includes = [
            "raw_text",
            "normalized_text",
            "collapsed_text",
            "replaced_text",
            "expected_label",
            "source",
            "notes",
            "tokens_for_debug",
        ]

    def retrieve_document_hits(
        self,
        *,
        query: dict[str, Any] | None = None,
        search_body: dict[str, Any] | None = None,
        size: int | None = None,
        timeout: float | None = None,
        min_score: float | None = None,
        source_includes: list[str] | None = None,
    ) -> list[DocumentHit]:
        final_size = size if size is not None else self.default_size
        final_timeout = timeout if timeout is not None else self.default_timeout
        final_min_score = min_score if min_score is not None else self.default_min_score
        final_source = source_includes if source_includes is not None else self.default_source_includes

        try:
            if search_body is not None:
                body = dict(search_body)
                body.setdefault("size", final_size)
                body.setdefault("min_score", final_min_score)
                if "_source" not in body and "_source_includes" not in body:
                    body["_source"] = final_source
                response = self.elastic_client.search(
                    index=self.settings.noisy_text_docs_index,
                    body=body,
                    timeout=final_timeout,
                )
            else:
                if query is None:
                    raise ValueError("Either `query` or `search_body` must be provided.")
                response = self.elastic_client.search(
                    index=self.settings.noisy_text_docs_index,
                    query=query,
                    size=final_size,
                    timeout=final_timeout,
                    min_score=final_min_score,
                    source_includes=final_source,
                )
        except ElasticClientError as exc:
            raise RetrieverError("Failed to retrieve document hits from Elasticsearch.") from exc

        raw_hits = response.get("hits", {}).get("hits", [])
        if not isinstance(raw_hits, list):
            return []

        mapped_hits: list[DocumentHit] = []
        for raw_hit in raw_hits:
            mapped = self._map_raw_hit(raw_hit)
            if mapped is not None:
                mapped_hits.append(mapped)
        return mapped_hits

    def ping(self) -> bool:
        try:
            return self.elastic_client.ping()
        except ElasticClientError as exc:
            raise RetrieverError("Elasticsearch ping failed during document retrieval health check.") from exc

    def health(self) -> dict[str, Any]:
        try:
            return self.elastic_client.health()
        except ElasticClientError as exc:
            raise RetrieverError("Elasticsearch health check failed during document retrieval.") from exc

    def _map_raw_hit(self, raw_hit: Any) -> DocumentHit | None:
        if not isinstance(raw_hit, dict):
            return None

        source = raw_hit.get("_source")
        if not isinstance(source, dict):
            return None

        raw_text = source.get("raw_text")
        if not isinstance(raw_text, str) or not raw_text.strip():
            return None

        normalized_text = source.get("normalized_text")
        collapsed_text = source.get("collapsed_text")
        replaced_text = source.get("replaced_text")
        expected_label = source.get("expected_label")
        origin = source.get("source")
        notes = source.get("notes")
        tokens_for_debug = source.get("tokens_for_debug")

        return DocumentHit(
            raw_text=raw_text,
            normalized_text=normalized_text if isinstance(normalized_text, str) else raw_text,
            collapsed_text=collapsed_text if isinstance(collapsed_text, str) else raw_text.replace(" ", ""),
            replaced_text=replaced_text if isinstance(replaced_text, str) else raw_text,
            expected_label=expected_label if isinstance(expected_label, str) else "UNKNOWN",
            source=origin if isinstance(origin, str) else "unknown",
            notes=notes if isinstance(notes, str) else "",
            tokens_for_debug=(
                [item for item in tokens_for_debug if isinstance(item, str)]
                if isinstance(tokens_for_debug, list)
                else []
            ),
            es_score=float(raw_hit.get("_score", 0.0)),
        )
