from __future__ import annotations

from typing import Any

from elasticsearch import Elasticsearch

from app.config import Settings, get_settings


class ElasticClientError(RuntimeError):
    """Raised when Elasticsearch operations fail."""


class ElasticClient:
    def __init__(
        self,
        *,
        settings: Settings | None = None,
        client: Elasticsearch | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.host = self.settings.elasticsearch_url
        self.request_timeout = self.settings.elasticsearch_request_timeout
        self.profanity_lexicon_index = self.settings.profanity_lexicon_index
        self.noisy_text_docs_index = self.settings.noisy_text_docs_index

        if client is not None:
            self._client = client
            return

        basic_auth: tuple[str, str] | None = None
        if self.settings.elasticsearch_username and self.settings.elasticsearch_password:
            basic_auth = (
                self.settings.elasticsearch_username,
                self.settings.elasticsearch_password,
            )

        self._client = Elasticsearch(
            hosts=[self.host],
            basic_auth=basic_auth,
            request_timeout=self.request_timeout,
        )

    def ping(self) -> bool:
        try:
            return bool(self._client.ping())
        except Exception as exc:  # pragma: no cover - external client exception detail
            raise ElasticClientError("Failed to ping Elasticsearch.") from exc

    def health(self) -> dict[str, Any]:
        try:
            return self._client.cluster.health()
        except Exception as exc:  # pragma: no cover - external client exception detail
            raise ElasticClientError("Failed to fetch Elasticsearch health.") from exc

    def search(
        self,
        *,
        index: str,
        query: dict[str, Any] | None = None,
        body: dict[str, Any] | None = None,
        size: int = 10,
        timeout: float | None = None,
        min_score: float | None = None,
        source_includes: list[str] | None = None,
    ) -> dict[str, Any]:
        if query is None and body is None:
            raise ValueError("Either `query` or `body` must be provided.")
        if query is not None and body is not None:
            raise ValueError("Provide either `query` or `body`, not both.")

        request_timeout = timeout if timeout is not None else self.request_timeout
        kwargs: dict[str, Any] = {"index": index, "request_timeout": request_timeout}

        if body is not None:
            kwargs["body"] = body
        else:
            kwargs["query"] = query
            kwargs["size"] = size
            if min_score is not None:
                kwargs["min_score"] = min_score
            if source_includes is not None:
                kwargs["_source_includes"] = source_includes

        try:
            return self._client.search(**kwargs)
        except Exception as exc:  # pragma: no cover - external client exception detail
            raise ElasticClientError("Elasticsearch search failed.") from exc
