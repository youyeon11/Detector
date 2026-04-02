from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = Field(default="Detector")
    app_env: str = Field(default="local")
    app_debug: bool = Field(default=True)

    elasticsearch_url: str = Field(default="http://localhost:9200")
    elasticsearch_username: str | None = Field(default=None)
    elasticsearch_password: str | None = Field(default=None)
    profanity_lexicon_index: str = Field(default="profanity_lexicon")
    noisy_text_docs_index: str = Field(default="noisy_text_docs")
    variation_detected_messages_index: str = Field(default="variation_detected_messages")
    elasticsearch_request_timeout: float = Field(default=3.0)

    redis_url: str = Field(default="redis://localhost:6379/0")
    query_cache_ttl_seconds: int = Field(default=60)

    term_boost: float = Field(default=10.0)
    norm_boost: float = Field(default=8.0)
    ngram_boost: float = Field(default=4.0)
    edge_boost: float = Field(default=2.0)
    nori_boost: float = Field(default=0.5)
    fuzzy_boost: float = Field(default=3.0)
    min_score: float = Field(default=0.5)
    review_threshold: float = Field(default=0.5)
    block_threshold: float = Field(default=0.85)
    minimum_should_match: str = Field(default="70%")
    prefix_length: int = Field(default=1)

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @property
    def v4_detected_messages_index(self) -> str:
        return self.variation_detected_messages_index


@lru_cache
def get_settings() -> Settings:
    return Settings()
