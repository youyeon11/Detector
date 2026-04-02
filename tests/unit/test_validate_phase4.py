import importlib.util
from pathlib import Path

import pytest

from app.config import Settings
from app.domain.models import NormalizedQuery


ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = ROOT / "scripts" / "validate_phase4.py"

spec = importlib.util.spec_from_file_location("validate_phase4", MODULE_PATH)
assert spec is not None
assert spec.loader is not None
validate_phase4 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(validate_phase4)


class FakeIndicesClient:
    """
    ES IndicesClient의 최소한의 기능을 모방하는 클래스
    analyze_calls에 호출 이력을 누적해 '몇 번 호출됐는지'를 검증

    tokens_by_analyzer는 analyzer별·입력 텍스트별로 예상 토큰 결과를 정의
    각 analyzer의 특성 반영:
      - ko_norm_index_analyzer : 노이즈 정규화 후 단일 토큰
      - ko_ngram_index_analyzer: n-gram 분절, 정규화 미적용(그대로 토큰화함)
      - ko_nori_analyzer       : 형태소 분석, 공백 분리(ex.두 개의 토큰으로 나뉨)
    """
    def __init__(self) -> None:
        self.existing_indices = {"profanity_lexicon", "noisy_text_docs"}
        self.analyze_calls: list[dict] = []

    def analyze(self, *, index: str, body: dict) -> dict:
        self.analyze_calls.append({"index": index, "body": body})

        analyzer = body["analyzer"]
        text = body["text"]
        tokens_by_analyzer = {
            "ko_norm_index_analyzer": {
                "씨발": ["씨발"],
                "씨 발": ["씨발"],
                "ㅆ발": ["씨발"],
                "씌발": ["씨발"],
                "시이발": ["시발"],
            },
            "ko_ngram_index_analyzer": {
                "씨발": ["씨발"],
                "씨 발": ["씨발"],
                "ㅆ발": ["씨발"],
                "씌발": ["씌발"],
                "시이발": ["시이", "이발"],
            },
            "ko_nori_analyzer": {
                "씨발": ["씨발"],
                "씨 발": ["씨", "발"],
                "ㅆ발": ["ㅆ발"],
                "씌발": ["씌발"],
                "시이발": ["시이발"],
            },
        }
        return {
            "tokens": [
                {"token": token}
                for token in tokens_by_analyzer[analyzer][text]
            ]
        }

    def exists(self, *, index: str) -> bool:
        return index in self.existing_indices


class FakeElasticsearchClient:
    """
    ES Client 전체의 테스트 대역

    search 결과를 인덱스별로 분기해 반환
    - profanity_lexicon : 사전 인덱스 검색 결과 (canonical 필드 포함)
    - noisy_text_docs   : 문서 인덱스 검색 결과 (raw_text 필드 포함)
    
    search_calls에 호출 이력을 누적해 호출 횟수·파라미터를 검증 가능
    """
    def __init__(self) -> None:
        self.indices = FakeIndicesClient()
        self.search_calls: list[dict] = []

    def search(self, *, index: str, body: dict) -> dict:
        self.search_calls.append({"index": index, "body": body})

        if index == "profanity_lexicon":
            return {
                "hits": {
                    "hits": [
                        {
                            "_id": "lex-1",
                            "_score": 3.2,
                            "_source": {"canonical": "씨발"},
                        }
                    ]
                }
            }

        return {
            "hits": {
                "hits": [
                    {
                        "_id": "doc-1",
                        "_score": 1.7,
                        "_source": {"raw_text": "씨 발 진짜"},
                    }
                ]
            }
        }


@pytest.fixture
def settings() -> Settings:
    return Settings(
        profanity_lexicon_index="profanity_lexicon",
        noisy_text_docs_index="noisy_text_docs",
    )


@pytest.fixture
def fake_client() -> FakeElasticsearchClient:
    return FakeElasticsearchClient()


def test_format_hits_returns_compact_rows() -> None:
    """
    format_hits가 ES 검색 응답을 compact한 형태로 변환했는지 검증
    """
    response = {
        "hits": {
            "hits": [
                {"_id": "1", "_score": 2.5, "_source": {"canonical": "씨발"}},
                {"_id": "2", "_score": 1.0},
            ]
        }
    }

    rows = validate_phase4.format_hits(response)

    assert rows == [
        {"id": "1", "score": 2.5, "source": {"canonical": "씨발"}},
        {"id": "2", "score": 1.0, "source": {}},
    ]


def test_ensure_indices_exist_raises_for_missing_index(
    monkeypatch: pytest.MonkeyPatch,
    fake_client: FakeElasticsearchClient,
    settings: Settings,
) -> None:
    """
    존재하지 않는 인덱스가 있는 경우 RuntimeError를 발생
    """
    fake_client.indices.existing_indices = {"profanity_lexicon"}
    monkeypatch.setattr(validate_phase4, "get_settings", lambda: settings)

    with pytest.raises(RuntimeError, match="Missing indices: noisy_text_docs"):
        validate_phase4.ensure_indices_exist(fake_client)


def test_build_report_collects_analyze_and_search_results(
    monkeypatch: pytest.MonkeyPatch,
    fake_client: FakeElasticsearchClient,
    settings: Settings,
) -> None:
    """
    리포트가 analyze 결과와 search 결과를 올바르게 수집하는지 검증
    """
    monkeypatch.setattr(validate_phase4, "get_settings", lambda: settings)

    report = validate_phase4.build_report(fake_client)

    assert report["indices"] == {
        "lexicon": "profanity_lexicon",
        "docs": "noisy_text_docs",
    }
    assert len(report["analyze_results"]) == 5  # 테스트용 입력 5개에 대해 analyze 호출
    assert report["analyze_results"][0]["ko_norm_index_analyzer"] == ["씨발"] # 첫번째 결과 확인 (ko_norm_index_anlyzer 확인)
    assert report["search_results"]["lexicon_query_field"] == "canonical.ngram" # search_results에서 lexicon_query_field 확인(canonical.ngram, raw_text.ngram과 각 top hit의 id 확인)
    assert report["search_results"]["docs_query_field"] == "raw_text.ngram"
    assert report["search_results"]["lexicon_top_hits"][0]["id"] == "lex-1" # lexicon 1회 호출
    assert report["search_results"]["docs_top_hits"][0]["id"] == "doc-1" # docs 1회 호출
    assert len(fake_client.indices.analyze_calls) == 15
    assert len(fake_client.search_calls) == 2


def test_build_report_compares_python_and_es_normalization(
    monkeypatch: pytest.MonkeyPatch,
    fake_client: FakeElasticsearchClient,
    settings: Settings,
) -> None:
    """
    Python과 Elasticsearch 정규화 결과를 비교하는지 검증
    """
    monkeypatch.setattr(validate_phase4, "get_settings", lambda: settings)

    def fake_normalize_text(text: str) -> NormalizedQuery:
        normalized = "씨발" if text in {"씨발", "씨 발", "ㅆ발", "씌발"} else "시발"
        return NormalizedQuery(
            raw=text,
            normalized=normalized,
            collapsed=normalized,
            replaced=normalized,
            tokens_for_debug=[],
        )

    monkeypatch.setattr(validate_phase4, "normalize_text", fake_normalize_text)

    report = validate_phase4.build_report(fake_client)
    comparison = {row["text"]: row for row in report["python_vs_es_normalization"]}

    assert comparison["씨발"]["is_same_as_collapsed"] is True
    assert comparison["씨 발"]["es_norm_analyzer_joined"] == "씨발"
    assert comparison["시이발"]["python_collapsed"] == "시발"
    assert comparison["시이발"]["is_same_as_collapsed"] is True
