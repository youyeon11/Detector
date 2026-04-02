from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from app.domain.enums import DetectionLabel
from app.domain.models import DetectionResult, NormalizedQuery, VariationDetectionDocument, VariationHit
from app.domain.variation_types import VariationType
from app.main import create_app
from app.services.variation_detection import (
    VariationDetectionExecutionResult,
    get_variation_detection_service,
)

# 테스트용 fixture 파일이 위치한 루트 디렉토리
FIXTURE_ROOT = Path(__file__).resolve().parent / "fixtures"


class FakeDetectorService:
    """
    실제 탐지 서비스 대신 사용하는 테스트용 가짜 서비스.
    항상 동일한 결과를 반환하여 테스트를 안정적으로 수행할 수 있게 한다.
    """

    def __init__(self) -> None:
        # 텍스트 탐지 결과
        self.detect_result = DetectionResult(
            label=DetectionLabel.BLOCK,
            score=0.93,
            matched_term="씨발",
            reasons=["정확히 접힌 문자열 매칭", "계산된 점수=9.3"],
            normalized_text="씨발",
        )

        # 텍스트 분석 결과
        self.analyze_result = (
            NormalizedQuery(
                raw="시발",
                normalized="씨발",
                collapsed="씨발",
                replaced="씨발",
                tokens_for_debug=["원본=시발", "치환=씨발"],
            ),
            {
                "lexicon_query": {
                    "query": {
                        "bool": {
                            "should": [
                                {
                                    "match": {
                                        "canonical.ngram": {
                                            "query": "씨발",
                                            "boost": 4.0,
                                        }
                                    }
                                }
                            ]
                        }
                    }
                },
                "document_query": {},
            },
        )

        # 문서 단위 욕설 탐지 결과
        self.detect_document_result = VariationDetectionDocument(
            message="시발",
            message_normalized="씨발",
            profanity_detected=True,
            profanity_hits=[
                VariationHit(
                    canonical="씨발",
                    matched_variant="시발",
                    variation_type=VariationType.PHONETIC_VARIATION,
                    severity=5,
                    risk_score=0.99,
                    label=DetectionLabel.BLOCK,
                    score=0.93,
                    reasons=["기준 단어 매칭=씨발"],
                )
            ],
        )

    # 단일 텍스트 욕설 탐지
    def detect(self, text: str) -> DetectionResult:
        assert isinstance(text, str)
        return self.detect_result

    # 텍스트 분석 (정규화 + 검색 쿼리 생성)
    def analyze(self, text: str) -> tuple[NormalizedQuery, dict[str, Any]]:
        assert isinstance(text, str)
        return self.analyze_result

    # 문서 단위 욕설 탐지
    def detect_document(self, text: str) -> VariationDetectionDocument:
        assert isinstance(text, str)
        return self.detect_document_result

    # 탐지 결과 + 문서 분석 결과를 함께 반환
    def detect_with_document(self, text: str) -> VariationDetectionExecutionResult:
        assert isinstance(text, str)
        return VariationDetectionExecutionResult(
            detection_result=self.detect_result,
            document_result=self.detect_document_result,
        )

    # 사전 조회 기능을 동일 객체로 처리
    @property
    def lexicon_retriever(self) -> "FakeDetectorService":
        return self

    # 서비스 상태 확인
    def ping(self) -> bool:
        return True


# 테스트용 JSON 데이터를 로드하는 함수
def load_test_fixture(name: str) -> dict[str, Any]:
    path = FIXTURE_ROOT / name
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


# detector_cases.json 테스트 데이터를 세션 단위로 로드
@pytest.fixture(scope="session")
def detector_cases() -> dict[str, Any]:
    return load_test_fixture("detector_cases.json")


# 가짜 탐지 서비스 fixture
@pytest.fixture
def fake_detector_service() -> FakeDetectorService:
    return FakeDetectorService()


# FastAPI 테스트용 클라이언트 생성
@pytest.fixture
def unit_client(
    fake_detector_service: FakeDetectorService,
    monkeypatch: pytest.MonkeyPatch,
) -> TestClient:
    # 실제 DetectorService 대신 FakeDetectorService를 사용하도록 교체
    monkeypatch.setattr("app.main.DetectorService", lambda: fake_detector_service)

    app = create_app()

    # FastAPI 의존성 주입을 가짜 서비스로 덮어쓰기
    app.dependency_overrides[get_variation_detection_service] = lambda: fake_detector_service

    return TestClient(app)


# pytest 실행 시 테스트 경로에 따라 자동으로 마커(unit / integration)를 부여
def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    for item in items:
        path = str(item.fspath)

        # unit 테스트
        if "\\tests\\unit\\" in path:
            item.add_marker(pytest.mark.unit)

        # integration 테스트
        elif "\\tests\\integration\\" in path:
            item.add_marker(pytest.mark.integration)