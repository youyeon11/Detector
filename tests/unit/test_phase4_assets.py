import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]

# ES 없이 josn 파일 구조만 검증
def load_json(relative_path: str) -> dict:
    """
    프로젝트 json 파일 로드 유틸리티
    """
    return json.loads((ROOT / relative_path).read_text(encoding="utf-8"))


def test_lexicon_index_uses_phase4_analyzer_names() -> None:
    """
    setting 제대로 설정되어있는지 확인
    profanity_lexicon_index에 analyzer와 char_filter 선언 확인
    """ 
    body = load_json("elastic/profanity_lexicon_index.json")
    analysis = body["settings"]["analysis"]

    assert set(analysis["analyzer"]) >= {
        "ko_norm_index_analyzer",
        "ko_norm_search_analyzer",
        "ko_ngram_index_analyzer",
        "ko_ngram_search_analyzer",
        "ko_edge_index_analyzer",
        "ko_edge_search_analyzer",
        "ko_nori_analyzer",
    }
    assert "ko_noise_map" in analysis["char_filter"]
    assert "remove_spaces" in analysis["char_filter"]


def test_lexicon_index_fields_reference_expected_analyzers() -> None:
    """
    profanity_lexicon_index의 canonical·variants 필드 각각에 대해
    norm/ngram/edge/nori multi-field가 analyzer 이름을
    index_analyzer·search_analyzer 모두 올바르게 참조하고 있는지 검증
    """
    body = load_json("elastic/profanity_lexicon_index.json")
    properties = body["mappings"]["properties"]

    for field_name in ("canonical", "variants"):
        fields = properties[field_name]["fields"]
        assert fields["norm"]["analyzer"] == "ko_norm_index_analyzer"
        assert fields["norm"]["search_analyzer"] == "ko_norm_search_analyzer"
        assert fields["ngram"]["analyzer"] == "ko_ngram_index_analyzer"
        assert fields["ngram"]["search_analyzer"] == "ko_ngram_search_analyzer"
        assert fields["edge"]["analyzer"] == "ko_edge_index_analyzer"
        assert fields["edge"]["search_analyzer"] == "ko_edge_search_analyzer"
        assert fields["nori"]["analyzer"] == "ko_nori_analyzer"


def test_docs_index_preserves_norm_ngram_edge_nori_multi_fields() -> None:
    """
    noisy_text_docs_index의 raw_text 필드에 norm/ngram/edge/nori multi-field 유지
    normalized_text 필드가 ko_norm analyzer를 index·search 양쪽에 올바르게 지정하고 있는지 검증
    """
    body = load_json("elastic/noisy_text_docs_index.json")
    properties = body["mappings"]["properties"]
    raw_text_fields = properties["raw_text"]["fields"]

    assert raw_text_fields["norm"]["analyzer"] == "ko_norm_index_analyzer"
    assert raw_text_fields["ngram"]["analyzer"] == "ko_ngram_index_analyzer"
    assert raw_text_fields["edge"]["analyzer"] == "ko_edge_index_analyzer"
    assert raw_text_fields["nori"]["analyzer"] == "ko_nori_analyzer"

    normalized_text = properties["normalized_text"]
    assert normalized_text["analyzer"] == "ko_norm_index_analyzer"
    assert normalized_text["search_analyzer"] == "ko_norm_search_analyzer"


def test_phase4_noise_map_contains_expected_normalization_rules() -> None:
    body = load_json("elastic/profanity_lexicon_index.json")
    mappings = set(body["settings"]["analysis"]["char_filter"]["ko_noise_map"]["mappings"])

    assert "ㅆ발 => 씨발" in mappings
    assert "씌발 => 씨발" in mappings
    assert "시이발 => 시발" in mappings
