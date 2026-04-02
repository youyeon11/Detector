from app.services.normalizer import (
    NORMALIZER_REPLACEMENTS_PATH,
    QueryNormalizer,
    load_replacement_map,
    normalize_text,
)


def test_normalizer_removes_punctuation_and_builds_collapsed_text() -> None:
    result = normalize_text("씨-발 진짜")

    assert result.raw == "씨-발 진짜"
    assert result.normalized == "씨발 진짜"
    assert result.collapsed == "씨발진짜"
    assert any(token == "collapsed_no_space=씨발진짜" for token in result.tokens_for_debug)


def test_normalizer_replaces_common_evasive_spellings() -> None:
    normalizer = QueryNormalizer()

    result = normalizer.normalize("ㅆ발")

    assert result.replaced == "씨발"
    assert result.normalized == "씨발"
    assert result.collapsed == "씨발"


def test_normalizer_collapses_repeated_characters() -> None:
    normalizer = QueryNormalizer()

    result = normalizer.normalize("시이이발")

    assert result.replaced == "시발"
    assert result.collapsed == "시발"


def test_normalizer_keeps_normal_words_stable() -> None:
    normalizer = QueryNormalizer()

    result = normalizer.normalize("시발점에서 출발")

    assert result.normalized == "시발점에서 출발"
    assert result.collapsed == "시발점에서출발"


def test_normalizer_preserves_non_offensive_word_without_over_replacing() -> None:
    normalizer = QueryNormalizer()

    result = normalizer.normalize("개시 합니다")

    assert result.replaced == "개시 합니다"
    assert result.normalized == "개시 합니다"
    assert result.collapsed == "개시합니다"


def test_default_rule_set_loads_replacement_map_from_json_dataset() -> None:
    replacement_map = load_replacement_map()

    assert NORMALIZER_REPLACEMENTS_PATH.exists()
    assert replacement_map["ㅆ발"] == "씨발"
    assert replacement_map["개색기"] == "개새끼"


def test_normalizer_uses_extended_replacement_dataset() -> None:
    normalizer = QueryNormalizer()

    result = normalizer.normalize("개색기 같은 소리")

    assert result.replaced == "개새끼 같은 소리"
    assert result.normalized == "개새끼 같은 소리"
    assert result.collapsed == "개새끼같은소리"


def test_normalizer_maps_swiibal_style_variant_to_sibal() -> None:
    normalizer = QueryNormalizer()

    result = normalizer.normalize("쉬이발")

    assert result.replaced == "시발"
    assert result.normalized == "시발"
    assert result.collapsed == "시발"


def test_normalizer_maps_sibbal_style_variant_to_sibal() -> None:
    normalizer = QueryNormalizer()

    result = normalizer.normalize("시빨")

    assert result.replaced == "시발"
    assert result.normalized == "시발"
    assert result.collapsed == "시발"
