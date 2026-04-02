from pathlib import Path

import pytest

from scripts.validate_v4_eval_messages import (
    REQUIRED_CASE_GROUPS,
    iter_rows,
    load_canonicals,
    validate_rows,
)


pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[2]


def test_variation_eval_messages_have_required_case_group_coverage() -> None:
    rows = iter_rows(ROOT / "dataset" / "variation_eval_messages.jsonl")

    case_groups = {row["case_group"] for row in rows}

    assert REQUIRED_CASE_GROUPS <= case_groups


def test_variation_eval_messages_validate_against_schema_and_canonicals() -> None:
    rows = iter_rows(ROOT / "dataset" / "variation_eval_messages.jsonl")
    canonicals = load_canonicals(ROOT / "dataset" / "variation_canonical_dictionary.jsonl")

    errors = validate_rows(rows, canonicals)

    assert errors == []


def test_variation_eval_messages_include_safe_and_detected_rows() -> None:
    rows = iter_rows(ROOT / "dataset" / "variation_eval_messages.jsonl")

    assert any(row["expected_detected"] is False for row in rows)
    assert any(row["expected_detected"] is True for row in rows)
    assert any(row["case_group"] == "meta_safe" for row in rows)
