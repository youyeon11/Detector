from __future__ import annotations

import json
from pathlib import Path

import pytest


pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[2]

KIND_TO_RULE = {
    "숫자 사용": "numeric_substitution",
    "알파벳 사용": "alphabet_substitution",
    "음절 변형": "phonetic_variation",
    "철자변형": "orthographic_variation",
}


def parse_variation_sets() -> dict[str, set[str]]:
    text = (ROOT / "exec" / "variation_sets.md").read_text(encoding="utf-8")
    parsed: dict[str, set[str]] = {}
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped.startswith("|") or ":---" in stripped:
            continue
        parts = [part.strip() for part in stripped.strip("|").split("|")]
        if len(parts) != 3 or parts[0] not in KIND_TO_RULE:
            continue
        parsed[KIND_TO_RULE[parts[0]]] = {
            item.strip().rstrip(".")
            for item in parts[2].split(",")
            if item.strip().rstrip(".")
        }
    return parsed


def load_sibal_rule() -> dict[str, list[str]]:
    payload = json.loads((ROOT / "dataset" / "variation_rules.json").read_text(encoding="utf-8"))
    for rule in payload["rules"]:
        if rule["canonical"] == "시발":
            return rule["variant_rules"]
    raise AssertionError("시발 rule not found")


def load_sibal_lexicon_variants() -> set[str]:
    for line in (ROOT / "dataset" / "profanity_lexicon.jsonl").read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if row["canonical"] == "시발":
            return set(row["variants"])
    raise AssertionError("시발 lexicon row not found")


def test_sibal_variation_sets_are_all_reflected_in_v4_rules() -> None:
    expected = parse_variation_sets()
    actual = load_sibal_rule()

    for rule_name, expected_items in expected.items():
        assert expected_items <= set(actual[rule_name])


def test_sibal_variation_sets_are_all_reflected_in_lexicon_variants() -> None:
    expected = parse_variation_sets()
    actual = load_sibal_lexicon_variants()

    all_expected = set().union(*expected.values())

    assert all_expected <= actual
