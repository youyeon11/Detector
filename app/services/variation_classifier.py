from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

from app.domain.models import VariationClassificationResult
from app.domain.variation_types import VariationType

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RULE_PATH = ROOT / "dataset" / "variation_rules.json"


class VariationClassifier:
    def __init__(self, *, rules_path: Path | None = None) -> None:
        self.rules_path = rules_path or DEFAULT_RULE_PATH
        self.rules = load_variation_rules(self.rules_path)

    def classify(
        self,
        *,
        canonical: str,
        matched_variant: str,
        normalized_variant: str | None = None,
        collapsed_variant: str | None = None,
    ) -> VariationClassificationResult:
        canonical_key = canonical.strip()
        observed = matched_variant.strip()
        normalized = normalized_variant.strip() if isinstance(normalized_variant, str) else observed
        collapsed = collapsed_variant.strip() if isinstance(collapsed_variant, str) else self._collapse(normalized)

        if canonical_key not in self.rules:
            return VariationClassificationResult(
                canonical=canonical_key,
                matched_variant=observed,
                variation_type=VariationType.MIXED_VARIATION,
                reasons=["canonical rule not found", f"matched_variant={observed}"],
            )

        variation_map = self.rules[canonical_key]
        haystacks = {
            observed.lower(),
            normalized.lower(),
            collapsed.lower(),
            self._collapse(observed).lower(),
        }

        for variation_type in VariationType:
            candidates = variation_map.get(variation_type, [])
            normalized_candidates = {self._collapse(candidate).lower() for candidate in candidates}
            raw_candidates = {candidate.lower() for candidate in candidates}
            if haystacks & normalized_candidates or haystacks & raw_candidates:
                return VariationClassificationResult(
                    canonical=canonical_key,
                    matched_variant=observed,
                    variation_type=variation_type,
                    reasons=[
                        f"matched canonical={canonical_key}",
                        f"matched_variant={observed}",
                        f"variation_type={variation_type.value}",
                    ],
                )

        return VariationClassificationResult(
            canonical=canonical_key,
            matched_variant=observed,
            variation_type=VariationType.MIXED_VARIATION,
            reasons=[
                f"matched canonical={canonical_key}",
                f"matched_variant={observed}",
                "no explicit variation rule matched",
                f"variation_type={VariationType.MIXED_VARIATION.value}",
            ],
        )

    def _collapse(self, text: str) -> str:
        return "".join(text.split())


@lru_cache(maxsize=4)
def load_variation_rules(path: Path = DEFAULT_RULE_PATH) -> dict[str, dict[VariationType, list[str]]]:
    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)

    rules = payload.get("rules", [])
    if not isinstance(rules, list):
        raise ValueError("variation rules payload must contain a list under `rules`")

    parsed: dict[str, dict[VariationType, list[str]]] = {}
    for rule in rules:
        if not isinstance(rule, dict):
            continue
        canonical = rule.get("canonical")
        variant_rules = rule.get("variant_rules")
        if not isinstance(canonical, str) or not canonical.strip() or not isinstance(variant_rules, dict):
            continue

        typed_rules: dict[VariationType, list[str]] = {}
        for variation_type in VariationType:
            raw_values = variant_rules.get(variation_type.value, [])
            if isinstance(raw_values, list):
                typed_rules[variation_type] = [
                    value for value in raw_values if isinstance(value, str) and value.strip()
                ]
            else:
                typed_rules[variation_type] = []
        parsed[canonical.strip()] = typed_rules
    return parsed
