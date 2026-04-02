import json
import re
from dataclasses import dataclass
from pathlib import Path

from app.domain.models import NormalizedQuery

NORMALIZER_REPLACEMENTS_PATH = (
    Path(__file__).resolve().parents[2] / "dataset" / "normalizer_replacements.json"
)

"""
텍스트 전처리 및 정규화 담당 서비스
"""
@dataclass(frozen=True, slots=True)
class NormalizationRuleSet:
    punctuation_pattern: re.Pattern[str]
    whitespace_pattern: re.Pattern[str]
    repeated_char_pattern: re.Pattern[str]
    replacement_map: dict[str, str]
    replacement_pattern: re.Pattern[str]


class QueryNormalizer:
    def __init__(self, rules: NormalizationRuleSet | None = None) -> None:
        self.rules = rules or default_rule_set()

    def normalize(self, text: str) -> NormalizedQuery:
        raw = text.strip()
        replaced = self._replace_noise_tokens(raw)
        no_punctuation = self._remove_punctuation(replaced)
        normalized = self._normalize_whitespace(no_punctuation)
        collapsed = self._collapse_repeated_characters(normalized)
        tokens_for_debug = self._build_debug_tokens(
            raw=raw,
            replaced=replaced,
            normalized=normalized,
            collapsed=collapsed,
        )

        return NormalizedQuery(
            raw=raw,
            normalized=normalized,
            collapsed=collapsed.replace(" ", ""),
            replaced=replaced,
            tokens_for_debug=tokens_for_debug,
        )

    def _replace_noise_tokens(self, text: str) -> str:
        return self.rules.replacement_pattern.sub(
            lambda match: self.rules.replacement_map[match.group(0)],
            text,
        )

    def _remove_punctuation(self, text: str) -> str:
        return self.rules.punctuation_pattern.sub(" ", text)

    def _normalize_whitespace(self, text: str) -> str:
        compact = self.rules.whitespace_pattern.sub(" ", text).strip()
        return compact

    def _collapse_repeated_characters(self, text: str) -> str:
        return self.rules.repeated_char_pattern.sub(r"\1\1", text)

    def _build_debug_tokens(
        self,
        *,
        raw: str,
        replaced: str,
        normalized: str,
        collapsed: str,
    ) -> list[str]:
        tokens = [
            f"raw={raw}",
            f"replaced={replaced}",
            f"normalized={normalized}",
            f"collapsed={collapsed}",
            f"collapsed_no_space={collapsed.replace(' ', '')}",
        ]
        return tokens


def load_replacement_map(path: Path | None = None) -> dict[str, str]:
    dataset_path = path or NORMALIZER_REPLACEMENTS_PATH
    with dataset_path.open("r", encoding="utf-8") as file:
        payload = json.load(file)

    if not isinstance(payload, dict):
        raise ValueError("Normalizer replacement dataset must be a JSON object.")

    replacement_map: dict[str, str] = {}
    for source, target in payload.items():
        if not isinstance(source, str) or not isinstance(target, str):
            raise ValueError("Normalizer replacement dataset keys and values must be strings.")
        replacement_map[source] = target

    return replacement_map


def default_rule_set() -> NormalizationRuleSet:
    replacement_map = load_replacement_map()
    replacement_pattern = re.compile(
        "|".join(
            re.escape(source)
            for source in sorted(replacement_map, key=len, reverse=True)
        )
    )

    return NormalizationRuleSet(
        punctuation_pattern=re.compile(r"[!?,.\-_=~，・'\"`]+"),
        whitespace_pattern=re.compile(r"\s+"),
        repeated_char_pattern=re.compile(r"(.)\1{2,}"),
        replacement_map=replacement_map,
        replacement_pattern=replacement_pattern,
    )


def normalize_text(text: str) -> NormalizedQuery:
    return QueryNormalizer().normalize(text)
