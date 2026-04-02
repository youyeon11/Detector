from enum import StrEnum


class VariationType(StrEnum):
    NUMERIC_SUBSTITUTION = "numeric_substitution"
    ALPHABET_SUBSTITUTION = "alphabet_substitution"
    PHONETIC_VARIATION = "phonetic_variation"
    ORTHOGRAPHIC_VARIATION = "orthographic_variation"
    ABBREVIATION = "abbreviation"
    SPECIAL_CHAR_INSERTION = "special_char_insertion"
    SPACING_VARIATION = "spacing_variation"
    REPEATED_CHAR_EXPANSION = "repeated_char_expansion"
    MIXED_VARIATION = "mixed_variation"
