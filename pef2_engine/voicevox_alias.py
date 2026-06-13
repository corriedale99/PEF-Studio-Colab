from __future__ import annotations

from version import VERSION


HIRAGANA_START = ord("ぁ")
HIRAGANA_END = ord("ゖ")
KATAKANA_OFFSET = ord("ァ") - ord("ぁ")


def should_katakanaize_for_voicevox(item: dict) -> bool:
    return bool(item.get("voicevox_katakana"))


def hiragana_to_katakana(text: str) -> str:
    chars: list[str] = []
    for char in text:
        code = ord(char)
        if HIRAGANA_START <= code <= HIRAGANA_END:
            chars.append(chr(code + KATAKANA_OFFSET))
        else:
            chars.append(char)
    return "".join(chars)


def build_voicevox_alias(item: dict, reading: str) -> str:
    if should_katakanaize_for_voicevox(item):
        return hiragana_to_katakana(reading)
    return reading
