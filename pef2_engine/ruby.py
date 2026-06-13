from __future__ import annotations

import re

from version import VERSION


RUBY_WITH_MARK_PATTERN = re.compile(r"｜([^《》]+?)《([^《》]+?)》")
RUBY_KANJI_PATTERN = re.compile(r"([一-龯々〆ヵヶ]+)《([^《》]+?)》")


def strip_aozora_ruby(text: str) -> str:
    text = RUBY_WITH_MARK_PATTERN.sub(lambda match: match.group(1), text)
    return RUBY_KANJI_PATTERN.sub(lambda match: match.group(1), text)


def extract_ruby_annotations(text: str, segment_index: int | None = None) -> list[dict]:
    annotations: list[dict] = []
    for pattern in (RUBY_WITH_MARK_PATTERN, RUBY_KANJI_PATTERN):
        for match in pattern.finditer(text):
            surface = match.group(1)
            reading = match.group(2)
            annotations.append(
                {
                    "surface": surface,
                    "reading": reading,
                    "source": "ruby_local",
                    "target_dictionary": "work",
                    "segment_index": segment_index,
                }
            )
    return annotations
