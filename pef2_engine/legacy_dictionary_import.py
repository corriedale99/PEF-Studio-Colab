from __future__ import annotations

import json
from dataclasses import dataclass

from pef2_engine.dictionary_loader import first_value


SCHEMA_VERSION = "dictionary_review-1"
SOURCE = "legacy_pef_dictionary"
GENERATED_FROM = "legacy_dictionary.json"
WORD_KEYS = ("単語原文", "word", "英単語")
READING_KEYS = ("読み", "pronunciation", "カタカナ読み")
MEANING_KEYS = ("意味", "meaning", "description")
DIFFICULTY_KEYS = ("難易度", "difficulty")


@dataclass
class LegacyDictionaryImportValidationError(Exception):
    code: str
    message: str
    item_index: int | None = None


def load_legacy_dictionary_json_bytes(payload: bytes) -> object:
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as error:
        raise LegacyDictionaryImportValidationError(
            code="invalid_json",
            message="旧PEF辞書jsonを読み込めませんでした。",
        ) from error

    try:
        return json.loads(text)
    except json.JSONDecodeError as error:
        raise LegacyDictionaryImportValidationError(
            code="invalid_json",
            message="旧PEF辞書jsonを読み込めませんでした。",
        ) from error


def build_dictionary_review_from_legacy_dictionary(
    legacy_data: object,
    *,
    input_stem: str,
    generated_from: str = GENERATED_FROM,
) -> tuple[dict, list[dict]]:
    if not isinstance(legacy_data, list):
        raise LegacyDictionaryImportValidationError(
            code="invalid_top_level",
            message="旧PEF辞書jsonのtop-levelは配列である必要があります。",
        )

    items: list[dict] = []
    warnings: list[dict] = []
    seen_words: set[str] = set()

    for source_index, raw_item in enumerate(legacy_data, start=1):
        if not isinstance(raw_item, dict):
            raise LegacyDictionaryImportValidationError(
                code="invalid_item",
                message="旧PEF辞書jsonのitemはobjectである必要があります。",
                item_index=source_index,
            )

        word = first_value(raw_item, WORD_KEYS)
        if not word:
            raise LegacyDictionaryImportValidationError(
                code="missing_word",
                message="旧PEF辞書jsonにwordがありません。",
                item_index=source_index,
            )

        reading = first_value(raw_item, READING_KEYS)
        if not reading:
            raise LegacyDictionaryImportValidationError(
                code="missing_reading",
                message="旧PEF辞書jsonにreadingがありません。",
                item_index=source_index,
            )

        if word in seen_words:
            warnings.append(
                {
                    "code": "duplicate_word_skipped",
                    "word": word,
                    "item_index": source_index,
                    "message": "重複wordのため2件目以降をスキップしました。",
                }
            )
            continue

        seen_words.add(word)
        items.append(
            {
                "index": len(items) + 1,
                "word": word,
                "reading_suggested": reading,
                "reading_final": reading,
                "meaning": first_value(raw_item, MEANING_KEYS),
                "difficulty": _first_raw_value(raw_item, DIFFICULTY_KEYS),
                "confidence": "legacy_pef",
                "decision": "pending",
                "target_dictionary": "work",
                "promote_to_user_dictionary": False,
                "source": SOURCE,
                "notes": "旧PEF辞書から取り込み",
            }
        )

    if not items:
        raise LegacyDictionaryImportValidationError(
            code="empty_dictionary",
            message="読み込める辞書項目がありません。空の辞書を作る場合は「手動で辞書を作る」を使ってください。",
        )

    return (
        {
            "schema_version": SCHEMA_VERSION,
            "source": SOURCE,
            "input_stem": input_stem or "unknown",
            "generated_from": generated_from,
            "items": items,
        },
        warnings,
    )


def _first_raw_value(item: dict, keys: tuple[str, ...]) -> object:
    for key in keys:
        if key not in item:
            continue
        value = item.get(key)
        if value is None:
            continue
        if isinstance(value, str):
            value = value.strip()
            if value == "":
                continue
        return value
    return None
