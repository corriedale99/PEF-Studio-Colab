from __future__ import annotations

from pef2_engine.dictionary_loader import first_value
from pef2_engine.ruby import extract_ruby_annotations
from version import VERSION


DECISIONS = {"pending", "accept", "edit", "ignore", "promote"}
ITEM_LIST_KEYS = ("items", "terms", "dictionary", "work_dictionary", "results")
WORD_KEYS = ("単語原文", "term", "word", "surface", "英単語")
READING_KEYS = ("読み", "reading", "pronunciation", "カタカナ読み")
MEANING_KEYS = ("意味", "meaning", "description")
DIFFICULTY_KEYS = ("難易度", "difficulty")
GEMINI_SOURCES = {"gemini", "gemini_review", "ai_review"}


def build_review_items(draft_data: object, processed_data: object | None = None) -> list[dict]:
    raw_items = _extract_items(draft_data)
    items: list[dict] = []
    seen: set[tuple[str, str, str]] = set()

    for raw_item in raw_items:
        item = normalize_review_item(raw_item)
        if not item:
            continue
        key = (item["term"], item["reading"], item["source"])
        if key in seen:
            continue
        seen.add(key)
        items.append(item)

    for annotation in _ruby_annotations_from_processed(processed_data):
        item = normalize_review_item(annotation)
        if not item:
            continue
        key = (item["term"], item["reading"], item["source"])
        if key in seen:
            continue
        seen.add(key)
        items.append(item)

    for index, item in enumerate(items, start=1):
        item["id"] = item.get("id") or f"term-{index:04d}"
        if not item.get("segment_indexes"):
            item["segment_indexes"] = _find_segment_indexes(processed_data, item["term"])
    return items


def normalize_review_item(raw_item: object) -> dict | None:
    if not isinstance(raw_item, dict):
        return None
    term = first_value(raw_item, WORD_KEYS)
    reading = first_value(raw_item, READING_KEYS)
    if not term or not reading:
        return None

    source = str(raw_item.get("source") or "draft").strip()
    target_dictionary = str(raw_item.get("target_dictionary") or "work").strip()
    decision = validate_decision(str(raw_item.get("decision") or "pending").strip())
    item = {
        "id": str(raw_item.get("id") or "").strip(),
        "term": term,
        "reading": reading,
        "edited_reading": str(raw_item.get("edited_reading") or "").strip(),
        "meaning": first_value(raw_item, MEANING_KEYS),
        "difficulty": raw_item.get("難易度", raw_item.get("difficulty", "")),
        "confidence": raw_item.get("confidence", ""),
        "source": source,
        "target_dictionary": target_dictionary,
        "decision": decision,
        "notes": str(raw_item.get("notes") or "").strip(),
        "segment_indexes": _segment_indexes(raw_item),
        "promote_to_user_dictionary": False,
        "voicevox_katakana": _default_voicevox_katakana(raw_item, source, target_dictionary),
        "generator_version": VERSION,
    }
    if "term_type" in raw_item:
        item["term_type"] = raw_item["term_type"]
    return item


def validate_decision(decision: str) -> str:
    return decision if decision in DECISIONS else "pending"


def _extract_items(data: object) -> list:
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for key in ITEM_LIST_KEYS:
            value = data.get(key)
            if isinstance(value, list):
                return value
    return []


def _segment_indexes(raw_item: dict) -> list[int]:
    values = raw_item.get("segment_indexes", raw_item.get("segments", []))
    if isinstance(values, int):
        return [values]
    if not isinstance(values, list):
        return []
    indexes: list[int] = []
    for value in values:
        try:
            indexes.append(int(value))
        except (TypeError, ValueError):
            continue
    return indexes


def _default_voicevox_katakana(raw_item: dict, source: str, target_dictionary: str) -> bool:
    if "voicevox_katakana" in raw_item:
        return bool(raw_item["voicevox_katakana"])
    source_key = source.lower()
    return target_dictionary == "work" and (
        source_key in GEMINI_SOURCES or source_key.startswith("gemini") or source_key == "ruby_local"
    )


def _ruby_annotations_from_processed(processed_data: object | None) -> list[dict]:
    annotations: list[dict] = []
    for segment in _segments(processed_data):
        index = segment.get("index")
        text = _segment_text(segment)
        if text:
            annotations.extend(extract_ruby_annotations(text, index))
    return annotations


def _segments(processed_data: object | None) -> list[dict]:
    if isinstance(processed_data, dict):
        value = processed_data.get("segments", processed_data.get("remastered_data", []))
        return [item for item in value if isinstance(item, dict)] if isinstance(value, list) else []
    if isinstance(processed_data, list):
        return [item for item in processed_data if isinstance(item, dict)]
    return []


def _segment_text(segment: dict) -> str:
    display = segment.get("display", "")
    if isinstance(display, dict):
        return str(display.get("text") or "")
    return str(display or segment.get("text_raw") or segment.get("text") or "")


def _find_segment_indexes(processed_data: object | None, term: str) -> list[int]:
    indexes: list[int] = []
    for segment in _segments(processed_data):
        text = _segment_text(segment)
        if term in text:
            try:
                indexes.append(int(segment.get("index")))
            except (TypeError, ValueError):
                continue
    return indexes
