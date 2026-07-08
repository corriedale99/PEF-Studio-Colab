from __future__ import annotations

import html
import json
import re
from pathlib import Path

from pef2_engine.io_utils import read_json
from pef2_engine.paths import resolve_named_file_candidates
from version import VERSION


WORD_KEYS = ("単語原文", "term", "word", "surface", "英単語")
READING_KEYS = ("読み", "reading", "pronunciation", "カタカナ読み")
ASCII_WORD_PATTERN = re.compile(r"^[A-Za-z0-9 _-]+$")
SYMBOL_READING_RULES_FILENAME = "★読み上げ記号ルール.json"
SYMBOL_READING_RULES_SCHEMA_VERSION = "symbol-reading-rules-1"
SYMBOL_READING_RULE_CATEGORIES = ("drop", "pause_s", "pause_m", "keep")
SYMBOL_READING_RULE_NON_DISPLAY_CATEGORIES = ("sentence_end_pause",)


def load_dictionary_file(path: Path, priority: int, source_file: str | None = None) -> list[dict]:
    data = read_json(path, default=[])
    if not isinstance(data, list):
        return []

    entries: list[dict] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        word = first_value(item, WORD_KEYS)
        reading = first_value(item, READING_KEYS)
        if not word or not reading:
            continue
        entry = dict(item)
        entry["word"] = word
        entry["reading"] = reading
        entry["source_file"] = source_file or path.name
        entry["priority"] = priority
        entries.append(entry)
    return entries


def load_reading_dictionaries(
    system_dict_path: Path,
    standard_english_dict_path: Path,
    user_dict_path: Path,
    work_dict_path: Path,
) -> list[dict]:
    all_entries: list[dict] = []
    if system_dict_path.exists():
        all_entries.extend(load_dictionary_file(system_dict_path, 1))
    if standard_english_dict_path.exists():
        all_entries.extend(load_dictionary_file(standard_english_dict_path, 2))
    if work_dict_path.exists():
        all_entries.extend(load_dictionary_file(work_dict_path, 3))
    if user_dict_path.exists():
        all_entries.extend(load_dictionary_file(user_dict_path, 4))
    return merge_dictionaries_by_priority(all_entries)


def merge_dictionaries_by_priority(entries: list[dict]) -> list[dict]:
    entries_by_word: dict[str, dict] = {}
    for entry in entries:
        word = entry.get("word", "")
        if not word:
            continue
        current = entries_by_word.get(word)
        if current is None or entry.get("priority", 0) >= current.get("priority", 0):
            entries_by_word[word] = entry
    return sorted(entries_by_word.values(), key=lambda item: len(item["word"]), reverse=True)


def load_breath_rules(breath_rules_path: Path) -> dict:
    data = read_json(breath_rules_path, default={})
    return data if isinstance(data, dict) else {}


def load_symbol_reading_rules(workspace_root: Path) -> tuple[dict, dict, list[str]]:
    rules_path = _resolve_symbol_reading_rules_path(
        Path(workspace_root) / "dictionaries" / "system"
    )
    warnings: list[str] = []
    if not rules_path.exists():
        return {}, {}, [f"missing_symbol_reading_rules: {rules_path}"]

    try:
        data = read_json(rules_path)
    except json.JSONDecodeError as error:
        return {}, {}, [f"invalid_symbol_reading_rules_json: {type(error).__name__}: {error}"]
    except Exception as error:
        return {}, {}, [f"symbol_reading_rules_read_error: {type(error).__name__}: {error}"]

    if not isinstance(data, dict):
        return {}, {}, ["invalid_symbol_reading_rules: top-level must be object"]

    schema_version = data.get("schema_version")
    if schema_version != SYMBOL_READING_RULES_SCHEMA_VERSION:
        warnings.append(
            "invalid_symbol_reading_rules_schema_version: "
            f"expected {SYMBOL_READING_RULES_SCHEMA_VERSION}, got {schema_version}"
        )

    known_keys = (
        set(SYMBOL_READING_RULE_CATEGORIES)
        | set(SYMBOL_READING_RULE_NON_DISPLAY_CATEGORIES)
        | {"schema_version"}
    )
    for key in data:
        if key not in known_keys:
            warnings.append(f"unknown_symbol_reading_rules_category: {key}")

    rules_by_category: dict[str, list[str]] = {}
    symbol_to_category: dict[str, str] = {}
    for category in SYMBOL_READING_RULE_CATEGORIES:
        value = data.get(category)
        if value is None:
            continue
        if not isinstance(value, list):
            warnings.append(f"invalid_symbol_reading_rules_category: {category} must be array")
            continue

        rules: list[str] = []
        seen_in_category: set[str] = set()
        for position, symbol in enumerate(value):
            if not isinstance(symbol, str):
                warnings.append(
                    f"invalid_symbol_reading_rules_item: {category}[{position}] must be string"
                )
                continue
            if symbol in seen_in_category:
                warnings.append(
                    f"duplicate_symbol_reading_rules_item: {symbol} duplicated in {category}"
                )
                continue
            seen_in_category.add(symbol)
            if symbol in symbol_to_category:
                warnings.append(
                    "duplicate_symbol_reading_rules_category: "
                    f"{symbol} already in {symbol_to_category[symbol]}, ignored in {category}"
                )
                continue
            rules.append(symbol)
            symbol_to_category[symbol] = category

        rules_by_category[category] = rules

    return rules_by_category, symbol_to_category, warnings


def _resolve_symbol_reading_rules_path(system_dir: Path) -> Path:
    candidates = resolve_named_file_candidates(system_dir, SYMBOL_READING_RULES_FILENAME)
    fallback_path = system_dir / SYMBOL_READING_RULES_FILENAME
    if not candidates:
        return fallback_path
    if len(candidates) == 1:
        return candidates[0]

    for path in candidates:
        try:
            data = read_json(path)
        except Exception:
            continue
        if isinstance(data, dict) and data.get("sentence_end_pause"):
            return path
    return candidates[0]


def find_dictionary_matches(text: str, entries: list[dict]) -> list[dict]:
    matches: list[dict] = []
    occupied_ranges: list[tuple[int, int]] = []
    for entry in sorted(entries, key=lambda item: len(item.get("word", "")), reverse=True):
        word = entry.get("word", "")
        if not word:
            continue
        start = text.find(word)
        while start != -1:
            end = start + len(word)
            if _has_valid_boundary(text, word, start, end) and not _overlaps(
                start, end, occupied_ranges
            ):
                matches.append(
                    {
                        "index": len(matches),
                        "word": word,
                        "reading": entry["reading"],
                        "match_start": start,
                        "match_end": end,
                        "source_file": entry.get("source_file", ""),
                        "priority": entry.get("priority", 0),
                        "source": entry.get("source", "dictionary"),
                        "voicevox_katakana": bool(entry.get("voicevox_katakana")),
                    }
                )
                occupied_ranges.append((start, end))
            start = text.find(word, start + 1)
    matches.sort(key=lambda item: (item["match_start"], -len(item["word"])))
    for index, match in enumerate(matches):
        match["index"] = index
    return matches


def build_text_ssml(text: str, matches: list[dict]) -> tuple[str, list[dict]]:
    text_ssml = text
    substitutions: list[dict] = []
    for match in sorted(matches, key=lambda item: item.get("match_start", -1), reverse=True):
        start = match.get("match_start")
        end = match.get("match_end")
        word = match.get("word", "")
        reading = match.get("reading", "")
        if not _valid_match(text, word, start, end) or not reading:
            continue
        replacement = f'<sub alias="{html.escape(reading, quote=True)}">{html.escape(word)}</sub>'
        text_ssml = text_ssml[:start] + replacement + text_ssml[end:]
        substitutions.append({**match, "replacement": replacement})
    substitutions.sort(key=lambda item: item["match_start"])
    return text_ssml, substitutions


def first_value(item: dict, keys: tuple[str, ...]) -> str:
    for key in keys:
        value = item.get(key)
        if value is not None:
            text = str(value).strip()
            if text:
                return text
    return ""


def _valid_match(text: str, word: str, start: object, end: object) -> bool:
    if not isinstance(start, int) or not isinstance(end, int):
        return False
    return 0 <= start < end <= len(text) and text[start:end] == word


def _has_valid_boundary(text: str, word: str, start: int, end: int) -> bool:
    if not ASCII_WORD_PATTERN.fullmatch(word):
        return True
    previous_char = text[start - 1] if start > 0 else ""
    next_char = text[end] if end < len(text) else ""
    return not (_is_ascii_alnum(previous_char) or _is_ascii_alnum(next_char))


def _is_ascii_alnum(char: str) -> bool:
    return bool(char) and char.isascii() and char.isalnum()


def _overlaps(start: int, end: int, ranges: list[tuple[int, int]]) -> bool:
    return any(start < existing_end and end > existing_start for existing_start, existing_end in ranges)
