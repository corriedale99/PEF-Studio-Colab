from __future__ import annotations

import html
import json
import re
import unicodedata
from pathlib import Path
from typing import Any

from pef2_engine import workspace_paths
from pef2_engine.io_utils import read_json

SCHEMA_VERSION = "tts-pre-transform-1"
REPORT_SCHEMA_VERSION = "tts-precheck-report-1"
SYMBOL_RULES_SCHEMA_VERSION = "symbol-reading-rules-1"
SYMBOL_RULES_FILENAME = "★読み上げ記号ルール.json"

SILENCE = {
    "L": 1.2,
    "M": 0.7,
    "S": 0.4,
    "MIN": 0.2,
}

ALLOWED_PAUSE_TYPES = tuple(SILENCE.keys())
OLD_PAUSE_TYPES = {
    "L-PAUSE": "L",
    "M-PAUSE": "M",
    "S-PAUSE": "S",
}
PAUSE_PRIORITY = {"MIN": 0, "S": 1, "M": 2, "L": 3}
LEGACY_PAUSE_MARKER_PATTERN = re.compile(r"\[(?:S|M|L)-PAUSE\]")
SUB_ALIAS_PATTERN = re.compile(
    r"<sub\s+alias=(?P<quote>['\"])(?P<alias>.*?)(?P=quote)>(?P<surface>.*?)</sub>",
    re.DOTALL,
)
SUB_TAG_PATTERN = re.compile(r"<sub\b|</sub>", re.IGNORECASE)


def run_tts_pre_transform(work_dir: Path, workspace_root: Path | None = None) -> dict:
    work_dir = Path(work_dir)
    workspace_root = (
        Path(workspace_root) if workspace_root is not None else work_dir.parent
    )
    result = _new_result(work_dir.name)
    source_path = workspace_paths.processed_final_path(work_dir)
    rules_path = workspace_root / "dictionaries" / "system" / SYMBOL_RULES_FILENAME

    processed = _read_required_json(
        source_path,
        "missing_processed_final",
        "invalid_processed_final_json",
        result,
    )
    rules_data = _read_required_json(
        rules_path,
        "missing_symbol_reading_rules",
        "invalid_symbol_reading_rules_json",
        result,
    )
    if processed is None or rules_data is None:
        _finish_result(result)
        return result

    transform_result = transform_processed_to_tts_units(
        processed, rules_data, work_id=work_dir.name
    )
    result["tts_units"] = transform_result["tts_units"]
    result["errors"].extend(transform_result["errors"])
    result["warnings"].extend(transform_result["warnings"])
    result["summary"] = transform_result["summary"]
    result["_segment_count"] = transform_result.get("_segment_count", 0)
    _finish_result(result)
    return result


def transform_processed_to_tts_units(
    processed: object,
    symbol_rules_data: object,
    *,
    work_id: str = "",
) -> dict:
    result = _new_result(work_id)
    normalized_rules = normalize_symbol_rules(symbol_rules_data)
    result["errors"].extend(normalized_rules["errors"])
    result["warnings"].extend(normalized_rules["warnings"])

    segments = _extract_segments(processed, result)
    result["_segment_count"] = len(segments)
    if result["errors"]:
        _finish_result(result)
        return result

    rules = normalized_rules["rules"]
    tts_units: list[dict] = []
    for segment in sorted(segments, key=lambda item: item.get("index", 0)):
        segment_units = _build_segment_units(segment, rules, result)
        _assign_unit_ids(segment_units)
        tts_units.extend(segment_units)

    result["tts_units"] = tts_units
    _finish_result(result)
    return result


def normalize_symbol_rules(data: object) -> dict:
    errors: list[dict] = []
    warnings: list[dict] = []
    rules = {
        "drop": [],
        "replace_with_comma": [],
        "pause": {pause_type: [] for pause_type in ALLOWED_PAUSE_TYPES},
        "keep": [],
        "sentence_end_pause": [],
        "unknown_symbol_action": "keep",
        "symbols": [],
    }

    if not isinstance(data, dict):
        errors.append(
            _error(
                "invalid_symbol_reading_rules", "symbol rules top-level must be object"
            )
        )
        return {"rules": rules, "errors": errors, "warnings": warnings}

    if data.get("schema_version") != SYMBOL_RULES_SCHEMA_VERSION:
        errors.append(
            _error(
                "invalid_symbol_reading_rules_schema_version",
                f"schema_version must be {SYMBOL_RULES_SCHEMA_VERSION}",
            )
        )

    _load_symbol_list(data, "drop", rules["drop"], errors)
    _load_symbol_list(data, "replace_with_comma", rules["replace_with_comma"], errors)
    _load_symbol_list(data, "keep", rules["keep"], errors)
    _load_symbol_list(data, "sentence_end_pause", rules["sentence_end_pause"], errors)

    if "pause_s" in data:
        warnings.append(
            _warning(
                "legacy_symbol_rule_key",
                "pause_s was treated as replace_with_comma",
                key="pause_s",
            )
        )
        _load_symbol_list(data, "pause_s", rules["replace_with_comma"], errors)
    if "pause_m" in data:
        warnings.append(
            _warning(
                "legacy_symbol_rule_key",
                "pause_m was treated as pause.M",
                key="pause_m",
            )
        )
        _load_symbol_list(data, "pause_m", rules["pause"]["M"], errors)

    pause_rules = data.get("pause", {})
    if pause_rules is None:
        pause_rules = {}
    if not isinstance(pause_rules, dict):
        errors.append(_error("invalid_symbol_reading_rules", "pause must be object"))
    else:
        for pause_type, symbols in pause_rules.items():
            if pause_type not in ALLOWED_PAUSE_TYPES:
                errors.append(
                    _error(
                        "unknown_symbol_pause_type",
                        "pause has unknown pause_type",
                        pause_type=pause_type,
                    )
                )
                continue
            _load_symbol_list(
                {"pause": symbols},
                "pause",
                rules["pause"][pause_type],
                errors,
                label=f"pause.{pause_type}",
            )

    defaults = data.get("defaults", {})
    if defaults is None:
        defaults = {}
    if not isinstance(defaults, dict):
        errors.append(_error("invalid_symbol_reading_rules", "defaults must be object"))
    else:
        unknown_action = defaults.get("unknown_symbol_action", "keep")
        if unknown_action not in {"keep", "drop"}:
            errors.append(
                _error(
                    "invalid_unknown_symbol_action",
                    "unknown_symbol_action must be keep or drop",
                )
            )
        else:
            rules["unknown_symbol_action"] = unknown_action

    _validate_symbol_category_overlap(rules, errors)
    symbols = []
    symbols.extend(rules["drop"])
    symbols.extend(rules["replace_with_comma"])
    symbols.extend(rules["keep"])
    symbols.extend(rules["sentence_end_pause"])
    for pause_type in ALLOWED_PAUSE_TYPES:
        symbols.extend(rules["pause"][pause_type])
    rules["symbols"] = sorted(set(symbols), key=len, reverse=True)
    return {"rules": rules, "errors": errors, "warnings": warnings}


def make_precheck_report(result: dict, work_id: str) -> dict:
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "work_id": work_id,
        "source_file": workspace_paths.PROCESSED_FINAL_FILENAME,
        "ok": not result.get("errors"),
        "error_count": len(result.get("errors", [])),
        "warning_count": len(result.get("warnings", [])),
        "errors": result.get("errors", []),
        "warnings": result.get("warnings", []),
        "summary": result.get("summary", {}),
    }


def _build_segment_units(segment: dict, rules: dict, result: dict) -> list[dict]:
    index = segment.get("index")
    block_type = str(segment.get("block_type") or "")
    if block_type == "image" or segment.get("is_image") is True:
        return _build_image_units(segment, result)
    return _build_text_units(segment, rules, result)


def _build_text_units(segment: dict, rules: dict, result: dict) -> list[dict]:
    index = segment.get("index")
    block_type = str(segment.get("block_type") or "")
    if "audio" not in segment:
        result["errors"].append(
            _error("missing_audio", "text segment has no audio", index=index)
        )
        return []

    audio = str(segment.get("audio") or "")
    if audio == "":
        result["warnings"].append(
            _warning("empty_audio", "text segment audio is empty", index=index)
        )
    if LEGACY_PAUSE_MARKER_PATTERN.search(audio):
        result["errors"].append(
            _error(
                "legacy_pause_marker_in_audio",
                "audio contains legacy pause marker",
                index=index,
            )
        )

    text = _expand_sub_alias(audio, index, result)
    units = _units_from_text(text, index, block_type, rules, result)
    _append_sentence_end_pause(units, index, block_type, rules, result)
    pause_after = segment.get("pause_after")
    if pause_after is not None:
        pause_type = _normalize_pause_value(
            pause_after, result, "invalid_pause_after", index=index
        )
        if pause_type:
            _append_pause(units, index, block_type, pause_type, "pause_after", result)

    if not units:
        result["warnings"].append(
            _warning(
                "empty_text_fallback", "text segment produced no tts unit", index=index
            )
        )
        _append_pause(units, index, block_type, "MIN", "empty_text_fallback", result)
    return units


def _build_image_units(segment: dict, result: dict) -> list[dict]:
    index = segment.get("index")
    block_type = str(segment.get("block_type") or "image")
    if "audio_policy" not in segment:
        result["errors"].append(
            _error(
                "missing_audio_policy", "image segment has no audio_policy", index=index
            )
        )
        return []
    if segment.get("audio_policy") != "pause_only":
        result["errors"].append(
            _error(
                "unsupported_image_audio_policy",
                "image audio_policy must be pause_only",
                index=index,
            )
        )
        return []
    pause_type = _normalize_pause_value(
        segment.get("pause_type"), result, "invalid_pause_type", index=index
    )
    if not pause_type:
        return []
    units: list[dict] = []
    _append_pause(units, index, block_type, pause_type, "image_pause_only", result)
    return units


def _units_from_text(
    text: str, index: int, block_type: str, rules: dict, result: dict
) -> list[dict]:
    units: list[dict] = []
    buffer: list[str] = []
    position = 0
    drop_only_since_unit = False
    while position < len(text):
        symbol = _match_symbol_at(text, position, rules["symbols"])
        if symbol:
            category = _symbol_category(symbol, rules)
            if category == "drop":
                if not buffer:
                    drop_only_since_unit = True
            elif category == "replace_with_comma":
                _append_comma(buffer, index, result)
                drop_only_since_unit = False
            elif category.startswith("pause."):
                if drop_only_since_unit and not buffer:
                    result["warnings"].append(
                        _warning(
                            "drop_empty_speak",
                            "drop symbols produced empty speak text",
                            index=index,
                        )
                    )
                    drop_only_since_unit = False
                _flush_speak(units, buffer, index, block_type)
                _append_pause(
                    units,
                    index,
                    block_type,
                    category.split(".", 1)[1],
                    "symbol_pause",
                    result,
                )
            elif category == "keep":
                _append_keep_symbol(buffer, symbol, index, result)
                drop_only_since_unit = False
            elif category == "sentence_end_pause":
                _append_keep_symbol(buffer, symbol, index, result)
                drop_only_since_unit = False
            position += len(symbol)
            continue

        char = text[position]
        if _is_unknown_symbol(char):
            result["warnings"].append(
                _warning(
                    "unknown_symbol",
                    "unknown symbol was kept",
                    index=index,
                    symbol=char,
                )
            )
            if rules["unknown_symbol_action"] == "drop":
                position += 1
                continue
        buffer.append(char)
        drop_only_since_unit = False
        position += 1

    if drop_only_since_unit and not buffer:
        result["warnings"].append(
            _warning(
                "drop_empty_speak",
                "drop symbols produced empty speak text",
                index=index,
            )
        )
    _flush_speak(units, buffer, index, block_type)
    return units


def _append_comma(buffer: list[str], index: int, result: dict) -> None:
    if buffer and buffer[-1] == "、":
        result["warnings"].append(
            _warning(
                "comma_compressed", "consecutive comma was compressed", index=index
            )
        )
        return
    buffer.append("、")


def _append_keep_symbol(
    buffer: list[str], symbol: str, index: int, result: dict
) -> None:
    if symbol == "、" and buffer and buffer[-1] == "、":
        result["warnings"].append(
            _warning(
                "comma_compressed", "consecutive comma was compressed", index=index
            )
        )
        return
    buffer.append(symbol)


def _append_sentence_end_pause(
    units: list[dict], index: int, block_type: str, rules: dict, result: dict
) -> None:
    if not units or units[-1].get("type") != "speak":
        return
    sentence_end_symbols = tuple(rules.get("sentence_end_pause") or [])
    if not sentence_end_symbols:
        return
    text = str(units[-1].get("text") or "").rstrip()
    if text.endswith(sentence_end_symbols):
        #        _append_pause(units, index, block_type, "S", "sentence_end_punctuation", result)
        _append_pause(
            units, index, block_type, "M", "sentence_end_punctuation", result
        )  # 一時テスト


def _flush_speak(
    units: list[dict], buffer: list[str], index: int, block_type: str
) -> None:
    text = "".join(buffer)
    buffer.clear()
    if not text:
        return
    units.append(
        {
            "type": "speak",
            "segment_index": index,
            "block_type": block_type,
            "text": text,
            "source": "audio",
        }
    )


def _append_pause(
    units: list[dict],
    index: int,
    block_type: str,
    pause_type: str,
    source: str,
    result: dict,
) -> None:
    if units and units[-1].get("type") == "pause":
        before = str(units[-1].get("pause_type"))
        merged = _stronger_pause(before, pause_type)
        units[-1]["pause_type"] = merged
        result["warnings"].append(
            _warning(
                "continuous_pause_merged",
                "continuous pause units were merged",
                index=index,
                before=[before, pause_type],
                merged=merged,
            )
        )
        return
    units.append(
        {
            "type": "pause",
            "segment_index": index,
            "block_type": block_type,
            "pause_type": pause_type,
            "source": source,
        }
    )


def _assign_unit_ids(units: list[dict]) -> None:
    for unit_index, unit in enumerate(units):
        segment_index = int(unit.get("segment_index", 0))
        unit["unit_id"] = f"seg_{segment_index:04d}_u{unit_index:03d}"


def _expand_sub_alias(text: str, index: int, result: dict) -> str:
    alias_errors: list[dict] = []

    def replace(match: re.Match[str]) -> str:
        alias = html.unescape(match.group("alias")).strip()
        if not alias:
            alias_errors.append(
                _error("empty_sub_alias", "sub alias is empty", index=index)
            )
            return ""
        return alias

    expanded = SUB_ALIAS_PATTERN.sub(replace, text)
    result["errors"].extend(alias_errors)
    if SUB_TAG_PATTERN.search(expanded):
        result["errors"].append(
            _error("broken_sub_tag", "sub tag is broken", index=index)
        )
    return expanded


def _extract_segments(processed: object, result: dict) -> list[dict]:
    if not isinstance(processed, dict):
        result["errors"].append(
            _error(
                "invalid_processed_final",
                "04_processed_final.json top-level must be object",
            )
        )
        return []
    segments = processed.get("segments")
    if segments is None and isinstance(processed.get("remastered_data"), list):
        result["warnings"].append(
            _warning(
                "legacy_remastered_data_used",
                "remastered_data was used as segment list",
            )
        )
        segments = processed.get("remastered_data")
    if not isinstance(segments, list):
        result["errors"].append(_error("missing_segments", "segments must be a list"))
        return []

    valid_segments: list[dict] = []
    seen: set[int] = set()
    for position, segment in enumerate(segments):
        if not isinstance(segment, dict):
            result["errors"].append(
                _error("invalid_segment", "segment must be object", position=position)
            )
            continue
        index = segment.get("index")
        if not isinstance(index, int):
            result["errors"].append(
                _error(
                    "missing_segment_index",
                    "segment.index is missing",
                    position=position,
                )
            )
            continue
        if index in seen:
            result["errors"].append(
                _error(
                    "duplicate_segment_index",
                    "segment.index is duplicated",
                    index=index,
                )
            )
            continue
        seen.add(index)
        if not segment.get("block_type"):
            result["errors"].append(
                _error(
                    "missing_block_type", "segment.block_type is missing", index=index
                )
            )
            continue
        valid_segments.append(segment)
    return valid_segments


def _normalize_pause_value(
    value: object, result: dict, code: str, *, index: int | None = None
) -> str | None:
    raw_value = value
    if isinstance(value, dict):
        raw_value = value.get("type")
    if not isinstance(raw_value, str):
        result["errors"].append(
            _error(code, "pause_type is invalid", index=index, value=raw_value)
        )
        return None
    pause_type = raw_value.strip()
    if pause_type in OLD_PAUSE_TYPES:
        return OLD_PAUSE_TYPES[pause_type]
    if pause_type not in ALLOWED_PAUSE_TYPES:
        result["errors"].append(
            _error(code, "pause_type is invalid", index=index, value=raw_value)
        )
        return None
    return pause_type


def _load_symbol_list(
    data: dict,
    key: str,
    output: list[str],
    errors: list[dict],
    *,
    label: str | None = None,
) -> None:
    value = data.get(key)
    if value is None:
        return
    display_key = label or key
    if not isinstance(value, list):
        errors.append(
            _error("invalid_symbol_reading_rules", f"{display_key} must be array")
        )
        return
    for position, symbol in enumerate(value):
        if not isinstance(symbol, str) or symbol == "":
            errors.append(
                _error(
                    "invalid_symbol_reading_rules",
                    f"{display_key}[{position}] must be non-empty string",
                )
            )
            continue
        output.append(symbol)


def _validate_symbol_category_overlap(rules: dict, errors: list[dict]) -> None:
    symbol_to_category: dict[str, str] = {}
    categories: list[tuple[str, list[str]]] = [
        ("drop", rules["drop"]),
        ("replace_with_comma", rules["replace_with_comma"]),
        ("keep", rules["keep"]),
    ]
    for pause_type in ALLOWED_PAUSE_TYPES:
        categories.append((f"pause.{pause_type}", rules["pause"][pause_type]))
    for category, symbols in categories:
        unique_symbols: list[str] = []
        seen_in_category: set[str] = set()
        for symbol in symbols:
            if symbol in seen_in_category:
                continue
            seen_in_category.add(symbol)
            if symbol in symbol_to_category:
                errors.append(
                    _error(
                        "duplicate_symbol_category",
                        "symbol is used in multiple categories",
                        symbol=symbol,
                        first_category=symbol_to_category[symbol],
                        second_category=category,
                    )
                )
                continue
            symbol_to_category[symbol] = category
            unique_symbols.append(symbol)
        symbols[:] = unique_symbols


def _symbol_category(symbol: str, rules: dict) -> str:
    if symbol in rules["drop"]:
        return "drop"
    if symbol in rules["replace_with_comma"]:
        return "replace_with_comma"
    if symbol in rules["keep"]:
        return "keep"
    if symbol in rules["sentence_end_pause"]:
        return "sentence_end_pause"
    for pause_type in ALLOWED_PAUSE_TYPES:
        if symbol in rules["pause"][pause_type]:
            return f"pause.{pause_type}"
    return "unknown"


def _match_symbol_at(text: str, position: int, symbols: list[str]) -> str | None:
    for symbol in symbols:
        if text.startswith(symbol, position):
            return symbol
    return None


def _is_unknown_symbol(char: str) -> bool:
    return unicodedata.category(char).startswith(("P", "S"))


def _stronger_pause(left: str, right: str) -> str:
    if PAUSE_PRIORITY.get(right, -1) > PAUSE_PRIORITY.get(left, -1):
        return right
    return left


def _read_required_json(
    path: Path, missing_code: str, invalid_code: str, result: dict
) -> object | None:
    if not path.exists():
        result["errors"].append(_error(missing_code, f"missing file: {path}"))
        return None
    try:
        return read_json(path)
    except json.JSONDecodeError as error:
        result["errors"].append(
            _error(invalid_code, f"{type(error).__name__}: {error}")
        )
    except Exception as error:
        result["errors"].append(
            _error("read_error", f"{type(error).__name__}: {error}", path=str(path))
        )
    return None


def _new_result(work_id: str) -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "work_id": work_id,
        "ok": False,
        "tts_units": [],
        "errors": [],
        "warnings": [],
        "summary": {
            "segments": 0,
            "tts_units": 0,
            "speak_units": 0,
            "pause_units": 0,
        },
    }


def _finish_result(result: dict) -> None:
    units = result.get("tts_units", [])
    result["ok"] = not result.get("errors")
    result["summary"] = {
        "segments": int(result.get("_segment_count") or 0),
        "tts_units": len(units),
        "speak_units": sum(1 for unit in units if unit.get("type") == "speak"),
        "pause_units": sum(1 for unit in units if unit.get("type") == "pause"),
    }


def _error(code: str, message: str, **extra: Any) -> dict:
    return {"level": "error", "code": code, "message": message, **extra}


def _warning(code: str, message: str, **extra: Any) -> dict:
    return {"level": "warning", "code": code, "message": message, **extra}
