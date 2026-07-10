from __future__ import annotations

import html
import copy
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pef2_engine.audio_preview import BudouxUnavailableError, apply_machine_breath, build_text_plain_for_tts
from pef2_engine.dictionary_loader import build_text_ssml, find_dictionary_matches
from pef2_engine.image_paths import ImagePathError, normalize_image_reference
from pef2_engine.io_utils import read_json, write_json
from pef2_engine.paths import resolve_named_file
from pef2_engine.ruby import strip_aozora_ruby
from pef2_engine import workspace_paths
from pef2_engine.tts_settings import resolve_tts_settings_with_sources
from version import VERSION


SCHEMA_VERSION = "processed-1"
REPORT_SCHEMA_VERSION = "processed-builder-report-1"
STEP6C_REPORT_SCHEMA_VERSION = "processed_report-1"
STEP6C_PARTIAL_SCHEMA_VERSION = "processed_partial_result-1"
IMAGE_MARKER_PATTERN = re.compile(r"^\[図[:：]\s*(?P<image_file>[^\]]+?)\s*\]$")
LEGACY_SUB_PATTERN = re.compile(
    r"<sub\s+alias=(?P<quote>['\"])(?P<alias>.*?)(?P=quote)>(?P<surface>.*?)</sub>"
)
LEGACY_PAUSE_PATTERN = re.compile(r"\s*\[(?:S|M|L)-PAUSE\]\s*")
LEGACY_PAUSE_MARKER_PATTERN = re.compile(r"\[(?:S|M|L)-PAUSE\]")
TRAILING_LEGACY_PAUSE_MARKER_PATTERN = re.compile(r"\s*(?P<marker>\[(?:S|M|L)-PAUSE\])\s*$")
INCOMPLETE_SUB_PATTERN = re.compile(r"<sub\b|</sub>")

DEFAULT_IMAGE_AUDIO_POLICY = "pause_only"
DEFAULT_IMAGE_PAUSE_TYPE = "M-PAUSE"
DEFAULT_INCLUDE_IMAGE_IN_AUDIO_TIMELINE = True
DEFAULT_INCLUDE_IMAGE_IN_HIGHLIGHT = True
ALLOWED_SOURCE_FORMATS = {"pef_legacy", "hd", "nlm", "nlm_simple", "nlm_fragments", "plain_text"}
ALLOWED_BLOCK_TYPES = {"text", "title", "paragraph", "image"}
TEXT_BLOCK_TYPES = {"text", "title", "paragraph"}
PLAIN_TTS_LENGTH_WARNING = 300

SYSTEM_FIXED_DICT_FILENAME = "★システム固定辞書.json"
STANDARD_ENGLISH_DICT_FILENAME = "★標準英単語辞書.json"
BREATH_RULES_FILENAME = "★汎用息継ぎ辞書.json"
SPEECH_SYMBOL_RULES_FILENAME = "★読み上げ記号ルール.json"
USER_DICT_FILENAME = "★ユーザ辞書.json"
PAUSE_AFTER_VALUES = {"S-PAUSE", "M-PAUSE", "L-PAUSE"}


def run_processed_workspace_partial(work_dir: Path, indexes: list[int]) -> dict:
    work_dir = Path(work_dir)
    report = _new_partial_report(work_dir, indexes)
    if not work_dir.exists():
        report["errors"].append(_error("missing_work_dir", f"work_dir does not exist: {work_dir}"))
        report["status"] = "failed"
        return report
    if not indexes:
        report["errors"].append(_error("missing_indexes", "indexes must not be empty"))
        report["status"] = "failed"
        return report

    target_path, target_filename = _resolve_partial_target(work_dir, report)
    if report["errors"]:
        report["status"] = "failed"
        return report
    report["target_file"] = target_filename

    pre_processed = _read_required_json(
        workspace_paths.pre_processed_path(work_dir),
        "missing_pre_processed",
        "invalid_pre_processed_json",
        report,
    )
    work_dictionary = _read_required_json(
        workspace_paths.work_dictionary_path(work_dir),
        "missing_work_dictionary",
        "invalid_work_dictionary_json",
        report,
    )
    finalize_report = _read_required_json(
        workspace_paths.dictionary_finalize_report_path(work_dir),
        "missing_dictionary_finalize_report",
        "invalid_dictionary_finalize_report_json",
        report,
    )
    target_processed = _read_required_json(
        target_path,
        "missing_target_processed",
        "invalid_target_processed_json",
        report,
    )

    if isinstance(finalize_report, dict):
        if finalize_report.get("status") != "success":
            report["errors"].append(
                _error(
                    "dictionary_finalize_not_success",
                    "01_dictionary_finalize_report.json status is not success",
                )
            )
        if finalize_report.get("errors"):
            report["errors"].append(
                _error(
                    "dictionary_finalize_has_errors",
                    "01_dictionary_finalize_report.json errors is not empty",
                )
            )

    _validate_work_dictionary(work_dictionary, report)
    _validate_step6c_pre_processed(pre_processed, report)
    _validate_partial_target_processed(target_processed, report)
    dictionary_entries, breath_rules = _load_step6c_dictionaries(work_dir, work_dictionary, report)
    if report["errors"]:
        report["status"] = "failed"
        return report

    pre_by_index = _segments_by_index(pre_processed["segments"], report, "pre_processed")
    target_by_index = _segments_by_index(target_processed["segments"], report, "target_processed")
    if report["errors"]:
        report["status"] = "failed"
        return report

    target_copy = copy.deepcopy(target_processed)
    target_positions = {
        segment.get("index"): position
        for position, segment in enumerate(target_copy.get("segments", []))
        if isinstance(segment, dict)
    }
    partial_dictionary_applied: list[dict] = []

    for index in indexes:
        if index not in target_by_index:
            report["errors"].append(_error("missing_target_index", "index is missing in target processed", index=index))
            continue
        if index not in pre_by_index:
            report["errors"].append(_error("missing_pre_processed_index", "index is missing in 00_pre_processed.json", index=index))
            continue
        pre_segment = pre_by_index[index]
        target_segment = target_by_index[index]
        if target_segment.get("block_type") == "image" or target_segment.get("is_image") is True:
            report["warnings"].append(_warning("image_segment_noop", "image segment was skipped", index=index))
            report["skipped_indexes"].append({"index": index, "reason": "image_segment"})
            continue

        temp_report = {"dictionary_applied": [], "breath_applied": [], "warnings": [], "errors": []}
        regenerated = _build_step6c_text_segment(pre_segment, dictionary_entries, breath_rules, temp_report)
        target_position = target_positions[index]
        next_segment = copy.deepcopy(target_copy["segments"][target_position])
        for key in ("audio", "dictionary_applied", "warnings"):
            next_segment[key] = regenerated.get(key)
        target_copy["segments"][target_position] = next_segment
        report["warnings"].extend(temp_report.get("warnings", []))
        report["errors"].extend(temp_report.get("errors", []))
        partial_dictionary_applied.extend(temp_report.get("dictionary_applied", []))
        report["updated_indexes"].append(index)

    if report["errors"]:
        report["status"] = "failed"
        return report

    write_json(target_path, target_copy)
    _update_step6c_report_after_partial(
        work_dir,
        target_filename,
        report["updated_indexes"],
        partial_dictionary_applied,
        report,
    )
    _update_partial_meta_status(work_dir, target_filename, report)
    report["status"] = "success"
    return report


def _new_partial_report(work_dir: Path, indexes: list[int]) -> dict:
    return {
        "schema_version": STEP6C_PARTIAL_SCHEMA_VERSION,
        "work_id": work_dir.name,
        "status": "pending",
        "target_file": "",
        "indexes": list(indexes),
        "updated_indexes": [],
        "skipped_indexes": [],
        "warnings": [],
        "errors": [],
    }


def _resolve_partial_target(work_dir: Path, report: dict) -> tuple[Path, str]:
    draft_path = workspace_paths.processed_draft_path(work_dir)
    final_path = workspace_paths.processed_final_path(work_dir)
    processed_path = workspace_paths.processed_path(work_dir)
    if draft_path.exists():
        return draft_path, workspace_paths.PROCESSED_DRAFT_FILENAME
    if final_path.exists():
        report["errors"].append(
            _error(
                "final_requires_reedit",
                "04_processed_final.json exists; create a draft before partial regeneration",
            )
        )
        return Path(), ""
    if processed_path.exists():
        return processed_path, workspace_paths.PROCESSED_JSON_FILENAME
    report["errors"].append(_error("missing_processed", "02_processed.json is missing"))
    return Path(), ""


def _validate_partial_target_processed(target_processed: object, report: dict) -> None:
    if target_processed is None:
        return
    if not isinstance(target_processed, dict):
        report["errors"].append(_error("invalid_target_processed", "target processed top-level must be object"))
        return
    if target_processed.get("schema_version") != SCHEMA_VERSION:
        report["errors"].append(_error("invalid_target_schema", "target processed schema_version must be processed-1"))
    segments = target_processed.get("segments")
    if not isinstance(segments, list):
        report["errors"].append(_error("invalid_target_segments", "target processed segments must be a list"))
        return
    seen_indexes: set[object] = set()
    for segment in segments:
        if not isinstance(segment, dict):
            report["errors"].append(_error("invalid_target_segment", "target processed segment must be object"))
            continue
        index = segment.get("index")
        if index in seen_indexes:
            report["errors"].append(_error("duplicate_target_index", "target processed index is duplicated", index=index))
        seen_indexes.add(index)


def _segments_by_index(segments: list[dict], report: dict, label: str) -> dict[int, dict]:
    by_index: dict[int, dict] = {}
    for segment in segments:
        if not isinstance(segment, dict):
            continue
        index = segment.get("index")
        if not isinstance(index, int):
            report["errors"].append(_error(f"invalid_{label}_index", "segment index must be integer"))
            continue
        if index in by_index:
            report["errors"].append(_error(f"duplicate_{label}_index", "segment index is duplicated", index=index))
            continue
        by_index[index] = segment
    return by_index


def _update_step6c_report_after_partial(
    work_dir: Path,
    target_filename: str,
    updated_indexes: list[int],
    dictionary_applied: list[dict],
    partial_report: dict,
) -> None:
    report_path = workspace_paths.processed_report_path(work_dir)
    if report_path.exists():
        try:
            step6c_report = read_json(report_path)
        except Exception as error:
            partial_report["warnings"].append(
                _warning("processed_report_read_error", f"{type(error).__name__}: {error}")
            )
            step6c_report = _new_step6c_report(work_dir.name)
    else:
        partial_report["warnings"].append(
            _warning("missing_processed_report", "02_processed_report.json was recreated with partial indexes only")
        )
        step6c_report = _new_step6c_report(work_dir.name)

    if not isinstance(step6c_report, dict):
        partial_report["warnings"].append(
            _warning("invalid_processed_report", "02_processed_report.json was recreated")
        )
        step6c_report = _new_step6c_report(work_dir.name)

    existing_dictionary_applied = step6c_report.get("dictionary_applied", [])
    if not isinstance(existing_dictionary_applied, list):
        existing_dictionary_applied = []
    updated_set = set(updated_indexes)
    step6c_report["dictionary_applied"] = [
        item
        for item in existing_dictionary_applied
        if isinstance(item, dict) and item.get("index") not in updated_set
    ] + dictionary_applied
    step6c_report.setdefault("schema_version", STEP6C_REPORT_SCHEMA_VERSION)
    step6c_report.setdefault("work_id", work_dir.name)
    step6c_report.setdefault("status", "success")
    step6c_report.setdefault("source", workspace_paths.PRE_PROCESSED_JSON_FILENAME)
    step6c_report.setdefault("work_dictionary", workspace_paths.WORK_DICTIONARY_FILENAME)
    step6c_report.setdefault("output", workspace_paths.PROCESSED_JSON_FILENAME)
    step6c_report["processed_source"] = target_filename
    step6c_report.setdefault("breath_applied", [])
    step6c_report.setdefault("warnings", [])
    step6c_report.setdefault("errors", [])
    step6c_report.setdefault("summary", {})
    step6c_report["summary"]["dictionary_applied_count"] = len(step6c_report.get("dictionary_applied", []))
    step6c_report["summary"]["breath_applied_count"] = len(step6c_report.get("breath_applied", []))
    step6c_report["summary"]["warning_count"] = len(step6c_report.get("warnings", []))
    step6c_report["summary"]["error_count"] = len(step6c_report.get("errors", []))
    write_json(report_path, step6c_report)


def _update_partial_meta_status(work_dir: Path, target_filename: str, report: dict) -> None:
    meta_path = work_dir / workspace_paths.WORK_META_FILENAME
    if not meta_path.exists():
        report["warnings"].append(_warning("missing_meta", "meta.json is missing; status was not updated"))
        return
    status = "draft_saved" if target_filename == workspace_paths.PROCESSED_DRAFT_FILENAME else "processed"
    try:
        workspace_paths.update_work_meta_status(work_dir, status)
    except Exception as error:
        report["warnings"].append(_warning("meta_update_failed", f"{type(error).__name__}: {error}"))


def run_processed_workspace_full(work_dir: Path) -> dict:
    work_dir = Path(work_dir)
    report_path = workspace_paths.processed_report_path(work_dir)
    processed_path = workspace_paths.processed_path(work_dir)
    processed, report = build_processed_workspace_full(work_dir)

    if processed is None:
        report["status"] = "failed"
        _finish_step6c_report(report)
        write_json(report_path, report)
        return report

    try:
        write_json(processed_path, processed)
        if (work_dir / workspace_paths.WORK_META_FILENAME).exists():
            workspace_paths.update_work_meta_status(work_dir, "processed")
        else:
            report["warnings"].append(
                _warning("missing_meta", "meta.json is missing; status was not updated")
            )
    except Exception as error:
        report["status"] = "failed"
        report["errors"].append(
            _error("write_or_meta_update_failed", f"{type(error).__name__}: {error}")
        )
        _finish_step6c_report(report)
        write_json(report_path, report)
        return report

    report["status"] = "success"
    _finish_step6c_report(report)
    write_json(report_path, report)
    return report


def build_processed_workspace_full(work_dir: Path) -> tuple[dict | None, dict]:
    work_dir = Path(work_dir)
    work_id = work_dir.name
    report = _new_step6c_report(work_id)
    if not work_dir.exists():
        report["errors"].append(_error("missing_work_dir", f"work_dir does not exist: {work_dir}"))
        _finish_step6c_report(report)
        return None, report
    if not work_dir.is_dir():
        report["errors"].append(_error("invalid_work_dir", f"work_dir is not a directory: {work_dir}"))
        _finish_step6c_report(report)
        return None, report

    meta = _read_optional_meta(work_dir, report)
    pre_processed = _read_required_json(
        workspace_paths.pre_processed_path(work_dir),
        "missing_pre_processed",
        "invalid_pre_processed_json",
        report,
    )
    work_dictionary = _read_required_json(
        workspace_paths.work_dictionary_path(work_dir),
        "missing_work_dictionary",
        "invalid_work_dictionary_json",
        report,
    )
    finalize_report = _read_required_json(
        workspace_paths.dictionary_finalize_report_path(work_dir),
        "missing_dictionary_finalize_report",
        "invalid_dictionary_finalize_report_json",
        report,
    )

    if isinstance(finalize_report, dict):
        if finalize_report.get("status") != "success":
            report["errors"].append(
                _error(
                    "dictionary_finalize_not_success",
                    "01_dictionary_finalize_report.json status is not success",
                )
            )
        if finalize_report.get("errors"):
            report["errors"].append(
                _error(
                    "dictionary_finalize_has_errors",
                    "01_dictionary_finalize_report.json errors is not empty",
                )
            )

    if work_dictionary == []:
        report["warnings"].append(_warning("empty_work_dictionary", "work_dictionary.json is []"))
    _validate_work_dictionary(work_dictionary, report)
    _validate_step6c_pre_processed(pre_processed, report)

    dictionary_entries, breath_rules = _load_step6c_dictionaries(work_dir, work_dictionary, report)
    if report["errors"]:
        _finish_step6c_report(report)
        return None, report

    tts_settings = resolve_tts_settings_with_sources(work_dir.parent, work_dir)
    breath_settings = {
        "choking_threshold": tts_settings["breath"]["choking_threshold"],
        "distance_threshold": tts_settings["breath"]["distance_threshold"],
    }
    report["breath_settings"] = tts_settings["breath"]
    processed_segments = _build_step6c_segments(
        pre_processed["segments"],
        dictionary_entries,
        breath_rules,
        _raw_source_format(pre_processed),
        report,
        breath_settings=breath_settings,
    )
    if report["errors"]:
        _finish_step6c_report(report)
        return None, report
    processed = {
        "schema_version": SCHEMA_VERSION,
        "version": VERSION,
        "work_id": work_id,
        "title": _resolve_step6c_title(work_dir, meta, pre_processed),
        "source": workspace_paths.PRE_PROCESSED_JSON_FILENAME,
        "work_dictionary": workspace_paths.WORK_DICTIONARY_FILENAME,
        "segments": processed_segments,
    }
    report["summary"]["segment_count"] = len(processed_segments)
    _finish_step6c_report(report)
    return processed, report


def _new_step6c_report(work_id: str) -> dict:
    return {
        "schema_version": STEP6C_REPORT_SCHEMA_VERSION,
        "work_id": work_id,
        "status": "pending",
        "source": workspace_paths.PRE_PROCESSED_JSON_FILENAME,
        "work_dictionary": workspace_paths.WORK_DICTIONARY_FILENAME,
        "output": workspace_paths.PROCESSED_JSON_FILENAME,
        "summary": {
            "segment_count": 0,
            "dictionary_applied_count": 0,
            "breath_applied_count": 0,
            "warning_count": 0,
            "error_count": 0,
        },
        "dictionary_applied": [],
        "breath_applied": [],
        "warnings": [],
        "errors": [],
    }


def _read_optional_meta(work_dir: Path, report: dict) -> dict:
    meta_path = work_dir / workspace_paths.WORK_META_FILENAME
    if not meta_path.exists():
        return {}
    try:
        meta = read_json(meta_path)
    except json.JSONDecodeError as error:
        report["errors"].append(_error("invalid_meta_json", f"{type(error).__name__}: {error}"))
        return {}
    except Exception as error:
        report["errors"].append(_error("meta_read_error", f"{type(error).__name__}: {error}"))
        return {}
    if not isinstance(meta, dict):
        report["errors"].append(_error("invalid_meta", "meta.json top-level must be object"))
        return {}
    return meta


def _read_required_json(path: Path, missing_code: str, invalid_code: str, report: dict) -> Any:
    if not path.exists():
        report["errors"].append(_error(missing_code, f"missing file: {path.name}"))
        return None
    try:
        return read_json(path)
    except json.JSONDecodeError as error:
        report["errors"].append(_error(invalid_code, f"{type(error).__name__}: {error}"))
    except Exception as error:
        report["errors"].append(_error(invalid_code, f"{type(error).__name__}: {error}"))
    return None


def _validate_work_dictionary(work_dictionary: object, report: dict) -> None:
    if work_dictionary is None:
        return
    if not isinstance(work_dictionary, list):
        report["errors"].append(_error("invalid_work_dictionary", "work_dictionary.json must be a list"))
        return
    for position, item in enumerate(work_dictionary):
        if not isinstance(item, dict):
            report["errors"].append(
                _error("invalid_work_dictionary_item", "work_dictionary item must be object", position=position)
            )
            continue
        if not str(item.get("単語原文") or "").strip():
            report["errors"].append(
                _error("missing_work_dictionary_word", "work_dictionary item has no 単語原文", position=position)
            )
        if not str(item.get("読み") or "").strip():
            report["errors"].append(
                _error("missing_work_dictionary_reading", "work_dictionary item has no 読み", position=position)
            )


def _validate_step6c_pre_processed(pre_processed: object, report: dict) -> None:
    if pre_processed is None:
        return
    if not isinstance(pre_processed, dict):
        report["errors"].append(_error("invalid_pre_processed", "00_pre_processed.json top-level must be object"))
        return
    segments = pre_processed.get("segments")
    if not isinstance(segments, list):
        report["errors"].append(_error("invalid_segments", "00_pre_processed.json segments must be a list"))
        return
    if not segments:
        report["errors"].append(_error("empty_segments", "00_pre_processed.json segments is empty"))
        return
    seen_indexes: set[object] = set()
    for position, segment in enumerate(segments):
        if not isinstance(segment, dict):
            report["errors"].append(
                _error("invalid_segment", "segment must be object", position=position)
            )
            continue
        index = segment.get("index")
        if index is None:
            report["errors"].append(_error("missing_index", "segment has no index", position=position))
        elif index in seen_indexes:
            report["errors"].append(_error("duplicate_index", "segment index is duplicated", index=index))
        seen_indexes.add(index)


def _load_step6c_dictionaries(
    work_dir: Path,
    work_dictionary: object,
    report: dict,
) -> tuple[list[dict], dict]:
    workspace_dir = work_dir.parent
    system_dir = workspace_dir / "dictionaries" / "system"
    user_dir = workspace_dir / "dictionaries" / "user"

    system_fixed_path = resolve_named_file(system_dir, SYSTEM_FIXED_DICT_FILENAME)
    standard_english_path = resolve_named_file(system_dir, STANDARD_ENGLISH_DICT_FILENAME)
    breath_rules_path = resolve_named_file(system_dir, BREATH_RULES_FILENAME)
    speech_symbol_rules_path = resolve_named_file(system_dir, SPEECH_SYMBOL_RULES_FILENAME)
    user_dict_path = resolve_named_file(user_dir, USER_DICT_FILENAME)

    entries: list[dict] = []
    for path, source_dictionary, priority in (
        (system_fixed_path, "system_fixed", 1),
        (standard_english_path, "standard_english", 2),
    ):
        data = _read_required_json(
            path,
            f"missing_{source_dictionary}",
            f"invalid_{source_dictionary}_json",
            report,
        )
        entries.extend(_dictionary_entries_from_data(data, source_dictionary, priority, report, required=True))

    entries.extend(
        _dictionary_entries_from_data(
            work_dictionary,
            "work_dictionary",
            3,
            report,
            required=True,
            strict_japanese_keys=True,
        )
    )

    if user_dict_path.exists():
        user_data = _read_required_json(
            user_dict_path,
            "missing_user_dictionary",
            "invalid_user_dictionary_json",
            report,
        )
        entries.extend(_dictionary_entries_from_data(user_data, "user_dictionary", 4, report, required=True))
    else:
        report["warnings"].append(_warning("missing_user_dictionary", "user dictionary was skipped"))

    breath_rules_data = _read_required_json(
        breath_rules_path,
        "missing_breath_rules",
        "invalid_breath_rules_json",
        report,
    )
    if breath_rules_data is None:
        breath_rules: dict = {}
    elif isinstance(breath_rules_data, dict):
        breath_rules = breath_rules_data
    else:
        report["errors"].append(_error("invalid_breath_rules", "breath rules must be object"))
        breath_rules = {}

    if speech_symbol_rules_path.exists():
        _read_required_json(
            speech_symbol_rules_path,
            "missing_speech_symbol_rules",
            "invalid_speech_symbol_rules_json",
            report,
        )
    else:
        report["warnings"].append(
            _warning("missing_speech_symbol_rules", "speech symbol rules were skipped")
        )

    return _merge_step6c_dictionary_entries(entries), breath_rules


def _dictionary_entries_from_data(
    data: object,
    source_dictionary: str,
    priority: int,
    report: dict,
    *,
    required: bool,
    strict_japanese_keys: bool = False,
) -> list[dict]:
    if data is None:
        return []
    if not isinstance(data, list):
        if required:
            report["errors"].append(
                _error("invalid_dictionary", f"{source_dictionary} dictionary must be a list")
            )
        return []

    entries: list[dict] = []
    for position, item in enumerate(data):
        if not isinstance(item, dict):
            if required:
                report["errors"].append(
                    _error("invalid_dictionary_item", "dictionary item must be object", source_dictionary=source_dictionary, position=position)
                )
            continue
        if strict_japanese_keys:
            word = str(item.get("単語原文") or "").strip()
            reading = str(item.get("読み") or "").strip()
        else:
            word = _first_dict_value(item, ("単語原文", "term", "word", "surface", "英単語"))
            reading = _first_dict_value(item, ("読み", "reading", "pronunciation", "カタカナ読み"))
        if not word or not reading:
            if required:
                report["errors"].append(
                    _error("invalid_dictionary_item", "dictionary item has no word or reading", source_dictionary=source_dictionary, position=position)
                )
            continue
        entries.append(
            {
                "word": word,
                "reading": reading,
                "priority": priority,
                "source": source_dictionary,
                "source_dictionary": source_dictionary,
            }
        )
    return entries


def _first_dict_value(item: dict, keys: tuple[str, ...]) -> str:
    for key in keys:
        value = item.get(key)
        if value is not None:
            text = str(value).strip()
            if text:
                return text
    return ""


def _merge_step6c_dictionary_entries(entries: list[dict]) -> list[dict]:
    entries_by_word: dict[str, dict] = {}
    for entry in entries:
        word = entry.get("word")
        if not word:
            continue
        current = entries_by_word.get(word)
        if current is None or entry.get("priority", 0) >= current.get("priority", 0):
            entries_by_word[word] = entry
    return sorted(entries_by_word.values(), key=lambda item: len(str(item.get("word") or "")), reverse=True)


def _build_step6c_segments(
    segments: list[dict],
    dictionary_entries: list[dict],
    breath_rules: dict,
    source_format: str,
    report: dict,
    *,
    breath_settings: dict | None = None,
) -> list[dict]:
    processed_segments: list[dict] = []
    for segment in segments:
        if segment.get("block_type") == "image" or segment.get("is_image") is True:
            processed_segments.append(_build_step6c_image_segment(segment))
        else:
            processed_segments.append(
                _build_step6c_text_segment(
                    segment,
                    dictionary_entries,
                    breath_rules,
                    report,
                    breath_settings=breath_settings,
                )
            )

    if source_format != "pef_legacy":
        for position, segment in enumerate(segments[:-1]):
            next_segment = segments[position + 1]
            if next_segment.get("para_start") is True and processed_segments[position].get("block_type") != "image":
                if _set_pause_after(
                    processed_segments[position],
                    "M-PAUSE",
                    "paragraph_boundary",
                    report,
                ):
                    applied = {
                        "index": segment.get("index"),
                        "type": "pause_after",
                        "pause_after": "M-PAUSE",
                        "reason": "paragraph_boundary",
                    }
                    report["breath_applied"].append(applied)
    return processed_segments


def _build_step6c_text_segment(
    segment: dict,
    dictionary_entries: list[dict],
    breath_rules: dict,
    report: dict,
    *,
    breath_settings: dict | None = None,
) -> dict:
    index = segment.get("index")
    block_type = str(segment.get("block_type") or "text")
    display = str(segment.get("display") or "")
    audio_base = str(segment.get("audio_seed") or display)
    warnings: list[dict] = []
    dictionary_applied: list[dict] = []
    audio_base, legacy_pause_after, legacy_pause_warnings = _normalize_trailing_legacy_pause_marker(
        audio_base,
        index,
    )
    warnings.extend(legacy_pause_warnings)

    breath_applied: list[dict] = []
    machine_breath_applied: list[dict] = []
    if LEGACY_SUB_PATTERN.search(audio_base) or INCOMPLETE_SUB_PATTERN.search(audio_base):
        audio = audio_base
        warnings.append(
            _warning(
                "existing_sub_respected",
                "audio_seed contains sub tag; dictionary replacement was skipped",
                index=index,
            )
        )
    else:
        text_for_breath = audio_base
        if breath_settings is not None:
            try:
                text_for_breath, breath_applied = apply_machine_breath(
                    audio_base,
                    breath_rules,
                    block_type == "title",
                    breath_settings=breath_settings,
                )
            except BudouxUnavailableError as error:
                report["errors"].append(
                    _error(
                        "budoux_unavailable",
                        "BudouX is required for automatic breath insertion; install budoux from requirements.txt",
                        index=index,
                        detail=str(error),
                    )
                )
                text_for_breath = audio_base
                breath_applied = []
        machine_breath_applied = [
            item for item in breath_applied if item.get("type") == "machine_inserted"
        ]
        if machine_breath_applied:
            report.setdefault("breath_applied", []).extend(
                [{**item, "index": index} for item in machine_breath_applied]
            )
        matches = find_dictionary_matches(text_for_breath, dictionary_entries)
        audio, substitutions = build_text_ssml(text_for_breath, matches)
        dictionary_applied = [
            {
                "index": index,
                "word": item.get("word", ""),
                "reading": item.get("reading", ""),
                "source_dictionary": item.get("source", "dictionary"),
            }
            for item in substitutions
        ]
        report["dictionary_applied"].extend(dictionary_applied)

    if warnings:
        report["warnings"].extend(warnings)

    processed = {
        "index": index,
        "block_type": block_type,
        "display": display,
        "audio": audio,
        "dictionary_applied": bool(dictionary_applied),
        "breath_applied": any(item.get("type") == "machine_inserted" for item in breath_applied),
        "warnings": warnings,
    }
    copy_structure_flags(processed, segment)
    source_pause_after = _normalize_pause_after_value(segment.get("pause_after"))
    if source_pause_after and not legacy_pause_after:
        _set_pause_after(processed, source_pause_after, "source_pause_after", report)
    if legacy_pause_after:
        if source_pause_after and source_pause_after != legacy_pause_after:
            report["warnings"].append(
                _warning(
                    "pause_after_conflict",
                    "source pause_after conflicted with legacy pause marker; legacy marker was used",
                    index=index,
                    existing_pause_after=source_pause_after,
                    requested_pause_after=legacy_pause_after,
                    reason="legacy_pause_marker",
                )
            )
        _set_pause_after(processed, legacy_pause_after, "legacy_pause_marker", report)
    return processed


def _normalize_trailing_legacy_pause_marker(audio: str, index: object) -> tuple[str, str, list[dict]]:
    matches = list(LEGACY_PAUSE_MARKER_PATTERN.finditer(audio))
    if not matches:
        return audio, "", []

    last_match = matches[-1]
    warnings: list[dict] = []
    if len(matches) != 1 or audio[last_match.end():].strip():
        warnings.append(
            _warning(
                "legacy_pause_marker_not_normalized",
                "audio_seed contains non-terminal or multiple legacy pause markers",
                index=index,
            )
        )
        return audio, "", warnings

    marker_match = TRAILING_LEGACY_PAUSE_MARKER_PATTERN.search(audio)
    if marker_match is None:
        warnings.append(
            _warning(
                "legacy_pause_marker_not_normalized",
                "audio_seed contains legacy pause marker but it was not terminal",
                index=index,
            )
        )
        return audio, "", warnings

    marker = marker_match.group("marker").strip("[]")
    cleaned_audio = audio[: marker_match.start()].rstrip()
    warnings.append(
        _warning(
            "legacy_pause_marker_structured",
            "terminal legacy pause marker was moved from audio to pause_after",
            index=index,
            pause_after=marker,
        )
    )
    return cleaned_audio, marker, warnings


def _normalize_pause_after_value(value: object) -> str:
    if isinstance(value, dict):
        value = value.get("type")
    if not isinstance(value, str):
        return ""
    pause_after = value.strip()
    return pause_after if pause_after in PAUSE_AFTER_VALUES else ""


def _set_pause_after(segment: dict, pause_after: str, reason: str, report: dict) -> bool:
    current = _normalize_pause_after_value(segment.get("pause_after"))
    if current:
        if current != pause_after:
            report["warnings"].append(
                _warning(
                    "pause_after_conflict",
                    "existing pause_after conflicted with new pause_after; existing value was kept",
                    index=segment.get("index"),
                    existing_pause_after=current,
                    requested_pause_after=pause_after,
                    reason=reason,
                )
            )
        return False
    segment["pause_after"] = {"type": pause_after}
    return True


def _build_step6c_image_segment(segment: dict) -> dict:
    processed = {
        "index": segment.get("index"),
        "block_type": "image",
        "is_image": True,
        "image_file": segment.get("image_file", ""),
        "display": "",
        "audio": "",
        "audio_policy": "pause_only",
        "pause_type": segment.get("pause_type") or "M-PAUSE",
        "sync": {
            "include_in_audio_timeline": True,
            "include_in_highlight": True,
        },
        "warnings": [],
    }
    copy_structure_flags(processed, segment)
    return processed


def _resolve_step6c_title(work_dir: Path, meta: dict, pre_processed: dict) -> str:
    meta_title = str(meta.get("title") or "").strip()
    if meta_title:
        return meta_title
    pre_processed_title = str(pre_processed.get("title") or "").strip()
    if pre_processed_title:
        return pre_processed_title
    return work_dir.name


def _finish_step6c_report(report: dict) -> None:
    report["summary"]["dictionary_applied_count"] = len(report.get("dictionary_applied", []))
    report["summary"]["breath_applied_count"] = len(report.get("breath_applied", []))
    report["summary"]["warning_count"] = len(report.get("warnings", []))
    report["summary"]["error_count"] = len(report.get("errors", []))


def build_processed_from_pre_processed(
    pre_processed: dict,
    dictionary_entries: list[dict],
    breath_rules: dict,
    *,
    source_id: str | None = None,
    user_audio_overrides: dict | None = None,
    generated_at: str | None = None,
    dictionary_profile: dict | None = None,
    breath_profile: dict | None = None,
    breath_settings: dict | None = None,
) -> tuple[dict | None, dict]:
    generated_at = generated_at or _utc_now()
    report = _new_report(pre_processed, source_id, generated_at)
    errors = validate_pre_processed_input(pre_processed, report)

    if errors:
        report["errors"].extend(errors)
        report["status"] = "failed"
        report["summary"] = _summary(pre_processed, None)
        return None, report

    source_format = _processed_source_format(pre_processed)
    resolved_source_id = _resolve_source_id(pre_processed, source_id)
    title, title_warnings = resolve_title(pre_processed, resolved_source_id)
    report["warnings"].extend(title_warnings)
    report["source_id"] = resolved_source_id

    processed_segments: list[dict] = []
    for segment in pre_processed.get("segments", []):
        segment_result, segment_report = build_processed_segment(
            segment,
            source_format,
            dictionary_entries,
            breath_rules,
            user_audio_overrides=user_audio_overrides,
            breath_settings=breath_settings,
        )
        processed_segments.append(segment_result)
        report["segments"].append(segment_report)
        report["warnings"].extend(segment_report.get("warnings", []))
        report["errors"].extend(segment_report.get("errors", []))

    if report["errors"]:
        report["status"] = "failed"
        report["summary"] = _summary(pre_processed, processed_segments)
        return None, report

    processed = {
        "schema_version": SCHEMA_VERSION,
        "title": title,
        "version": VERSION,
        "source_format": source_format,
        "source_id": resolved_source_id,
        "builder": {
            "generated_at": generated_at,
            "dictionary_profile": dictionary_profile or build_dictionary_profile(dictionary_entries),
            "breath_profile": breath_profile or build_breath_profile(breath_rules),
        },
        "segments": processed_segments,
    }

    report["status"] = "ok"
    report["summary"] = _summary(pre_processed, processed_segments)
    return processed, report


def validate_pre_processed_input(pre_processed: object, report: dict | None = None) -> list[dict]:
    errors: list[dict] = []
    if not isinstance(pre_processed, dict):
        return [_error("invalid_pre_processed", "pre_processed input must be a JSON object")]

    for field in ("schema_version", "source", "segments"):
        if field not in pre_processed:
            errors.append(_error("missing_top_level_field", f"missing top-level field: {field}"))

    schema_version = str(pre_processed.get("schema_version") or "")
    if not schema_version.startswith("pef2-pre-processed"):
        errors.append(
            _error(
                "invalid_pre_processed_schema",
                "processed builder accepts only pre_processed.json input",
            )
        )

    source_format = _raw_source_format(pre_processed)
    if source_format and source_format not in ALLOWED_SOURCE_FORMATS:
        errors.append(_error("unknown_source_format", f"unknown source_format: {source_format}"))

    segments = pre_processed.get("segments")
    if not isinstance(segments, list):
        errors.append(_error("invalid_segments", "segments must be a list"))
        return errors
    if not segments:
        errors.append(_error("empty_segments", "segments is empty"))
        return errors

    indexes: list[int] = []
    seen_indexes: set[int] = set()
    source_indexes: list[int] = []
    for position, segment in enumerate(segments):
        if not isinstance(segment, dict):
            errors.append(_error("invalid_segment", f"segment at position {position} is not an object"))
            continue

        index = segment.get("index")
        if not isinstance(index, int):
            errors.append(_error("invalid_index", f"segment at position {position} has no integer index"))
        elif index in seen_indexes:
            errors.append(_error("duplicate_index", f"duplicate segment index: {index}", index=index))
        else:
            seen_indexes.add(index)
            indexes.append(index)

        block_type = str(segment.get("block_type") or "")
        if block_type not in ALLOWED_BLOCK_TYPES:
            errors.append(_error("unknown_block_type", f"unknown block_type: {block_type}", index=index))
            continue

        if block_type == "image" or segment.get("is_image") is True:
            image_file = str(segment.get("image_file") or "").strip()
            if not image_file:
                errors.append(_error("missing_image_file", "image segment has no image_file", index=index))
            else:
                try:
                    normalize_image_reference(image_file)
                except ImagePathError:
                    errors.append(_error("invalid_image_file", "image segment has invalid image_file", index=index))
        elif block_type in TEXT_BLOCK_TYPES:
            if not str(segment.get("display") or "").strip():
                errors.append(_error("empty_display", "text/title segment display is empty", index=index))

        source_index = segment.get("source_index")
        if isinstance(source_index, int):
            source_indexes.append(source_index)

    expected_indexes = list(range(len(segments)))
    if sorted(indexes) != expected_indexes:
        errors.append(
            _error(
                "index_gap",
                f"segment indexes must be contiguous from 0: expected {expected_indexes}, got {sorted(indexes)}",
            )
        )

    if report is not None:
        report["warnings"].extend(_source_index_warnings(segments))
    if len(source_indexes) != len(set(source_indexes)) and report is not None:
        report["warnings"].append(_warning("duplicate_source_index", "source_index is duplicated"))

    return errors


def build_processed_segment(
    segment: dict,
    source_format: str,
    dictionary_entries: list[dict],
    breath_rules: dict,
    *,
    user_audio_overrides: dict | None = None,
    breath_settings: dict | None = None,
) -> tuple[dict, dict]:
    index = segment.get("index")
    source_index = segment.get("source_index")
    segment_report = _segment_report(index, source_index)

    if segment.get("block_type") == "image" or segment.get("is_image") is True:
        processed = build_image_segment(index, source_index, str(segment.get("image_file") or ""), segment)
        segment_report["warnings"].append(
            _warning(
                "image_file_not_verified",
                "image_file exists in JSON but actual file existence is not verified by processed builder",
                index=index,
                source_index=source_index,
            )
        )
        _finish_segment_report(segment_report)
        return processed, segment_report

    display = str(segment.get("display") or "")
    base_audio, base_source = resolve_audio_base(segment, user_audio_overrides)
    normalized_base, legacy_subs, base_warnings = normalize_audio_base(base_audio)
    normalized_base = strip_aozora_ruby(normalized_base)
    is_title = segment.get("block_type") == "title"

    try:
        text_for_breath, breath_applied = apply_machine_breath(
            normalized_base,
            breath_rules,
            is_title,
            breath_settings=breath_settings,
        )
    except BudouxUnavailableError as error:
        segment_report["errors"].append(
            _error(
                "budoux_unavailable",
                "BudouX is required for automatic breath insertion; install budoux from requirements.txt",
                index=index,
                source_index=source_index,
                detail=str(error),
            )
        )
        _finish_segment_report(segment_report)
        processed = {
            "index": index,
            "source_index": source_index,
            "block_type": "title" if is_title else "text",
            "is_image": False,
            "display": display,
            "audio": normalized_base,
            "text_ssml": normalized_base,
            "text_plain_for_tts": normalized_base,
            "sync": _text_sync(segment),
        }
        copy_structure_flags(processed, segment)
        return processed, segment_report
    dictionary_matches = find_dictionary_matches(text_for_breath, dictionary_entries)
    combined_matches = merge_legacy_subs(text_for_breath, dictionary_matches, legacy_subs)
    text_ssml, substitutions = build_text_ssml(text_for_breath, combined_matches)
    text_plain_for_tts, replacements = build_text_plain_for_tts(text_for_breath, combined_matches)
    dictionary_applied = dictionary_applied_report(substitutions, replacements)

    processed = {
        "index": index,
        "source_index": source_index,
        "block_type": "title" if is_title else "text",
        "is_image": False,
        "display": display,
        "audio": text_ssml,
        "text_ssml": text_ssml,
        "text_plain_for_tts": text_plain_for_tts,
        "sync": _text_sync(segment),
    }
    copy_structure_flags(processed, segment)

    if segment.get("lower_display"):
        processed["lower_display"] = segment.get("lower_display")
    lower_audio = segment.get("lower_audio")
    if lower_audio:
        processed["lower_audio"] = lower_audio
    notes = _short_notes(segment.get("notes", []))
    if notes:
        processed["notes"] = notes
    flags = build_text_flags(dictionary_applied, breath_applied, base_warnings)
    if any(flags.values()):
        processed["flags"] = flags

    segment_report["dictionary_applied"] = dictionary_applied
    segment_report["breath_applied"] = breath_applied
    segment_report["notes"].append(f"audio_base: {base_source}")
    segment_report["warnings"].extend(
        _segment_warnings(
            segment,
            source_format,
            display,
            base_audio,
            text_ssml,
            text_plain_for_tts,
            base_warnings,
        )
    )
    _finish_segment_report(segment_report)
    return processed, segment_report


def resolve_title(pre_processed: dict, source_id: str) -> tuple[str, list[dict]]:
    title = str(pre_processed.get("title") or "").strip()
    if title:
        return title, []
    if source_id:
        return source_id, [
            _warning("missing_top_level_title", "pre_processed top-level title is missing"),
            _warning("title_fallback_to_source_id", "processed title uses source_id fallback"),
        ]
    return "untitled", [
        _warning("missing_top_level_title", "pre_processed top-level title is missing"),
        _warning("title_fallback_to_untitled", "processed title uses untitled fallback"),
    ]


def resolve_audio_base(segment: dict, user_audio_overrides: dict | None) -> tuple[str, str]:
    override = _override_for_segment(segment, user_audio_overrides)
    if override is not None:
        return override, "user_audio_overrides"
    audio_seed = str(segment.get("audio_seed") or "")
    if audio_seed:
        return audio_seed, "audio_seed"
    return str(segment.get("display") or ""), "display"


def normalize_audio_base(text: str) -> tuple[str, list[dict], list[dict]]:
    warnings: list[dict] = []
    legacy_subs: list[dict] = []

    def replace_sub(match: re.Match) -> str:
        surface = html.unescape(match.group("surface"))
        alias = html.unescape(match.group("alias"))
        legacy_subs.append({"word": surface, "reading": alias})
        return surface

    if LEGACY_SUB_PATTERN.search(text):
        warnings.append(_warning("legacy_ssml_detected", "audio_seed contains legacy SSML sub tag"))
        text = LEGACY_SUB_PATTERN.sub(replace_sub, text)
    elif INCOMPLETE_SUB_PATTERN.search(text):
        warnings.append(_warning("incomplete_ssml_tag", "audio_seed contains incomplete SSML-like tag"))

    if LEGACY_PAUSE_PATTERN.search(text):
        warnings.append(_warning("legacy_pause_marker_detected", "audio_seed contains legacy pause marker"))
        text = LEGACY_PAUSE_PATTERN.sub("、", text)

    return _clean_audio_text(text), legacy_subs, warnings


def merge_legacy_subs(text: str, dictionary_matches: list[dict], legacy_subs: list[dict]) -> list[dict]:
    matches = list(dictionary_matches)
    occupied = [
        (item["match_start"], item["match_end"])
        for item in matches
        if isinstance(item.get("match_start"), int) and isinstance(item.get("match_end"), int)
    ]
    for legacy in legacy_subs:
        word = legacy.get("word", "")
        reading = legacy.get("reading", "")
        if not word or not reading:
            continue
        start = text.find(word)
        while start != -1:
            end = start + len(word)
            if not _overlaps(start, end, occupied):
                matches.append(
                    {
                        "index": len(matches),
                        "word": word,
                        "reading": reading,
                        "match_start": start,
                        "match_end": end,
                        "source_file": "audio_seed",
                        "priority": 0,
                        "source": "legacy_audio_seed",
                        "voicevox_katakana": False,
                    }
                )
                occupied.append((start, end))
            start = text.find(word, start + 1)
    matches.sort(key=lambda item: (item["match_start"], -len(item["word"])))
    for index, match in enumerate(matches):
        match["index"] = index
    return matches


def dictionary_applied_report(substitutions: list[dict], replacements: list[dict]) -> list[dict]:
    plain_by_word = {item.get("word"): item.get("replacement") for item in replacements}
    applied: list[dict] = []
    for item in substitutions:
        word = item.get("word", "")
        applied.append(
            {
                "surface": word,
                "reading": item.get("reading", ""),
                "plain_reading": plain_by_word.get(word, item.get("reading", "")),
                "source": item.get("source", "dictionary"),
                "source_file": item.get("source_file", ""),
                "method": "sub_alias",
                "confirmed": item.get("source") != "legacy_audio_seed",
            }
        )
    return applied


def build_dictionary_profile(dictionary_entries: list[dict]) -> dict:
    source_files: list[str] = []
    seen_source_files: set[str] = set()
    for item in sorted(
        dictionary_entries,
        key=lambda entry: (entry.get("priority", 0), str(entry.get("source_file") or "")),
    ):
        source_file = str(item.get("source_file") or "")
        if source_file and source_file not in seen_source_files:
            source_files.append(source_file)
            seen_source_files.add(source_file)
    return {
        "entry_count": len(dictionary_entries),
        "source_files": source_files,
        "priority_order": [
            "system_fixed",
            "standard_english",
            "work",
            "user",
        ],
    }


def build_breath_profile(breath_rules: dict) -> dict:
    return {
        "rule_keys": sorted(breath_rules.keys()) if isinstance(breath_rules, dict) else [],
        "rule_count": len(breath_rules) if isinstance(breath_rules, dict) else 0,
    }


def parse_image_marker(line: str) -> str:
    match = IMAGE_MARKER_PATTERN.fullmatch(line.strip())
    if not match:
        return ""
    image_file = match.group("image_file").strip()
    if not image_file:
        return ""
    try:
        return normalize_image_reference(image_file)
    except ImagePathError:
        return ""


def build_image_segment(index: int, source_index: object, image_file: str, source_segment: dict | None = None) -> dict:
    source_segment = source_segment or {}
    image_file = normalize_image_reference(image_file)
    processed = {
        "index": index,
        "source_index": source_index,
        "block_type": "image",
        "is_image": True,
        "image_file": image_file,
        "display": "",
        "audio": "",
        "text_ssml": "",
        "text_plain_for_tts": "",
        "audio_policy": source_segment.get("audio_policy") or DEFAULT_IMAGE_AUDIO_POLICY,
        "pause_type": source_segment.get("pause_type") or DEFAULT_IMAGE_PAUSE_TYPE,
        "sync": _image_sync(source_segment),
    }
    copy_structure_flags(processed, source_segment)
    return processed


def build_text_flags(dictionary_applied: list[dict], breath_applied: list[dict], warnings: list[dict]) -> dict:
    return {
        "needs_review": bool(dictionary_applied or warnings),
        "has_dictionary_applied": bool(dictionary_applied),
        "has_machine_breath": any(item.get("type") == "machine_inserted" for item in breath_applied),
        "tts_warning": bool(warnings),
    }


def build_processed_from_input(
    input_path: Path,
    dictionary_entries: list[dict],
    breath_rules: dict,
) -> dict:
    raise ValueError("plain txt input is not accepted by processed builder; use pre_processed.json")


def _new_report(pre_processed: object, source_id: str | None, generated_at: str) -> dict:
    first_title = None
    if isinstance(pre_processed, dict):
        first_title = _first_title_segment(pre_processed.get("segments", []))
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "source_id": source_id or _resolve_source_id(pre_processed, None),
        "generated_at": generated_at,
        "status": "pending",
        "errors": [],
        "warnings": [],
        "summary": {},
        "segments": [],
        "first_title_segment": first_title,
    }


def _segment_report(index: object, source_index: object) -> dict:
    return {
        "index": index,
        "source_index": source_index,
        "level": "ok",
        "codes": [],
        "messages": [],
        "errors": [],
        "warnings": [],
        "dictionary_applied": [],
        "breath_applied": [],
        "notes": [],
    }


def _finish_segment_report(segment_report: dict) -> None:
    codes: list[str] = []
    messages: list[str] = []
    level = "ok"
    if segment_report.get("errors"):
        level = "error"
        codes.extend(item["code"] for item in segment_report["errors"])
        messages.extend(item["message"] for item in segment_report["errors"])
    elif segment_report.get("warnings"):
        level = "warning"
        codes.extend(item["code"] for item in segment_report["warnings"])
        messages.extend(item["message"] for item in segment_report["warnings"])
    segment_report["level"] = level
    segment_report["codes"] = codes
    segment_report["messages"] = messages


def _segment_warnings(
    segment: dict,
    source_format: str,
    display: str,
    base_audio: str,
    text_ssml: str,
    text_plain_for_tts: str,
    base_warnings: list[dict],
) -> list[dict]:
    index = segment.get("index")
    source_index = segment.get("source_index")
    warnings = [
        {**warning, "index": index, "source_index": source_index}
        for warning in base_warnings
    ]
    if not str(segment.get("audio_seed") or ""):
        warnings.append(_warning("missing_audio_seed", "audio_seed is missing", index=index, source_index=source_index))
    if _large_text_divergence(display, base_audio):
        warnings.append(_warning("display_audio_divergence", "display and audio base may diverge", index=index, source_index=source_index))
    if source_format in {"nlm", "nlm_fragments"} and segment.get("notes"):
        warnings.append(_warning("nlm_notes_present", "NLM notes are present", index=index, source_index=source_index))
    if source_format in {"nlm", "nlm_fragments"} and _large_text_divergence(display, base_audio):
        warnings.append(_warning("nlm_audio_seed_mismatch", "NLM audio_seed may not correspond to display", index=index, source_index=source_index))
    if source_format == "hd" and not str(segment.get("lower_display") or ""):
        warnings.append(_warning("hd_lower_display_missing", "HD segment has no lower_display", index=index, source_index=source_index))
    if len(text_plain_for_tts) > PLAIN_TTS_LENGTH_WARNING:
        warnings.append(_warning("text_plain_for_tts_too_long", "text_plain_for_tts is long", index=index, source_index=source_index))
    if _has_incomplete_generated_tag(text_ssml):
        warnings.append(_warning("incomplete_ssml_tag", "text_ssml may contain incomplete tag", index=index, source_index=source_index))
    return warnings


def _source_index_warnings(segments: list[dict]) -> list[dict]:
    warnings: list[dict] = []
    seen: set[int] = set()
    for segment in segments:
        source_index = segment.get("source_index")
        index = segment.get("index")
        if source_index is None:
            warnings.append(_warning("missing_source_index", "source_index is missing", index=index))
        elif isinstance(source_index, int):
            if source_index in seen:
                warnings.append(_warning("duplicate_source_index", "source_index is duplicated", index=index, source_index=source_index))
            seen.add(source_index)
        else:
            warnings.append(_warning("invalid_source_index", "source_index is not an integer", index=index))
    return warnings


def _processed_source_format(pre_processed: dict) -> str:
    source_format = _raw_source_format(pre_processed)
    if source_format == "nlm_simple":
        return "nlm"
    return source_format


def _raw_source_format(pre_processed: dict) -> str:
    source = pre_processed.get("source") if isinstance(pre_processed, dict) else {}
    if isinstance(source, dict):
        return str(source.get("source_format") or "")
    return ""


def _resolve_source_id(pre_processed: object, source_id: str | None) -> str:
    if source_id:
        return source_id
    if not isinstance(pre_processed, dict):
        return ""
    source = pre_processed.get("source") if isinstance(pre_processed.get("source"), dict) else {}
    explicit = str(source.get("source_id") or pre_processed.get("source_id") or "").strip()
    if explicit:
        return explicit
    source_path = str(source.get("source_path") or "").strip()
    if source_path:
        return Path(source_path).stem
    return ""


def _first_title_segment(segments: object) -> dict | None:
    if not isinstance(segments, list):
        return None
    for segment in segments:
        if isinstance(segment, dict) and segment.get("block_type") == "title":
            return {
                "index": segment.get("index"),
                "source_index": segment.get("source_index"),
                "display": segment.get("display", ""),
            }
    return None


def _summary(pre_processed: object, processed_segments: list[dict] | None) -> dict:
    input_segments = pre_processed.get("segments", []) if isinstance(pre_processed, dict) else []
    output_segments = processed_segments or []
    return {
        "input_segment_count": len(input_segments) if isinstance(input_segments, list) else 0,
        "output_segment_count": len(output_segments),
        "image_segment_count": sum(1 for item in output_segments if item.get("block_type") == "image"),
        "text_segment_count": sum(1 for item in output_segments if item.get("block_type") in {"text", "title"}),
    }


def _override_for_segment(segment: dict, user_audio_overrides: dict | None) -> str | None:
    if not isinstance(user_audio_overrides, dict):
        return None
    index = segment.get("index")
    source_index = segment.get("source_index")
    for container_key, value in (
        ("by_index", index),
        ("by_source_index", source_index),
    ):
        container = user_audio_overrides.get(container_key)
        if isinstance(container, dict):
            override = _lookup_override(container, value)
            if override is not None:
                return override
    return _lookup_override(user_audio_overrides, index) or _lookup_override(user_audio_overrides, source_index)


def _lookup_override(overrides: dict, key: object) -> str | None:
    if key is None:
        return None
    if key in overrides and overrides[key] is not None:
        return str(overrides[key])
    text_key = str(key)
    if text_key in overrides and overrides[text_key] is not None:
        return str(overrides[text_key])
    return None


def _image_sync(segment: dict) -> dict:
    sync = segment.get("sync") if isinstance(segment.get("sync"), dict) else {}
    return {
        "include_in_audio_timeline": bool(sync.get("include_in_audio_timeline", DEFAULT_INCLUDE_IMAGE_IN_AUDIO_TIMELINE)),
        "include_in_highlight": bool(sync.get("include_in_highlight", DEFAULT_INCLUDE_IMAGE_IN_HIGHLIGHT)),
    }


def _text_sync(segment: dict) -> dict:
    sync = segment.get("sync") if isinstance(segment.get("sync"), dict) else {}
    return {
        "include_in_audio_timeline": bool(sync.get("include_in_audio_timeline", True)),
        "include_in_highlight": bool(sync.get("include_in_highlight", True)),
    }


def copy_structure_flags(target: dict, source: dict) -> None:
    for key in ("para_start", "line_start"):
        if key in source:
            target[key] = bool(source.get(key))


def _short_notes(notes: object) -> list[str]:
    if isinstance(notes, list):
        return [str(note)[:120] for note in notes if str(note).strip()][:3]
    if isinstance(notes, str) and notes.strip():
        return [notes[:120]]
    return []


def _clean_audio_text(text: str) -> str:
    return re.sub(r"、{2,}", "、", text).strip()


def _large_text_divergence(display: str, audio: str) -> bool:
    if not display or not audio:
        return False
    display_plain = re.sub(r"\s+", "", display)
    audio_plain = re.sub(r"\s+", "", normalize_audio_base(audio)[0])
    if not display_plain or not audio_plain:
        return False
    shared = sum(1 for char in display_plain if char in audio_plain)
    ratio = shared / max(len(display_plain), len(audio_plain))
    return ratio < 0.45


def _has_incomplete_generated_tag(text: str) -> bool:
    return text.count("<sub") != text.count("</sub>")


def _overlaps(start: int, end: int, ranges: list[tuple[int, int]]) -> bool:
    return any(start < existing_end and end > existing_start for existing_start, existing_end in ranges)


def _error(code: str, message: str, **extra: Any) -> dict:
    return {"level": "error", "code": code, "message": message, **extra}


def _warning(code: str, message: str, **extra: Any) -> dict:
    return {"level": "warning", "code": code, "message": message, **extra}


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
