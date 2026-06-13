from __future__ import annotations

import copy
import html
import json
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from pef2_engine import workspace_paths
from pef2_engine.io_utils import read_json, write_json


REPORT_SCHEMA_VERSION = "processed_editing_report-1"
EDITING_SESSION_SCHEMA_VERSION = "editing_session-1"
PROCESSED_SCHEMA_VERSION = "processed-1"
JST = timezone(timedelta(hours=9))
SUB_ALIAS_PATTERN = re.compile(
    r"<sub\s+alias=(?P<quote>['\"])(?P<alias>.*?)(?P=quote)>(?P<surface>.*?)</sub>"
)
EDIT_RUBY_PATTERN = re.compile(r"(?P<surface>[一-龯々〆ヵヶ]+)《(?P<reading>[^《》]+?)》")
PAUSE_MARKER_PATTERN = re.compile(r"\s*\[(?:S|M|L)-PAUSE\]\s*")
SYMBOL_ONLY_PATTERN = re.compile(r"^[・、。？！「」『』（）()【】［］\[\]…—ー・\s]*$")
MARKED_RUBY_MARKERS = ("｜", "|")


def create_processed_draft(work_dir: Path) -> dict:
    work_dir = Path(work_dir)
    report = _new_report(
        "create_draft",
        work_dir,
        workspace_paths.PROCESSED_JSON_FILENAME,
        workspace_paths.PROCESSED_DRAFT_FILENAME,
    )
    source_path = workspace_paths.processed_path(work_dir)
    output_path = workspace_paths.processed_draft_path(work_dir)
    source = _read_processed_source(source_path, report)
    if report["errors"]:
        report["status"] = "failed"
        return report

    draft = copy.deepcopy(source)
    draft["edit_state"] = "draft"
    draft["source"] = workspace_paths.PROCESSED_JSON_FILENAME
    write_json(output_path, draft)
    _write_editing_session(work_dir, report)
    _update_meta_status_if_present(work_dir, "draft_saved", report)
    report["status"] = "success"
    return report


def finalize_processed_edit(work_dir: Path) -> dict:
    work_dir = Path(work_dir)
    draft_path = workspace_paths.processed_draft_path(work_dir)
    if draft_path.exists():
        source_filename = workspace_paths.PROCESSED_DRAFT_FILENAME
        source_path = draft_path
    else:
        source_filename = workspace_paths.PROCESSED_JSON_FILENAME
        source_path = workspace_paths.processed_path(work_dir)

    report = _new_report(
        "finalize",
        work_dir,
        source_filename,
        workspace_paths.PROCESSED_FINAL_FILENAME,
    )
    source = _read_processed_source(source_path, report)
    if report["errors"]:
        report["status"] = "failed"
        return report

    final = copy.deepcopy(source)
    final["edit_state"] = "final"
    final["source"] = source_filename
    write_json(workspace_paths.processed_final_path(work_dir), final)
    _update_meta_status_if_present(work_dir, "finalized", report)
    report["status"] = "success"
    return report


def audio_to_edit_text(audio: str) -> str:
    text = PAUSE_MARKER_PATTERN.sub("", str(audio or ""))

    def replace_sub(match: re.Match[str]) -> str:
        alias = html.unescape(match.group("alias"))
        surface = html.unescape(match.group("surface"))
        if _is_symbol_substitution(alias, surface):
            return surface
        return f"{surface}《{alias}》"

    return SUB_ALIAS_PATTERN.sub(replace_sub, text)


def edit_text_to_audio(edit_text: str, source_audio: str | None = None) -> str:
    text = PAUSE_MARKER_PATTERN.sub("", str(edit_text or ""))
    _validate_empty_reading_annotation(text)
    text = _replace_marked_reading_annotations(text)
    for annotation in _reading_annotations_from_audio(source_audio):
        marker = f'{annotation["surface"]}《{annotation["alias"]}》'
        text = text.replace(marker, _sub_alias(annotation["surface"], annotation["alias"]))
    text = EDIT_RUBY_PATTERN.sub(
        lambda match: _sub_alias(match.group("surface"), match.group("reading")),
        text,
    )
    _validate_no_unparsed_annotations(text)
    return text


def build_audio_edit_spans(
    edit_text: str,
    symbol_to_category: dict,
    source_audio: str | None = None,
) -> list[dict]:
    if source_audio is not None:
        return _build_audio_edit_spans_from_source(source_audio, symbol_to_category)

    text = str(edit_text or "")
    symbol_keys = sorted(
        [str(symbol) for symbol in symbol_to_category if str(symbol)],
        key=len,
        reverse=True,
    )
    spans: list[dict] = []
    position = 0
    while position < len(text):
        reading_match = EDIT_RUBY_PATTERN.match(text, position)
        if reading_match:
            spans.append({"text": reading_match.group(0), "kind": "reading_annotation"})
            position = reading_match.end()
            continue

        matched_symbol = _match_symbol_at(text, position, symbol_keys)
        if matched_symbol is not None:
            spans.append(
                {
                    "text": matched_symbol,
                    "kind": "symbol",
                    "category": str(symbol_to_category[matched_symbol]),
                }
            )
            position += len(matched_symbol)
            continue

        spans.append({"text": text[position], "kind": "text"})
        position += 1
    return spans


def _new_report(operation: str, work_dir: Path, source: str, output: str) -> dict:
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "operation": operation,
        "work_id": work_dir.name,
        "status": "pending",
        "source": source,
        "output": output,
        "warnings": [],
        "errors": [],
    }


def _read_processed_source(path: Path, report: dict) -> dict | None:
    if not path.exists():
        report["errors"].append(_error("missing_source", f"missing file: {path.name}"))
        return None
    try:
        data = read_json(path)
    except json.JSONDecodeError as error:
        report["errors"].append(_error("invalid_json", f"{type(error).__name__}: {error}"))
        return None
    except Exception as error:
        report["errors"].append(_error("read_error", f"{type(error).__name__}: {error}"))
        return None

    if not isinstance(data, dict):
        report["errors"].append(_error("invalid_top_level", "processed JSON top-level must be object"))
        return None
    if data.get("schema_version") != PROCESSED_SCHEMA_VERSION:
        report["errors"].append(
            _error("invalid_schema_version", "schema_version must be processed-1")
        )
    if not isinstance(data.get("segments"), list):
        report["errors"].append(_error("invalid_segments", "segments must be a list"))
    if report["errors"]:
        return None
    return data


def _write_editing_session(work_dir: Path, report: dict) -> None:
    session_path = workspace_paths.editing_session_path(work_dir)
    now = _jst_now()
    session = _read_existing_session(session_path, report)
    session_id = str(session.get("session_id") or now.strftime("%Y%m%d-%H%M%S"))
    started_at = str(session.get("started_at") or now.isoformat())
    write_json(
        session_path,
        {
            "schema_version": EDITING_SESSION_SCHEMA_VERSION,
            "work_id": work_dir.name,
            "status": "editing",
            "base_file": workspace_paths.PROCESSED_JSON_FILENAME,
            "draft_file": workspace_paths.PROCESSED_DRAFT_FILENAME,
            "final_file": workspace_paths.PROCESSED_FINAL_FILENAME,
            "session_id": session_id,
            "started_at": started_at,
            "last_saved_at": now.isoformat(),
        },
    )


def _read_existing_session(path: Path, report: dict) -> dict:
    if not path.exists():
        return {}
    try:
        data = read_json(path)
    except Exception as error:
        report["warnings"].append(
            _warning("editing_session_read_error", f"{type(error).__name__}: {error}")
        )
        return {}
    if not isinstance(data, dict):
        report["warnings"].append(
            _warning("invalid_editing_session", "existing editing session is not object")
        )
        return {}
    return data


def _update_meta_status_if_present(work_dir: Path, status: str, report: dict) -> None:
    meta_path = work_dir / workspace_paths.WORK_META_FILENAME
    if not meta_path.exists():
        report["warnings"].append(_warning("missing_meta", "meta.json is missing; status was not updated"))
        return
    try:
        workspace_paths.update_work_meta_status(work_dir, status)
    except Exception as error:
        report["warnings"].append(_warning("meta_update_failed", f"{type(error).__name__}: {error}"))


def _jst_now() -> datetime:
    return datetime.now(JST).replace(microsecond=0)


def _reading_annotations_from_audio(source_audio: str | None) -> list[dict]:
    annotations: list[dict] = []
    if not source_audio:
        return annotations
    for match in SUB_ALIAS_PATTERN.finditer(str(source_audio)):
        alias = html.unescape(match.group("alias"))
        surface = html.unescape(match.group("surface"))
        if _is_symbol_substitution(alias, surface):
            continue
        annotations.append({"surface": surface, "alias": alias})
    annotations.sort(key=lambda item: len(item["surface"]), reverse=True)
    return annotations


def _is_symbol_substitution(alias: str, surface: str) -> bool:
    return bool(SYMBOL_ONLY_PATTERN.fullmatch(alias)) or bool(SYMBOL_ONLY_PATTERN.fullmatch(surface))


def _sub_alias(surface: str, alias: str) -> str:
    return f'<sub alias="{html.escape(alias, quote=True)}">{html.escape(surface)}</sub>'


def _validate_empty_reading_annotation(text: str) -> None:
    if "《》" in text:
        raise ValueError("読み指定の読みが空です。")


def _validate_no_unparsed_annotations(text: str) -> None:
    if "《" in text or "》" in text or any(marker in text for marker in MARKED_RUBY_MARKERS):
        raise ValueError("読み指定の書き方が正しくありません。")


def _replace_marked_reading_annotations(text: str) -> str:
    if not any(marker in text for marker in MARKED_RUBY_MARKERS):
        return text

    parts: list[str] = []
    position = 0
    while position < len(text):
        marker_index = _find_marked_ruby_start(text, position)
        if marker_index < 0:
            parts.append(text[position:])
            break

        parts.append(text[position:marker_index])
        open_index = text.find("《", marker_index + 1)
        close_index = text.find("》", open_index + 1) if open_index >= 0 else -1
        if open_index < 0 or close_index < 0:
            raise ValueError("読み指定の書き方が正しくありません。")

        surface = text[marker_index + 1 : open_index]
        reading = text[open_index + 1 : close_index]
        if not surface or not reading:
            raise ValueError("読み指定の読みが空です。")
        if any(char in surface for char in "|｜《》") or any(char in reading for char in "|｜《》"):
            raise ValueError("読み指定の書き方が正しくありません。")

        parts.append(_sub_alias(surface, reading))
        position = close_index + 1

    return "".join(parts)


def _find_marked_ruby_start(text: str, position: int) -> int:
    indexes = [text.find(marker, position) for marker in MARKED_RUBY_MARKERS]
    valid_indexes = [index for index in indexes if index >= 0]
    if not valid_indexes:
        return -1
    return min(valid_indexes)


def _match_symbol_at(text: str, position: int, symbol_keys: list[str]) -> str | None:
    for symbol in symbol_keys:
        if text.startswith(symbol, position):
            return symbol
    return None


def _build_audio_edit_spans_from_source(source_audio: str, symbol_to_category: dict) -> list[dict]:
    spans: list[dict] = []
    audio_text = str(source_audio or "")
    position = 0
    for match in SUB_ALIAS_PATTERN.finditer(audio_text):
        _append_plain_text_spans(
            spans,
            PAUSE_MARKER_PATTERN.sub("", html.unescape(audio_text[position : match.start()])),
            symbol_to_category,
        )
        alias = html.unescape(match.group("alias"))
        surface = html.unescape(match.group("surface"))
        if _is_symbol_substitution(alias, surface):
            _append_plain_text_spans(spans, surface, symbol_to_category)
        else:
            spans.append({"text": f"{surface}《{alias}》", "kind": "reading_annotation"})
        position = match.end()
    _append_plain_text_spans(
        spans,
        PAUSE_MARKER_PATTERN.sub("", html.unescape(audio_text[position:])),
        symbol_to_category,
    )
    return spans


def _append_plain_text_spans(spans: list[dict], text: str, symbol_to_category: dict) -> None:
    symbol_keys = sorted(
        [str(symbol) for symbol in symbol_to_category if str(symbol)],
        key=len,
        reverse=True,
    )
    position = 0
    while position < len(text):
        matched_symbol = _match_symbol_at(text, position, symbol_keys)
        if matched_symbol is not None:
            spans.append(
                {
                    "text": matched_symbol,
                    "kind": "symbol",
                    "category": str(symbol_to_category[matched_symbol]),
                }
            )
            position += len(matched_symbol)
            continue
        spans.append({"text": text[position], "kind": "text"})
        position += 1


def _error(code: str, message: str, **extra: Any) -> dict:
    return {"level": "error", "code": code, "message": message, **extra}


def _warning(code: str, message: str, **extra: Any) -> dict:
    return {"level": "warning", "code": code, "message": message, **extra}
