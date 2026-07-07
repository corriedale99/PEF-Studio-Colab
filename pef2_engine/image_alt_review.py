from __future__ import annotations

import struct
from copy import deepcopy
from pathlib import Path
from typing import Any

from pef2_engine import workspace_paths
from pef2_engine.io_utils import read_json, write_json


SCHEMA_VERSION = "image-alt-review-1"
SMALL_IMAGE_THRESHOLD_PX = 65
AUTO_DECORATIVE_REASON = "auto_small_image_65px"
ALLOWED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}
DEFAULT_IMAGE_ALT_LENGTH_TARGET = 100
IMAGE_ALT_LENGTH_TARGET_OPTIONS = (50, DEFAULT_IMAGE_ALT_LENGTH_TARGET, 300)

ALT_REVIEW_KEYS = {
    "gemini_alt_ja",
    "gemini_alt_en",
    "user_alt_ja",
    "user_alt_en",
    "is_decorative",
    "send_to_ai",
    "decorative_reason",
    "comment",
    "status",
    "error_message",
    "generated_at",
    "updated_at",
    "image_uploaded_at",
}


def image_alt_review_path(work_dir: Path) -> Path:
    return workspace_paths.image_alt_review_path(work_dir)


def load_image_alt_review(work_dir: Path) -> dict[str, Any]:
    return read_json(image_alt_review_path(work_dir), default=_empty_review(work_dir))


def save_image_alt_review(work_dir: Path, review: dict[str, Any]) -> None:
    write_json(image_alt_review_path(work_dir), review)


def normalize_alt_length_target(value: Any) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError):
        return DEFAULT_IMAGE_ALT_LENGTH_TARGET
    if number not in IMAGE_ALT_LENGTH_TARGET_OPTIONS:
        return DEFAULT_IMAGE_ALT_LENGTH_TARGET
    return number


def image_alt_review_settings(review: dict[str, Any] | None) -> dict[str, Any]:
    settings = review.get("settings") if isinstance(review, dict) else {}
    if not isinstance(settings, dict):
        settings = {}
    return {
        "alt_length_target": normalize_alt_length_target(settings.get("alt_length_target")),
    }


def sync_image_alt_review(work_dir: Path, *, save: bool = True) -> dict[str, Any]:
    processed_path = workspace_paths.processed_path(work_dir)
    processed = read_json(processed_path)
    if not isinstance(processed, dict):
        raise ValueError("02_processed.json top-level must be object")

    existing = load_image_alt_review(work_dir)
    if not isinstance(existing, dict):
        existing = _empty_review(work_dir)

    current_items = _current_items_from_processed(work_dir, processed)
    previous_items = _previous_items(existing)
    merged_items, matched_ids = _merge_current_items(current_items, previous_items)
    orphaned_items = _orphaned_items(previous_items, matched_ids)

    review = _base_review(work_dir, processed)
    review["settings"] = image_alt_review_settings(existing)
    review["items"] = merged_items
    review["orphaned_items"] = orphaned_items
    review["summary"] = {
        "image_count": len(merged_items),
        "uploaded_count": sum(1 for item in merged_items if item.get("image_exists") is True),
        "missing_count": sum(1 for item in merged_items if item.get("image_exists") is False),
        "decorative_count": sum(1 for item in merged_items if item.get("is_decorative") is True),
        "orphaned_count": len(orphaned_items),
    }

    if save:
        save_image_alt_review(work_dir, review)
    return review


def _base_review(work_dir: Path, processed: dict[str, Any]) -> dict[str, Any]:
    meta = read_json(work_dir / workspace_paths.WORK_META_FILENAME, default={})
    if not isinstance(meta, dict):
        meta = {}
    return {
        "schema_version": SCHEMA_VERSION,
        "source_processed_file": workspace_paths.PROCESSED_JSON_FILENAME,
        "work_id": str(processed.get("work_id") or meta.get("work_id") or work_dir.name),
        "title": str(processed.get("title") or meta.get("title") or ""),
        "settings": {
            "alt_length_target": DEFAULT_IMAGE_ALT_LENGTH_TARGET,
        },
        "items": [],
        "orphaned_items": [],
    }


def _empty_review(work_dir: Path) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "source_processed_file": workspace_paths.PROCESSED_JSON_FILENAME,
        "work_id": work_dir.name,
        "title": "",
        "settings": {
            "alt_length_target": DEFAULT_IMAGE_ALT_LENGTH_TARGET,
        },
        "items": [],
        "orphaned_items": [],
    }


def _current_items_from_processed(work_dir: Path, processed: dict[str, Any]) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    order = 1
    for segment in _processed_segments(processed):
        if not _is_image_segment(segment):
            continue
        image_path = _normalize_image_path(str(segment.get("image_file") or ""))
        filename = _safe_image_filename(image_path)
        resolved_path = work_dir / "images" / filename if filename else None
        width, height = image_dimensions(resolved_path)
        exists = bool(resolved_path and resolved_path.is_file())
        auto_decorative = bool(width and height and (width <= SMALL_IMAGE_THRESHOLD_PX or height <= SMALL_IMAGE_THRESHOLD_PX))
        item = {
            "order": order,
            "segment_index": segment.get("index"),
            "filename": filename,
            "image_path": image_path,
            "image_exists": exists,
            "image_width": width,
            "image_height": height,
            "gemini_alt_ja": "",
            "gemini_alt_en": "",
            "user_alt_ja": "",
            "user_alt_en": "",
            "is_decorative": auto_decorative,
            "send_to_ai": not auto_decorative,
            "decorative_reason": AUTO_DECORATIVE_REASON if auto_decorative else "",
            "comment": "",
            "status": "skipped" if auto_decorative else "pending",
            "error_message": "" if exists else "画像ファイルが見つかりません。",
            "generated_at": "",
            "updated_at": "",
        }
        items.append(item)
        order += 1
    return items


def _merge_current_items(current_items: list[dict[str, Any]], previous_items: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], set[int]]:
    matched_ids: set[int] = set()
    merged: list[dict[str, Any]] = []
    by_segment = _previous_by_segment(previous_items)
    by_image_path = _previous_by_unique_key(previous_items, "image_path")
    by_filename = _previous_by_unique_key(previous_items, "filename")

    for current in current_items:
        previous = _match_previous(current, by_segment, by_image_path, by_filename, matched_ids)
        item = dict(current)
        if previous is not None:
            matched_ids.add(id(previous))
            _copy_review_values(item, previous)
        merged.append(item)
    return merged, matched_ids


def _match_previous(
    current: dict[str, Any],
    by_segment: dict[str, dict[str, Any]],
    by_image_path: dict[str, dict[str, Any]],
    by_filename: dict[str, dict[str, Any]],
    matched_ids: set[int],
) -> dict[str, Any] | None:
    segment_key = _index_key(current.get("segment_index"))
    if segment_key:
        previous = by_segment.get(segment_key)
        if previous is not None and id(previous) not in matched_ids:
            return previous
    image_path_key = _text_key(current.get("image_path"))
    if image_path_key:
        previous = by_image_path.get(image_path_key)
        if previous is not None and id(previous) not in matched_ids:
            return previous
    filename_key = _text_key(current.get("filename"))
    if filename_key:
        previous = by_filename.get(filename_key)
        if previous is not None and id(previous) not in matched_ids:
            return previous
    return None


def _copy_review_values(target: dict[str, Any], source: dict[str, Any]) -> None:
    for key in ALT_REVIEW_KEYS:
        if key in source:
            target[key] = deepcopy(source[key])
    if target.get("image_exists") is True and str(target.get("error_message") or "") == "画像ファイルが見つかりません。":
        target["error_message"] = ""
    if target.get("image_exists") is False and not str(target.get("error_message") or "").strip():
        target["error_message"] = "画像ファイルが見つかりません。"
    if target.get("is_decorative") is True:
        target.setdefault("decorative_reason", AUTO_DECORATIVE_REASON)
        target.setdefault("send_to_ai", False)


def _orphaned_items(previous_items: list[dict[str, Any]], matched_ids: set[int]) -> list[dict[str, Any]]:
    orphaned = []
    for item in previous_items:
        if id(item) in matched_ids:
            continue
        orphan = deepcopy(item)
        orphan["orphaned"] = True
        orphaned.append(orphan)
    return orphaned


def _previous_items(existing: dict[str, Any]) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for key in ("items", "orphaned_items", "archived_items"):
        for item in existing.get(key) or []:
            if isinstance(item, dict):
                items.append(item)
    return items


def _previous_by_segment(previous_items: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    duplicates: set[str] = set()
    for item in previous_items:
        key = _index_key(item.get("segment_index"))
        if not key:
            continue
        if key in indexed:
            duplicates.add(key)
        else:
            indexed[key] = item
    for key in duplicates:
        indexed.pop(key, None)
    return indexed


def _previous_by_unique_key(previous_items: list[dict[str, Any]], field: str) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    duplicates: set[str] = set()
    for item in previous_items:
        key = _text_key(item.get(field))
        if not key:
            continue
        if key in indexed:
            duplicates.add(key)
        else:
            indexed[key] = item
    for key in duplicates:
        indexed.pop(key, None)
    return indexed


def _processed_segments(processed: dict[str, Any]) -> list[dict[str, Any]]:
    segments = processed.get("segments")
    if not isinstance(segments, list):
        return []
    return [segment for segment in segments if isinstance(segment, dict)]


def _is_image_segment(segment: dict[str, Any]) -> bool:
    return segment.get("block_type") == "image" or segment.get("is_image") is True


def _normalize_image_path(image_file: str) -> str:
    raw = image_file.strip().replace("\\", "/")
    if not raw:
        return ""
    parts = [part for part in raw.split("/") if part]
    if len(parts) == 1:
        return f"images/{parts[0]}"
    return "/".join(parts)


def _safe_image_filename(image_path: str) -> str:
    raw = image_path.strip().replace("\\", "/")
    if raw.startswith("/") or not raw:
        return ""
    parts = [part for part in raw.split("/") if part]
    if len(parts) != 2 or parts[0] != "images":
        return ""
    filename = parts[1]
    if filename in {".", ".."} or ".." in filename:
        return ""
    if "/" in filename or "\\" in filename:
        return ""
    if Path(filename).name != filename:
        return ""
    if Path(filename).suffix.lower() not in ALLOWED_IMAGE_EXTENSIONS:
        return ""
    return filename


def _index_key(value: Any) -> str:
    text = str(value or "").strip()
    return text


def _text_key(value: Any) -> str:
    return str(value or "").strip()


def image_dimensions(path: Path | None) -> tuple[int | None, int | None]:
    if path is None or not path.is_file():
        return None, None
    try:
        with path.open("rb") as file:
            header = file.read(32)
            file.seek(0)
            if header.startswith(b"\x89PNG\r\n\x1a\n"):
                file.seek(16)
                width, height = struct.unpack(">II", file.read(8))
                return int(width), int(height)
            if header.startswith(b"\xff\xd8"):
                return _jpeg_dimensions(file.read())
    except (OSError, struct.error):
        return None, None
    return None, None


def _jpeg_dimensions(data: bytes) -> tuple[int | None, int | None]:
    index = 2
    while index < len(data) - 9:
        if data[index] != 0xFF:
            index += 1
            continue
        marker = data[index + 1]
        index += 2
        if marker in {0xD8, 0xD9}:
            continue
        if index + 2 > len(data):
            break
        length = struct.unpack(">H", data[index : index + 2])[0]
        if marker in {0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF}:
            if index + 7 <= len(data):
                height = struct.unpack(">H", data[index + 3 : index + 5])[0]
                width = struct.unpack(">H", data[index + 5 : index + 7])[0]
                return int(width), int(height)
            break
        index += max(length, 2)
    return None, None
