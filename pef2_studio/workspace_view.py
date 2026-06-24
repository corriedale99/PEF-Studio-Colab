from __future__ import annotations

import copy
import math
import os
import re
import shutil
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from threading import Event
from typing import Any, Callable, Mapping

from pef2_engine import workspace_paths
from pef2_engine.dictionary_finalize import run_finalize_dictionary
from pef2_engine.dictionary_review import (
    DictionaryReviewValidationError,
    append_manual_dictionary_review_item,
    apply_dictionary_review_form_update,
    build_dictionary_review,
    build_empty_dictionary_review,
    build_manual_dictionary_review,
)
from pef2_engine.dictionary_loader import load_symbol_reading_rules
from pef2_engine.io_utils import read_json, write_json
from pef2_engine.legacy_dictionary_import import (
    LegacyDictionaryImportValidationError,
    build_dictionary_review_from_legacy_dictionary,
    load_legacy_dictionary_json_bytes,
)
from pef2_engine.pre_processed_parser import parse_source_file_to_pre_processed
from pef2_engine.processed_builder import run_processed_workspace_full
from pef2_engine.processed_editing import (
    audio_to_edit_text,
    build_audio_edit_spans,
    edit_text_to_audio,
)
from pef2_engine.gemini_dictionary_review import AIDictionaryReviewCancelled, MAX_REVIEW_TERMS
from pef2_engine.generation_lock import active_generation_lock_message, generation_lock_path, read_generation_lock
from pef2_engine.step5_dictionary_draft import (
    GEMINI_REVIEW_RAW_FILENAME,
    STEP5_DIRNAME,
    run_step5_dictionary_draft,
)
from pef2_engine.tts_settings import (
    read_workspace_settings,
    read_work_tts_settings,
    resolve_tts_settings,
    work_tts_settings_path,
    workspace_settings_path,
    write_workspace_settings,
    write_work_tts_settings,
)
from pef2_engine.tts_generator import VOICE_PREVIEW_DIRNAME, VOICE_PREVIEW_FILENAME, WORKSPACE_TEMP_DIRNAME

STATUS_LABELS = {
    "pre_processed": "原稿取込済み",
    "dictionary_review_ready": "辞書確認待ち",
    "dictionary_finalized": "辞書確定済み",
    "processed": "編集できます",
    "draft_saved": "編集中",
    "finalized": "確定済み",
    "audio_generated": "確定済み",
    "exported": "出力済み",
}
DICTIONARY_REVIEW_DECISION_LABELS = {
    "pending": "未確認",
    "accept": "採用",
    "edit": "修正して採用",
    "ignore": "採用しない",
}
GENERATION_LABELS = {
    "tts": "音声生成",
    "epub": "EPUB生成",
}
MODE_LABELS = {
    "processed": "編集できます",
    "draft": "編集中",
    "final_readonly": "確定済み",
    "unavailable": "編集できません",
}
MODE_DESCRIPTIONS = {
    "processed": "機械生成版から編集を開始します。保存すると途中保存が作成されます。",
    "draft": "途中保存があります。",
    "final_readonly": "編集するには「再編集開始」を押してください。",
    "unavailable": "辞書作成に進めます。",
}
PER_PAGE_OPTIONS = [10, 20, 50]
DEFAULT_PER_PAGE = 20
WORKS_SORT_OPTIONS = {
    "updated_desc": "更新日時が新しい順",
    "updated_asc": "更新日時が古い順",
    "created_desc": "作成日時が新しい順",
    "created_asc": "作成日時が古い順",
    "title_asc": "作品名順",
    "title_desc": "作品名逆順",
    "status": "状態順",
}
DEFAULT_WORKS_SORT = "updated_desc"
RESERVED_WORK_DIR_NAMES = {"_trash", "backups"}
ALLOWED_IMAGE_UPLOAD_EXTENSIONS = {".png", ".jpg", ".jpeg"}
IMAGE_UPLOAD_MAX_BYTES = 10 * 1024 * 1024
IMAGE_UPLOAD_SUCCESS_MESSAGE = "画像を保存しました。変更をEPUBに反映するには、EPUB生成をやり直してください。"
IMAGE_UPLOAD_FORMAT_ERROR_MESSAGE = "PNGまたはJPEG画像を選んでください。"
IMAGE_UPLOAD_SIZE_ERROR_MESSAGE = "画像ファイルが大きすぎます。10MB以下の画像を選んでください。"
DICTIONARY_RESET_BACKUP_FILENAMES = (
    workspace_paths.LEGACY_DICTIONARY_FILENAME,
    workspace_paths.DICTIONARY_REVIEW_FILENAME,
    workspace_paths.WORK_DICTIONARY_FILENAME,
    workspace_paths.DICTIONARY_FINALIZE_REPORT_FILENAME,
    workspace_paths.PROCESSED_JSON_FILENAME,
    workspace_paths.PROCESSED_REPORT_FILENAME,
)
DICTIONARY_FINALIZE_BACKUP_FILENAMES = (
    workspace_paths.WORK_DICTIONARY_FILENAME,
    workspace_paths.DICTIONARY_FINALIZE_REPORT_FILENAME,
    workspace_paths.PROCESSED_JSON_FILENAME,
    workspace_paths.PROCESSED_REPORT_FILENAME,
)
GEMINI_DICTIONARY_RESET_BACKUP_FILENAMES = (
    workspace_paths.DICTIONARY_REVIEW_FILENAME,
    workspace_paths.WORK_DICTIONARY_DRAFT_FILENAME,
    workspace_paths.WORK_DICTIONARY_FILENAME,
    workspace_paths.DICTIONARY_FINALIZE_REPORT_FILENAME,
    workspace_paths.PROCESSED_JSON_FILENAME,
    workspace_paths.PROCESSED_REPORT_FILENAME,
)


def get_workspace_root(
    env: dict[str, str] | None = None,
    cwd: Path | None = None,
) -> Path:
    source_env = env if env is not None else os.environ
    if source_env.get("PEF_WORKSPACE"):
        return Path(source_env["PEF_WORKSPACE"]).expanduser()
    return (cwd or Path.cwd()) / "workspace"


def list_works(workspace_root: Path, sort: str = DEFAULT_WORKS_SORT) -> list[dict]:
    workspace_root = Path(workspace_root)
    if not workspace_root.exists():
        return []

    works: list[dict] = []
    for item in sorted(workspace_root.iterdir(), key=lambda path: path.name):
        if (
            not item.is_dir()
            or item.name == "dictionaries"
            or _is_hidden_workspace_dir(item)
        ):
            continue
        meta = _read_optional_dict(item / workspace_paths.WORK_META_FILENAME)
        status = str(meta.get("status") or "")
        effective_status = _effective_status(item, status)
        has_final = (item / workspace_paths.PROCESSED_FINAL_FILENAME).exists()
        has_draft = (item / workspace_paths.PROCESSED_DRAFT_FILENAME).exists()
        works.append(
            {
                "work_id": item.name,
                "work_created_label": _work_created_label(item.name),
                "title": _work_title(item.name, meta),
                "status": status or "unknown",
                "effective_status": effective_status,
                "status_label": status_label(effective_status),
                "created_at": str(meta.get("created_at") or ""),
                "created_at_label": _datetime_label(
                    str(meta.get("created_at") or "")
                )
                or _work_created_label(item.name),
                "updated_at": str(meta.get("updated_at") or ""),
                "updated_at_label": _datetime_label(str(meta.get("updated_at") or "")),
                "has_meta": bool(meta),
                "has_final": has_final,
                "has_draft": has_draft,
                "can_generate": has_final and not has_draft,
                "has_epub": _has_official_epub(item),
            }
        )
    return _sort_works(works, normalize_works_sort(sort))


def normalize_works_sort(sort: str | None) -> str:
    value = str(sort or "").strip()
    if value in WORKS_SORT_OPTIONS:
        return value
    return DEFAULT_WORKS_SORT


def save_workspace_tts_settings_submission(workspace_root: Path, form_data) -> dict:
    speaker_id = _parse_speaker_id(form_data.get("speaker_id", ""))
    if speaker_id is None:
        return {
            "status": "failed",
            "message": "話者IDを確認してください。",
            "errors": ["話者IDを確認してください。"],
        }
    breath_settings = _parse_breath_settings(
        form_data,
        resolve_tts_settings(workspace_root)["breath"],
    )
    if breath_settings is None:
        return {
            "status": "failed",
            "message": "息継ぎパラメータを確認してください。",
            "errors": ["息継ぎパラメータを確認してください。"],
        }

    existing = read_workspace_settings(workspace_root)
    tts = existing.get("tts") if isinstance(existing.get("tts"), dict) else {}
    voice = tts.get("voice") if isinstance(tts.get("voice"), dict) else {}
    tts["voice"] = {**voice, "speaker_id": speaker_id}
    tts["breath"] = breath_settings
    write_workspace_settings(workspace_root, tts)
    return {
        "status": "success",
        "message": "全体設定を保存しました。作品別設定がない作品では、この設定が使われます。",
        "speaker_id": speaker_id,
    }


def build_workspace_tts_settings_view(workspace_root: Path) -> dict:
    resolved = resolve_tts_settings(workspace_root)
    preview_path = Path(workspace_root) / WORKSPACE_TEMP_DIRNAME / VOICE_PREVIEW_FILENAME
    preview_exists = preview_path.is_file()
    return {
        "speaker_id": resolved["voice"]["speaker_id"],
        "backend": resolved["voice"]["backend"],
        "choking_threshold": resolved["breath"]["choking_threshold"],
        "distance_threshold": resolved["breath"]["distance_threshold"],
        "has_workspace_settings": workspace_settings_path(workspace_root).exists(),
        "has_voice_preview": preview_exists,
        "voice_preview_version": int(preview_path.stat().st_mtime) if preview_exists else 0,
        "speaker_options": _speaker_options(),
    }


def load_work_detail(
    workspace_root: Path,
    work_id: str,
    view: str | None = None,
    page: object | None = None,
    per_page: object | None = None,
    save_errors: list[str] | None = None,
    posted_audio_edits: Mapping[str, str] | None = None,
    segment_errors: Mapping[str, str] | None = None,
) -> dict | None:
    work_dir = resolve_work_dir(workspace_root, work_id)
    if work_dir is None:
        return None

    meta = _read_optional_dict(work_dir / workspace_paths.WORK_META_FILENAME)
    selected = select_processed_file(work_dir, view=view)
    effective_status = _effective_status(work_dir, str(meta.get("status") or ""))
    rules_by_category, symbol_to_category, symbol_rule_warnings = (
        load_symbol_reading_rules(workspace_root)
    )
    del rules_by_category

    processed_data: Any = {}
    errors: list[str] = []
    if selected["path"] is not None:
        try:
            processed_data = read_json(selected["path"])
        except Exception as error:
            errors.append(f"{type(error).__name__}: {error}")

    all_segments = build_segment_views(
        processed_data,
        symbol_to_category,
        posted_audio_edits=posted_audio_edits,
        segment_errors=segment_errors,
    )
    image_summary = build_work_image_summary(work_dir)
    pagination = paginate_segments(all_segments, page=page, per_page=per_page)
    dictionary_review_info = _dictionary_review_debug_info(work_dir)
    gemini_review_debug = _gemini_review_debug_info(work_dir)
    active_generation_lock = _active_generation_lock_info(work_dir)
    has_pre_processed = (
        work_dir / workspace_paths.PRE_PROCESSED_JSON_FILENAME
    ).exists()
    has_dictionary_review = (
        work_dir / workspace_paths.DICTIONARY_REVIEW_FILENAME
    ).exists()
    has_draft = (work_dir / workspace_paths.PROCESSED_DRAFT_FILENAME).exists()
    has_final = (work_dir / workspace_paths.PROCESSED_FINAL_FILENAME).exists()
    tts_settings = build_tts_settings_view(workspace_root, work_dir)
    show_dictionary_file_import = has_pre_processed
    can_import_dictionary_file = (
        show_dictionary_file_import
        and not has_draft
        and not has_final
    )
    can_direct_finalize_dictionary = (
        has_dictionary_review and not has_draft and not has_final
    )
    can_create_manual_dictionary_review = (
        has_pre_processed
        and not has_dictionary_review
        and not has_draft
        and not has_final
    )
    can_create_ai_dictionary_review = (
        has_pre_processed
        and not has_draft
        and not has_final
        and (work_dir / workspace_paths.SOURCE_ORIGINAL_FILENAME).exists()
    )

    return {
        "work_id": work_dir.name,
        "title": _work_title(work_dir.name, meta),
        "input_stem": str(meta.get("input_stem") or ""),
        "import_type": str(meta.get("import_type") or ""),
        "source_format": str(meta.get("source_format") or ""),
        "import_type_label": _import_type_label(work_dir, meta),
        "status": str(meta.get("status") or "unknown"),
        "status_label": status_label(effective_status),
        "status_description": status_description(effective_status, selected["mode"]),
        "workflow_steps": workflow_steps(effective_status),
        "opened_file": selected["filename"],
        "mode": selected["mode"],
        "can_save_draft": selected["mode"] in {"processed", "draft"},
        "can_save_final": selected["mode"] in {"processed", "draft"},
        "can_start_reedit": selected["mode"] == "final_readonly",
        "show_legacy_dictionary_import": show_dictionary_file_import,
        "can_import_legacy_dictionary": can_import_dictionary_file,
        "show_dictionary_file_import": show_dictionary_file_import,
        "can_import_dictionary_file": can_import_dictionary_file,
        "can_reset_dictionary_file": (
            can_import_dictionary_file and has_dictionary_review
        ),
        "can_direct_finalize_dictionary": can_direct_finalize_dictionary,
        "can_create_manual_dictionary_review": can_create_manual_dictionary_review,
        "can_create_ai_dictionary_review": can_create_ai_dictionary_review,
        "has_dictionary_review_items": dictionary_review_info["items"] > 0,
        "dictionary_file_import_block_reason": _dictionary_file_import_block_reason(
            has_dictionary_review=has_dictionary_review,
            has_draft=has_draft,
            has_final=has_final,
        ),
        "has_dictionary_review": has_dictionary_review,
        "can_open_dictionary_review": has_dictionary_review,
        "can_create_empty_dictionary_processed": _can_create_empty_dictionary_processed(
            work_dir, meta
        ),
        "has_legacy_dictionary": (
            work_dir / workspace_paths.LEGACY_DICTIONARY_FILENAME
        ).exists(),
        "dictionary_review_items": dictionary_review_info["items"],
        "dictionary_review_source": dictionary_review_info["source"],
        "gemini_review_debug": gemini_review_debug,
        "active_generation_lock": active_generation_lock,
        "has_pre_processed": has_pre_processed,
        "has_processed": (work_dir / workspace_paths.PROCESSED_JSON_FILENAME).exists(),
        "has_draft": has_draft,
        "has_final": has_final,
        "has_audio": _has_audio_outputs(work_dir),
        "has_epub": _has_official_epub(work_dir),
        "can_download_epub": _has_current_epub(workspace_root, work_dir),
        "can_generate": (work_dir / workspace_paths.PROCESSED_FINAL_FILENAME).exists()
        and not (work_dir / workspace_paths.PROCESSED_DRAFT_FILENAME).exists(),
        "audio_needs_regeneration": _audio_needs_regeneration(workspace_root, work_dir),
        "epub_needs_regeneration": _epub_needs_regeneration(workspace_root, work_dir),
        "tts_settings_outputs_stale": _tts_settings_newer_than_outputs(
            workspace_root, work_dir
        ),
        "image_summary": image_summary,
        "can_enter_audio_edit": selected["mode"] in {"processed", "draft"},
        "segments": pagination["segments"],
        "segment_total": len(all_segments),
        "pagination": pagination,
        "symbol_rule_warnings": symbol_rule_warnings,
        "tts_settings": tts_settings,
        "errors": errors,
        "save_errors": list(save_errors or []),
        "error_segment_indexes": sorted(
            [str(index) for index in (segment_errors or {}).keys()],
            key=lambda index: (len(index), index),
        ),
    }


def save_work_tts_settings_submission(workspace_root: Path, work_id: str, form_data) -> dict | None:
    work_dir = resolve_work_dir(workspace_root, work_id)
    if work_dir is None:
        return None

    speaker_id = _parse_speaker_id(form_data.get("speaker_id", ""))
    if speaker_id is None:
        return {
            "status": "failed",
            "message": "話者IDを確認してください。",
            "errors": ["話者IDを確認してください。"],
        }
    breath_settings = _parse_breath_settings(
        form_data,
        resolve_tts_settings(workspace_root, work_dir)["breath"],
    )
    if breath_settings is None:
        return {
            "status": "failed",
            "message": "息継ぎパラメータを確認してください。",
            "errors": ["息継ぎパラメータを確認してください。"],
        }

    existing = read_work_tts_settings(work_dir)
    voice = existing.get("voice") if isinstance(existing.get("voice"), dict) else {}
    existing["voice"] = {**voice, "speaker_id": speaker_id}
    existing["breath"] = breath_settings
    write_work_tts_settings(work_dir, existing)
    return {
        "status": "success",
        "message": "音声設定を保存しました。変更を反映するには、EPUB生成をやり直してください。",
        "speaker_id": speaker_id,
    }


def load_work_images_page(
    workspace_root: Path,
    work_id: str,
    result: dict | None = None,
) -> dict | None:
    work_dir = resolve_work_dir(workspace_root, work_id)
    if work_dir is None:
        return None
    meta = _read_optional_dict(work_dir / workspace_paths.WORK_META_FILENAME)
    return {
        "work_id": work_dir.name,
        "title": _work_title(work_dir.name, meta),
        "image_summary": build_work_image_summary(work_dir),
        "result": result,
    }


def save_work_image_upload(workspace_root: Path, work_id: str, segment_index: str, image_upload) -> dict | None:
    work_dir = resolve_work_dir(workspace_root, work_id)
    if work_dir is None:
        return None

    item = _image_item_by_index(work_dir, segment_index)
    if item is None:
        return _image_upload_result("failed", "画像指定が見つかりません。", "missing_image_segment")
    if not item.get("is_safe"):
        return _image_upload_result("failed", "画像の保存先を確認してください。", "invalid_image_file")
    if image_upload is None or not getattr(image_upload, "filename", ""):
        return _image_upload_result("failed", IMAGE_UPLOAD_FORMAT_ERROR_MESSAGE, "missing_upload")

    upload_name = str(getattr(image_upload, "filename", "") or "")
    if Path(upload_name).suffix.lower() not in ALLOWED_IMAGE_UPLOAD_EXTENSIONS:
        return _image_upload_result("failed", IMAGE_UPLOAD_FORMAT_ERROR_MESSAGE, "invalid_extension")

    image_dir = work_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    target_path = item["path"]
    tmp_path: Path | None = None
    total_size = 0
    try:
        with tempfile.NamedTemporaryFile(delete=False, dir=image_dir, prefix=".upload_", suffix=".tmp") as tmp:
            tmp_path = Path(tmp.name)
            while True:
                chunk = image_upload.stream.read(1024 * 1024)
                if not chunk:
                    break
                total_size += len(chunk)
                if total_size > IMAGE_UPLOAD_MAX_BYTES:
                    return _image_upload_result("failed", IMAGE_UPLOAD_SIZE_ERROR_MESSAGE, "too_large")
                tmp.write(chunk)
        if total_size == 0 or not _looks_like_allowed_image(tmp_path):
            return _image_upload_result("failed", IMAGE_UPLOAD_FORMAT_ERROR_MESSAGE, "invalid_image")
        os.replace(tmp_path, target_path)
        tmp_path = None
    finally:
        if tmp_path is not None:
            try:
                tmp_path.unlink()
            except OSError:
                pass

    return _image_upload_result(
        "success",
        IMAGE_UPLOAD_SUCCESS_MESSAGE,
        "saved",
        segment_index=str(item.get("index") or ""),
        image_file=str(item.get("image_file") or ""),
    )


def build_work_image_summary(work_dir: Path) -> dict:
    selected = select_image_processed_file(work_dir)
    items: list[dict] = []
    errors: list[str] = []
    if selected["path"] is not None:
        try:
            processed_data = read_json(selected["path"])
            items = build_image_items(work_dir, processed_data)
        except Exception as error:
            errors.append(f"{type(error).__name__}: {error}")
    uploaded_count = sum(1 for item in items if item["exists"])
    missing_count = len(items) - uploaded_count
    return {
        "source_file": selected["filename"],
        "items": items,
        "count": len(items),
        "uploaded_count": uploaded_count,
        "missing_count": missing_count,
        "has_images": bool(items),
        "epub_needs_regeneration": _image_epub_needs_regeneration(work_dir, items),
        "errors": errors,
    }


def select_image_processed_file(work_dir: Path) -> dict:
    candidates = (
        workspace_paths.PROCESSED_FINAL_FILENAME,
        workspace_paths.PROCESSED_DRAFT_FILENAME,
        workspace_paths.PROCESSED_JSON_FILENAME,
    )
    for filename in candidates:
        path = work_dir / filename
        if path.exists():
            return {"path": path, "filename": filename}
    return {"path": None, "filename": "none"}


def build_image_items(work_dir: Path, processed_data: Any) -> list[dict]:
    items: list[dict] = []
    for segment in _processed_segments(processed_data):
        if not _is_image_segment(segment):
            continue
        image_file = str(segment.get("image_file") or "").strip()
        filename = _safe_image_filename(image_file)
        image_path = work_dir / "images" / filename if filename else work_dir / "images"
        exists = bool(filename) and image_path.is_file()
        items.append(
            {
                "index": segment.get("index", ""),
                "image_file": image_file,
                "filename": filename,
                "is_safe": bool(filename),
                "path": image_path,
                "exists": exists,
                "status_label": "アップロード済み" if exists else "未アップロード",
            }
        )
    return items


def resolve_work_image_file(workspace_root: Path, work_id: str, segment_index: str) -> Path | None:
    work_dir = resolve_work_dir(workspace_root, work_id)
    if work_dir is None:
        return None
    item = _image_item_by_index(work_dir, segment_index)
    if item is None or not item.get("is_safe") or not item.get("exists"):
        return None
    return item["path"]


def _image_epub_needs_regeneration(work_dir: Path, image_items: list[dict]) -> bool:
    latest_epub = _latest_official_epub(work_dir)
    if latest_epub is None:
        return False
    try:
        latest_epub_mtime = latest_epub.stat().st_mtime
        return any(item.get("exists") and item["path"].stat().st_mtime > latest_epub_mtime for item in image_items)
    except (OSError, TypeError, ValueError):
        return False


def build_tts_settings_view(workspace_root: Path, work_dir: Path) -> dict:
    resolved = resolve_tts_settings(workspace_root, work_dir)
    work_settings_exists = work_tts_settings_path(work_dir).exists()
    workspace_settings_exists = workspace_settings_path(workspace_root).exists()
    preview_path = work_dir / "audio" / VOICE_PREVIEW_DIRNAME / VOICE_PREVIEW_FILENAME
    preview_exists = preview_path.is_file()
    source_label = (
        "この作品の設定を使用中"
        if work_settings_exists
        else "全体設定または標準設定を使用中"
    )
    return {
        "speaker_id": resolved["voice"]["speaker_id"],
        "backend": resolved["voice"]["backend"],
        "choking_threshold": resolved["breath"]["choking_threshold"],
        "distance_threshold": resolved["breath"]["distance_threshold"],
        "source_label": source_label,
        "has_work_settings": work_settings_exists,
        "has_workspace_settings": workspace_settings_exists,
        "has_voice_preview": preview_exists,
        "voice_preview_version": int(preview_path.stat().st_mtime) if preview_exists else 0,
        "speaker_options": _speaker_options(),
    }


def _parse_speaker_id(value: object) -> int | None:
    text = str(value or "").strip()
    if not text.isdigit():
        return None
    speaker_id = int(text)
    return speaker_id if speaker_id >= 0 else None


def _parse_breath_settings(form_data, fallback: dict) -> dict | None:
    choking_threshold = _parse_optional_bounded_int(
        form_data.get("choking_threshold"),
        fallback.get("choking_threshold", 20),
        minimum=5,
        maximum=80,
    )
    distance_threshold = _parse_optional_bounded_int(
        form_data.get("distance_threshold"),
        fallback.get("distance_threshold", 6),
        minimum=0,
        maximum=30,
    )
    if choking_threshold is None or distance_threshold is None:
        return None
    return {
        "choking_threshold": choking_threshold,
        "distance_threshold": distance_threshold,
    }


def _parse_optional_bounded_int(
    value: object,
    fallback: object,
    *,
    minimum: int,
    maximum: int,
) -> int | None:
    text = str(value or "").strip()
    if not text:
        candidate = fallback
    elif text.isdigit():
        candidate = int(text)
    else:
        return None
    if isinstance(candidate, bool):
        return None
    try:
        number = int(candidate)
    except (TypeError, ValueError):
        return None
    return number if minimum <= number <= maximum else None


def _speaker_options() -> list[dict]:
    return [
        {"label": "めたん", "speaker_id": 2},
        {"label": "ずんだもん", "speaker_id": 3},
        {"label": "つむぎ", "speaker_id": 8},
        {"label": "青山龍星", "speaker_id": 13},
    ]


def import_legacy_pef_upload(
    workspace_root: Path,
    *,
    json_upload,
    txt_upload,
    keep_failed: bool | None = None,
) -> dict:
    errors = _validate_legacy_uploads(json_upload, txt_upload)
    if errors:
        return {"status": "failed", "errors": errors}

    workspace_root = Path(workspace_root)
    _ensure_workspace_root_layout(workspace_root)
    keep_failed_import = (
        _keep_failed_imports_enabled() if keep_failed is None else keep_failed
    )
    input_stem = _legacy_json_base_stem(Path(str(json_upload.filename).strip()).stem)
    work_id = _unique_generated_work_id(workspace_root, input_stem)
    failed_dir_name = f"_import_failed_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    temp_dir_path = Path(
        tempfile.mkdtemp(
            prefix="_import_tmp_",
            dir=str(workspace_root),
        )
    )
    final_work_dir = workspace_root / work_id
    json_filename = f"{input_stem}.json"
    txt_filename = f"{input_stem}.txt"

    try:
        _ensure_workspace_runtime_dirs(temp_dir_path)
        json_path = temp_dir_path / json_filename
        txt_path = temp_dir_path / txt_filename
        source_original_path = temp_dir_path / workspace_paths.SOURCE_ORIGINAL_FILENAME

        json_bytes = _upload_bytes(json_upload)
        txt_bytes = _upload_bytes(txt_upload)
        json_path.write_bytes(json_bytes)
        txt_path.write_bytes(txt_bytes)
        source_original_path.write_bytes(txt_bytes)

        pre_processed = parse_source_file_to_pre_processed(json_path)
        write_json(workspace_paths.pre_processed_path(temp_dir_path), pre_processed)

        meta = workspace_paths.create_work_meta(
            work_id=work_id,
            title=input_stem,
            source_original_filename=txt_filename,
            input_stem=input_stem,
            status="pre_processed",
        )
        meta["import_type"] = "legacy_pef_import"
        meta["source_format"] = "pef_legacy"
        write_json(temp_dir_path / workspace_paths.WORK_META_FILENAME, meta)
        temp_dir_path.rename(final_work_dir)
        return {
            "status": "success",
            "work_id": work_id,
            "work_dir": final_work_dir,
            "title": input_stem,
        }
    except Exception as error:
        report = {
            "status": "failed",
            "error_type": type(error).__name__,
            "error": str(error),
            "json_filename": json_filename,
            "txt_filename": txt_filename,
            "note": "v1.1-2持ち越し: 作品名-辞書.json は今回取り込みません。",
        }
        _write_import_failure_report(temp_dir_path, report)
        if keep_failed_import:
            failed_dir = workspace_root / failed_dir_name
            if failed_dir.exists():
                failed_dir = workspace_root / f"{failed_dir_name}_{os.getpid()}"
            try:
                temp_dir_path.rename(failed_dir)
            except OSError:
                pass
        else:
            shutil.rmtree(temp_dir_path, ignore_errors=True)
        return {"status": "failed", "errors": [_import_error_message(error)]}


def import_text_upload(
    workspace_root: Path,
    *,
    title: str,
    txt_upload,
    keep_failed: bool | None = None,
) -> dict:
    errors, resolved_title, input_stem = _validate_text_upload(title, txt_upload)
    if errors:
        return {"status": "failed", "errors": errors}

    workspace_root = Path(workspace_root)
    _ensure_workspace_root_layout(workspace_root)
    keep_failed_import = (
        _keep_failed_imports_enabled() if keep_failed is None else keep_failed
    )
    work_id = _unique_generated_work_id(workspace_root, resolved_title)
    failed_dir_name = f"_import_failed_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    temp_dir_path = Path(
        tempfile.mkdtemp(
            prefix="_import_tmp_",
            dir=str(workspace_root),
        )
    )
    final_work_dir = workspace_root / work_id
    txt_filename = f"{input_stem}.txt"

    try:
        _ensure_workspace_runtime_dirs(temp_dir_path)
        txt_path = temp_dir_path / txt_filename
        source_original_path = temp_dir_path / workspace_paths.SOURCE_ORIGINAL_FILENAME

        txt_bytes = _upload_bytes(txt_upload)
        txt_path.write_bytes(txt_bytes)
        source_original_path.write_bytes(txt_bytes)

        pre_processed = parse_source_file_to_pre_processed(txt_path)
        write_json(workspace_paths.pre_processed_path(temp_dir_path), pre_processed)

        meta = workspace_paths.create_work_meta(
            work_id=work_id,
            title=resolved_title,
            source_original_filename=txt_filename,
            input_stem=input_stem,
            status="pre_processed",
        )
        meta["import_type"] = "text_import"
        meta["source_format"] = "plain_text"
        write_json(temp_dir_path / workspace_paths.WORK_META_FILENAME, meta)
        temp_dir_path.rename(final_work_dir)
        return {
            "status": "success",
            "work_id": work_id,
            "work_dir": final_work_dir,
            "title": resolved_title,
        }
    except Exception as error:
        report = {
            "status": "failed",
            "error_type": type(error).__name__,
            "error": str(error),
            "txt_filename": txt_filename,
        }
        _write_import_failure_report(temp_dir_path, report)
        if keep_failed_import:
            failed_dir = workspace_root / failed_dir_name
            if failed_dir.exists():
                failed_dir = workspace_root / f"{failed_dir_name}_{os.getpid()}"
            try:
                temp_dir_path.rename(failed_dir)
            except OSError:
                pass
        else:
            shutil.rmtree(temp_dir_path, ignore_errors=True)
        return {"status": "failed", "errors": [_text_import_error_message(error)]}


def import_legacy_dictionary_upload(
    workspace_root: Path,
    work_id: str,
    *,
    dictionary_upload,
) -> dict | None:
    work_dir = resolve_work_dir(workspace_root, work_id)
    if work_dir is None:
        return None

    errors = _validate_legacy_dictionary_upload(dictionary_upload)
    if errors:
        return {"status": "failed", "errors": errors, "warnings": []}

    review_path = workspace_paths.dictionary_review_path(work_dir)
    had_dictionary_review = review_path.exists()
    meta_path = work_dir / workspace_paths.WORK_META_FILENAME
    original_meta = _read_optional_dict(meta_path)
    edit_artifacts = detect_existing_edit_artifacts(work_dir)
    if edit_artifacts["should_block"]:
        return {
            "status": "failed",
            "errors": [
                "読みと息継ぎの編集結果があります。辞書を変更する場合は、編集済みデータへの反映方法を別途選ぶ必要があります。"
            ],
            "warnings": [],
        }

    payload = _upload_bytes(dictionary_upload)
    try:
        with tempfile.TemporaryDirectory(
            prefix="_dictionary_reset_tmp_", dir=str(work_dir)
        ) as temp_dir_name:
            temp_dir = Path(temp_dir_name)
            temp_legacy_path = temp_dir / workspace_paths.LEGACY_DICTIONARY_FILENAME
            temp_review_path = temp_dir / workspace_paths.DICTIONARY_REVIEW_FILENAME
            temp_legacy_path.write_bytes(payload)
            legacy_data = load_legacy_dictionary_json_bytes(payload)
            review_data, warnings = build_dictionary_review_from_legacy_dictionary(
                legacy_data,
                input_stem=_detail_input_stem(work_dir),
            )
            write_json(temp_review_path, review_data)

            backup = _backup_existing_files(
                work_dir,
                "dictionary_reset",
                DICTIONARY_RESET_BACKUP_FILENAMES,
            )
            try:
                shutil.move(
                    str(temp_legacy_path),
                    str(workspace_paths.legacy_dictionary_path(work_dir)),
                )
                shutil.move(str(temp_review_path), str(review_path))
                workspace_paths.update_work_meta_status(
                    work_dir, "dictionary_review_ready"
                )
            except Exception:
                _restore_backup_files(
                    work_dir,
                    backup,
                    DICTIONARY_RESET_BACKUP_FILENAMES,
                )
                _restore_meta_status(
                    meta_path, original_meta, str(original_meta.get("status") or "")
                )
                raise
    except LegacyDictionaryImportValidationError as error:
        return {
            "status": "failed",
            "title": "作品辞書読み込み",
            "errors": [_legacy_dictionary_import_error_message(error)],
            "warnings": [],
            "display_scope": "dictionary_card",
            "card_anchor": "dictionary-card",
        }
    except Exception as error:
        return {
            "status": "failed",
            "title": "作品辞書読み込み",
            "errors": [f"作品辞書ファイルを読み込めませんでした。{type(error).__name__}: {error}"],
            "warnings": [],
            "display_scope": "dictionary_card",
            "card_anchor": "dictionary-card",
        }

    return {
        "status": "success",
        "title": "作品辞書読み込み",
        "message": "作品辞書を読み込みました。",
        "errors": [],
        "warnings": warnings,
        "display_scope": "dictionary_card" if had_dictionary_review else "header",
    }


def load_dictionary_review_page(
    workspace_root: Path,
    work_id: str,
    *,
    form_items: list[dict] | None = None,
) -> dict | None:
    work_dir = resolve_work_dir(workspace_root, work_id)
    if work_dir is None:
        return None

    meta = _read_optional_dict(work_dir / workspace_paths.WORK_META_FILENAME)
    review_path = workspace_paths.dictionary_review_path(work_dir)
    review_data = _read_required_dictionary_review(review_path)
    if not isinstance(review_data, dict):
        return None

    original_items = review_data.get("items")
    if not isinstance(original_items, list):
        return None

    display_items = _dictionary_review_items_for_display(
        original_items, form_items=form_items
    )

    return {
        "work_id": work_dir.name,
        "title": _work_title(work_dir.name, meta),
        "status": str(meta.get("status") or "unknown"),
        "status_label": status_label(
            _effective_status(work_dir, str(meta.get("status") or ""))
        ),
        "back_to_detail_ready": (
            work_dir / workspace_paths.PROCESSED_JSON_FILENAME
        ).exists(),
        "can_add_dictionary_item": not (
            work_dir / workspace_paths.PROCESSED_DRAFT_FILENAME
        ).exists()
        and not (work_dir / workspace_paths.PROCESSED_FINAL_FILENAME).exists(),
        "items": display_items,
        "item_count": len(display_items),
        "opened_file": workspace_paths.DICTIONARY_REVIEW_FILENAME,
        "decision_labels": DICTIONARY_REVIEW_DECISION_LABELS,
    }


def create_manual_dictionary_review_submission(
    workspace_root: Path,
    work_id: str,
) -> dict | None:
    work_dir = resolve_work_dir(workspace_root, work_id)
    if work_dir is None:
        return None

    edit_artifacts = detect_existing_edit_artifacts(work_dir)
    if edit_artifacts["should_block"]:
        return _dictionary_review_result(
            "failed",
            "読みと息継ぎ編集の途中データまたは確定データがあるため、手動辞書を作成できません。",
            failed_stage="preflight",
            error_type="existing_edit_artifacts",
            error_message="editing artifact already exists",
            affected_file=", ".join(edit_artifacts["existing_files"]),
            committed=False,
            meta_status_changed=False,
            existing_edit_artifacts=edit_artifacts["existing_files"],
            future_recovery_hint="dictionary_change_reapply_to_draft_required",
        )

    conflict = _manual_dictionary_review_conflict(work_dir)
    if conflict is not None:
        return conflict

    write_json(
        workspace_paths.dictionary_review_path(work_dir),
        build_manual_dictionary_review(input_stem=_detail_input_stem(work_dir)),
    )
    workspace_paths.update_work_meta_status(work_dir, "dictionary_review_ready")
    return _dictionary_review_result(
        "success",
        "手動辞書を作成しました。",
        committed=True,
        meta_status_changed=True,
    )


def create_ai_dictionary_review_submission(
    workspace_root: Path,
    work_id: str,
    *,
    cancel_event: Event | None = None,
    before_commit: Callable[[], bool] | None = None,
) -> dict | None:
    work_dir = resolve_work_dir(workspace_root, work_id)
    if work_dir is None:
        return None

    meta_path = work_dir / workspace_paths.WORK_META_FILENAME
    original_meta = _read_optional_dict(meta_path)
    original_status = str(original_meta.get("status") or "")
    edit_artifacts = detect_existing_edit_artifacts(work_dir)
    if edit_artifacts["should_block"]:
        return _dictionary_review_result(
            "failed",
            "読みと息継ぎの編集を始めた後は、AI辞書候補の作り直しはできません。必要な場合は、新しい作品として取り込み直してください。",
            failed_stage="preflight",
            error_type="existing_edit_artifacts",
            error_message="editing artifact already exists",
            affected_file=", ".join(edit_artifacts["existing_files"]),
            committed=False,
            meta_status_changed=False,
            existing_edit_artifacts=edit_artifacts["existing_files"],
            future_recovery_hint="reimport_as_new_work_required",
        )

    conflict = _ai_dictionary_review_conflict(work_dir)
    if conflict is not None:
        return conflict

    backup: dict | None = None
    try:
        backup = _backup_existing_files(
            work_dir,
            "gemini_dictionary_reset",
            GEMINI_DICTIONARY_RESET_BACKUP_FILENAMES,
        )
        step5_result = run_step5_dictionary_draft(
            work_dir=work_dir,
            project_root=workspace_paths.PROJECT_ROOT,
            run_gemini=True,
            cancel_event=cancel_event,
            before_commit=before_commit,
        )
        step5_status = str(step5_result.get("status") or "")
        if step5_status in {"missing_api_key", "import_error"}:
            _restore_backup_files(
                work_dir,
                backup,
                GEMINI_DICTIONARY_RESET_BACKUP_FILENAMES,
            )
            _restore_meta_status(meta_path, original_meta, original_status)
            return _ai_dictionary_review_failed_result(step5_result, step5_status)

        draft_path = workspace_paths.work_dictionary_draft_path(work_dir)
        draft_data = read_json(draft_path, default=[])
        draft_count = len(draft_data) if isinstance(draft_data, list) else 0
        if step5_status == "completed_with_errors" and draft_count == 0:
            _restore_backup_files(
                work_dir,
                backup,
                GEMINI_DICTIONARY_RESET_BACKUP_FILENAMES,
            )
            _restore_meta_status(meta_path, original_meta, original_status)
            return _ai_dictionary_review_failed_result(step5_result, step5_status)

        review_data = build_dictionary_review(
            draft_data,
            input_stem=_detail_input_stem(work_dir),
            generated_from=workspace_paths.display_path(draft_path, work_dir),
        )
        write_json(workspace_paths.dictionary_review_path(work_dir), review_data)
        workspace_paths.update_work_meta_status(work_dir, "dictionary_review_ready")
    except AIDictionaryReviewCancelled:
        _restore_backup_files(
            work_dir,
            backup,
            GEMINI_DICTIONARY_RESET_BACKUP_FILENAMES,
        )
        _restore_meta_status(meta_path, original_meta, original_status)
        raise
    except Exception as error:
        _restore_backup_files(
            work_dir,
            backup,
            GEMINI_DICTIONARY_RESET_BACKUP_FILENAMES,
        )
        _restore_meta_status(meta_path, original_meta, original_status)
        return _dictionary_review_result(
            "failed",
            "AI辞書候補の準備に失敗しました。",
            failed_stage="step5_dictionary_draft",
            error_type=type(error).__name__,
            error_message=str(error),
            affected_file=workspace_paths.WORK_DICTIONARY_DRAFT_FILENAME,
            committed=False,
            meta_status_changed=False,
        )

    candidate_count = int(step5_result.get("candidate_count") or 0)
    draft_count = int(step5_result.get("draft_count") or 0)
    step5_status = str(step5_result.get("status") or "")
    if candidate_count == 0:
        message = "AIに確認する候補語は見つかりませんでした。辞書を使わずに進むか、手動で辞書項目を追加してください。"
    elif step5_status == "completed_with_errors":
        message = "AI辞書候補を一部作成しました。一部の候補は取得できませんでした。辞書編集画面で内容を確認してください。"
    else:
        message = "AI辞書候補を作成しました。必要に応じて「辞書を編集する」から内容を確認してください。"
    result = _dictionary_review_result(
        "success",
        message,
        committed=True,
        meta_status_changed=True,
    )
    result["title"] = "AI辞書候補"
    result["candidate_count"] = candidate_count
    result["draft_count"] = draft_count
    result["api_called"] = bool(step5_result.get("api_called"))
    result["skip_reason"] = step5_status if not step5_result.get("api_called") else ""
    result["failed_count"] = int(step5_result.get("failed_count") or 0)
    result["timeout_count"] = int(step5_result.get("timeout_count") or 0)
    result["backup_dir"] = (
        str(backup.get("backup_dir")) if backup and backup.get("backup_dir") else ""
    )
    result["warnings"] = _ai_dictionary_review_warnings(work_dir, step5_result)
    return result


def load_ai_dictionary_review_confirmation(
    workspace_root: Path,
    work_id: str,
) -> dict | None:
    work_dir = resolve_work_dir(workspace_root, work_id)
    if work_dir is None:
        return None

    meta = _read_optional_dict(work_dir / workspace_paths.WORK_META_FILENAME)
    edit_artifacts = detect_existing_edit_artifacts(work_dir)
    if edit_artifacts["should_block"]:
        return {
            "status": "blocked",
            "work_id": work_dir.name,
            "title": _work_title(work_dir.name, meta),
            "result": _dictionary_review_result(
                "failed",
                "読みと息継ぎの編集を始めた後は、AI辞書候補の作り直しはできません。必要な場合は、新しい作品として取り込み直してください。",
                failed_stage="preflight",
                error_type="existing_edit_artifacts",
                error_message="editing artifact already exists",
                affected_file=", ".join(edit_artifacts["existing_files"]),
                committed=False,
                meta_status_changed=False,
                existing_edit_artifacts=edit_artifacts["existing_files"],
                future_recovery_hint="reimport_as_new_work_required",
            ),
        }

    conflict = _ai_dictionary_review_conflict(work_dir)
    if conflict is not None:
        return {
            "status": "blocked",
            "work_id": work_dir.name,
            "title": _work_title(work_dir.name, meta),
            "result": conflict,
        }

    return {
        "status": "ready",
        "work_id": work_dir.name,
        "title": _work_title(work_dir.name, meta),
        "active_generation_lock": _active_generation_lock_info(work_dir),
        "has_dictionary_review": (
            work_dir / workspace_paths.DICTIONARY_REVIEW_FILENAME
        ).exists(),
    }


def _ai_dictionary_review_failed_result(step5_result: dict, status: str) -> dict:
    if status == "missing_api_key":
        result = _dictionary_review_result(
            "failed",
            "AIの準備ができていないため、辞書候補を作成できませんでした。",
            failed_stage="step5_dictionary_draft",
            error_type="missing_api_key",
            error_message="GEMINI_API_KEY が設定されていません。.env または環境変数に GEMINI_API_KEY を設定してください。",
            affected_file=workspace_paths.WORK_DICTIONARY_DRAFT_FILENAME,
            committed=False,
            meta_status_changed=False,
        )
    elif status == "import_error":
        result = _dictionary_review_result(
            "failed",
            "AI辞書候補を作成できませんでした。時間をおいてもう一度試すか、手動で辞書項目を追加してください。",
            failed_stage="step5_dictionary_draft",
            error_type="gemini_import_error",
            error_message="Gemini API library import error",
            affected_file=workspace_paths.WORK_DICTIONARY_DRAFT_FILENAME,
            committed=False,
            meta_status_changed=False,
        )
    else:
        result = _dictionary_review_result(
            "failed",
            "AI辞書候補を作成できませんでした。時間をおいてもう一度試すか、手動で辞書項目を追加してください。",
            failed_stage="step5_dictionary_draft",
            error_type="gemini_api_error",
            error_message=_ai_dictionary_review_error_summary(step5_result),
            affected_file=workspace_paths.WORK_DICTIONARY_DRAFT_FILENAME,
            committed=False,
            meta_status_changed=False,
        )
    result["title"] = "AI辞書候補"
    result["candidate_count"] = int(step5_result.get("candidate_count") or 0)
    result["draft_count"] = int(step5_result.get("draft_count") or 0)
    result["api_called"] = bool(step5_result.get("api_called"))
    result["skip_reason"] = status if not step5_result.get("api_called") else ""
    result["failed_count"] = int(step5_result.get("failed_count") or 0)
    result["timeout_count"] = int(step5_result.get("timeout_count") or 0)
    return result


def _ai_dictionary_review_error_summary(step5_result: dict) -> str:
    if int(step5_result.get("timeout_count") or 0) > 0:
        return "timeout"
    if int(step5_result.get("failed_count") or 0) > 0:
        return "Gemini API error"
    return "Gemini API error"


def _ai_dictionary_review_warnings(work_dir: Path, step5_result: dict) -> list[dict]:
    warnings: list[dict] = []
    ai_review_terms = read_json(work_dir / "step5" / "ai_review_terms.json", default={})
    total_candidates = 0
    if isinstance(ai_review_terms, dict):
        try:
            total_candidates = int(ai_review_terms.get("total_candidates") or 0)
        except (TypeError, ValueError):
            total_candidates = 0
    candidate_count = int(step5_result.get("candidate_count") or 0)
    if total_candidates > candidate_count and candidate_count >= MAX_REVIEW_TERMS:
        warnings.append(
            {"message": f"{MAX_REVIEW_TERMS}件を超える候補が見つかりましたが、今回は最大{MAX_REVIEW_TERMS}件までAIに確認しました。"}
        )
    return warnings


def add_dictionary_review_item_submission(
    workspace_root: Path,
    work_id: str,
    form_data: Mapping[str, str],
) -> dict | None:
    work_dir = resolve_work_dir(workspace_root, work_id)
    if work_dir is None:
        return None

    review_path = workspace_paths.dictionary_review_path(work_dir)
    original_review = _read_required_dictionary_review(review_path)
    if not isinstance(original_review, dict):
        return _dictionary_review_result(
            "failed",
            "辞書の確認内容に問題があります。",
            failed_stage="load_review",
            error_type="missing_or_invalid_review",
            error_message="01_dictionary_review.json を読み込めませんでした。",
            affected_file=workspace_paths.DICTIONARY_REVIEW_FILENAME,
            committed=False,
            meta_status_changed=False,
        )

    edit_artifacts = detect_existing_edit_artifacts(work_dir)
    if edit_artifacts["should_block"]:
        return _dictionary_review_result(
            "failed",
            "読みと息継ぎ編集の途中データまたは確定データがあるため、辞書項目を追加できません。",
            failed_stage="preflight",
            error_type="existing_edit_artifacts",
            error_message="editing artifact already exists",
            affected_file=", ".join(edit_artifacts["existing_files"]),
            committed=False,
            meta_status_changed=False,
            existing_edit_artifacts=edit_artifacts["existing_files"],
            future_recovery_hint="dictionary_change_reapply_to_draft_required",
        )

    form_items = _dictionary_review_form_items(form_data)
    try:
        updated_review = _append_unsaved_dictionary_review_item(
            original_review,
            form_items,
            word=form_data.get("new_word") or form_data.get("word"),
            reading=form_data.get("new_reading") or form_data.get("reading"),
            notes=form_data.get("new_notes") or form_data.get("notes"),
        )
    except DictionaryReviewValidationError as error:
        return _dictionary_review_result(
            "failed",
            _dictionary_review_validation_summary(error.errors),
            failed_stage="validate_manual_item",
            error_type="validation_error",
            error_message=_dictionary_review_validation_summary(error.errors),
            affected_file=workspace_paths.DICTIONARY_REVIEW_FILENAME,
            committed=False,
            meta_status_changed=False,
            errors=error.errors,
            form_items=form_items,
        )

    return _dictionary_review_result(
        "success",
        "辞書項目を追加しました。",
        committed=False,
        meta_status_changed=False,
        form_items=_dictionary_review_items_as_form_items(updated_review),
    )


def finalize_dictionary_review_submission(
    workspace_root: Path,
    work_id: str,
    form_data: Mapping[str, str],
) -> dict | None:
    work_dir = resolve_work_dir(workspace_root, work_id)
    if work_dir is None:
        return None

    meta_path = work_dir / workspace_paths.WORK_META_FILENAME
    original_meta = _read_optional_dict(meta_path)
    original_status = str(original_meta.get("status") or "")
    review_path = workspace_paths.dictionary_review_path(work_dir)
    original_review = _read_required_dictionary_review(review_path)
    if not isinstance(original_review, dict):
        return _dictionary_review_result(
            "failed",
            "辞書の確認内容に問題があります。",
            failed_stage="load_review",
            error_type="missing_or_invalid_review",
            error_message="01_dictionary_review.json を読み込めませんでした。",
            affected_file=workspace_paths.DICTIONARY_REVIEW_FILENAME,
            committed=False,
            meta_status_changed=False,
        )

    edit_artifacts = detect_existing_edit_artifacts(work_dir)
    if edit_artifacts["should_block"]:
        return _dictionary_review_result(
            "failed",
            "読みと息継ぎ編集の途中データまたは確定データがあるため、編集用データを再作成できません。",
            failed_stage="preflight",
            error_type="existing_edit_artifacts",
            error_message="editing artifact already exists",
            affected_file=", ".join(edit_artifacts["existing_files"]),
            committed=False,
            meta_status_changed=False,
            existing_edit_artifacts=edit_artifacts["existing_files"],
            future_recovery_hint="dictionary_change_reapply_to_draft_required",
        )

    form_items = _dictionary_review_form_items(form_data)
    try:
        updated_review = _apply_dictionary_review_form_update_with_added_items(
            original_review, form_items
        )
    except DictionaryReviewValidationError as error:
        return _dictionary_review_result(
            "failed",
            "辞書の確認内容に問題があります。",
            failed_stage="validate_review",
            error_type="validation_error",
            error_message=_dictionary_review_validation_summary(error.errors),
            affected_file=workspace_paths.DICTIONARY_REVIEW_FILENAME,
            committed=False,
            meta_status_changed=False,
            errors=error.errors,
            form_items=form_items,
        )

    write_json(review_path, updated_review)
    return _run_review_to_processed_pipeline(
        work_dir,
        meta_path=meta_path,
        original_meta=original_meta,
        original_status=original_status,
        success_message="辞書を確定し、編集用データを作成しました。",
        form_items=form_items,
    )


def finalize_dictionary_review_direct_submission(
    workspace_root: Path,
    work_id: str,
) -> dict | None:
    work_dir = resolve_work_dir(workspace_root, work_id)
    if work_dir is None:
        return None

    meta_path = work_dir / workspace_paths.WORK_META_FILENAME
    original_meta = _read_optional_dict(meta_path)
    original_status = str(original_meta.get("status") or "")
    review_path = workspace_paths.dictionary_review_path(work_dir)
    original_review = _read_required_dictionary_review(review_path)
    if not isinstance(original_review, dict):
        return _dictionary_review_result(
            "failed",
            "辞書編集画面で確認してください。",
            failed_stage="load_review",
            error_type="missing_or_invalid_review",
            error_message="01_dictionary_review.json を読み込めませんでした。",
            affected_file=workspace_paths.DICTIONARY_REVIEW_FILENAME,
            committed=False,
            meta_status_changed=False,
        )

    edit_artifacts = detect_existing_edit_artifacts(work_dir)
    if edit_artifacts["should_block"]:
        return _dictionary_review_result(
            "failed",
            "読みと息継ぎの編集結果があります。辞書を変更する場合は、編集済みデータへの反映方法を別途選ぶ必要があります。",
            failed_stage="preflight",
            error_type="existing_edit_artifacts",
            error_message="editing artifact already exists",
            affected_file=", ".join(edit_artifacts["existing_files"]),
            committed=False,
            meta_status_changed=False,
            existing_edit_artifacts=edit_artifacts["existing_files"],
            future_recovery_hint="dictionary_change_reapply_to_draft_required",
        )

    form_items = _dictionary_review_items_as_form_items(original_review)
    try:
        updated_review = apply_dictionary_review_form_update(
            original_review, form_items
        )
    except DictionaryReviewValidationError as error:
        return _dictionary_review_result(
            "failed",
            "辞書編集画面で確認してください。",
            failed_stage="validate_review",
            error_type="validation_error",
            error_message=_dictionary_review_validation_summary(error.errors),
            affected_file=workspace_paths.DICTIONARY_REVIEW_FILENAME,
            committed=False,
            meta_status_changed=False,
            errors=error.errors,
            form_items=form_items,
        )

    pending_errors = _direct_dictionary_finalize_pending_errors(updated_review)
    if pending_errors:
        return _dictionary_review_result(
            "failed",
            "辞書編集画面で確認してください。",
            failed_stage="validate_review",
            error_type="pending_reading_final_required",
            error_message=_dictionary_review_validation_summary(pending_errors),
            affected_file=workspace_paths.DICTIONARY_REVIEW_FILENAME,
            committed=False,
            meta_status_changed=False,
            errors=pending_errors,
            form_items=form_items,
        )

    backup: dict | None = None
    try:
        write_json(review_path, updated_review)
        backup = _backup_existing_files(
            work_dir,
            "dictionary_finalize",
            DICTIONARY_FINALIZE_BACKUP_FILENAMES,
        )
        result = _run_review_to_processed_pipeline(
            work_dir,
            meta_path=meta_path,
            original_meta=original_meta,
            original_status=original_status,
            success_message="辞書を確定し、編集用データを作成しました。",
            form_items=form_items,
        )
        if result.get("status") != "success":
            write_json(review_path, original_review)
            _restore_backup_files(
                work_dir,
                backup,
                DICTIONARY_FINALIZE_BACKUP_FILENAMES,
            )
            _restore_meta_status(meta_path, original_meta, original_status)
        return result
    except Exception as error:
        write_json(review_path, original_review)
        _restore_backup_files(
            work_dir,
            backup,
            DICTIONARY_FINALIZE_BACKUP_FILENAMES,
        )
        _restore_meta_status(meta_path, original_meta, original_status)
        return _dictionary_review_result(
            "failed",
            "編集用データの作成に失敗しました。",
            failed_stage="direct_finalize",
            error_type=type(error).__name__,
            error_message=str(error),
            affected_file=workspace_paths.PROCESSED_JSON_FILENAME,
            committed=False,
            meta_status_changed=False,
            form_items=form_items,
        )


def create_empty_dictionary_processed_submission(
    workspace_root: Path,
    work_id: str,
) -> dict | None:
    work_dir = resolve_work_dir(workspace_root, work_id)
    if work_dir is None:
        return None

    meta_path = work_dir / workspace_paths.WORK_META_FILENAME
    original_meta = _read_optional_dict(meta_path)
    original_status = str(original_meta.get("status") or "")
    edit_artifacts = detect_existing_edit_artifacts(work_dir)
    if edit_artifacts["should_block"]:
        return _dictionary_review_result(
            "failed",
            "読みと息継ぎ編集の途中データまたは確定データがあるため、編集用データを再作成できません。",
            failed_stage="preflight",
            error_type="existing_edit_artifacts",
            error_message="editing artifact already exists",
            affected_file=", ".join(edit_artifacts["existing_files"]),
            committed=False,
            meta_status_changed=False,
            existing_edit_artifacts=edit_artifacts["existing_files"],
            future_recovery_hint="dictionary_change_reapply_to_draft_required",
        )

    conflict = _empty_dictionary_conflict(work_dir, original_meta)
    if conflict is not None:
        return conflict

    review_path = workspace_paths.dictionary_review_path(work_dir)
    input_stem = _detail_input_stem(work_dir)
    write_json(review_path, build_empty_dictionary_review(input_stem=input_stem))
    return _run_review_to_processed_pipeline(
        work_dir,
        meta_path=meta_path,
        original_meta=original_meta,
        original_status=original_status,
        success_message="辞書を使わずに編集用データを作成しました。",
    )


def load_generation_placeholder(
    workspace_root: Path, work_id: str, generation_kind: str
) -> dict | None:
    work_dir = resolve_work_dir(workspace_root, work_id)
    if work_dir is None:
        return None

    meta = _read_optional_dict(work_dir / workspace_paths.WORK_META_FILENAME)
    has_final = (work_dir / workspace_paths.PROCESSED_FINAL_FILENAME).exists()
    has_draft = (work_dir / workspace_paths.PROCESSED_DRAFT_FILENAME).exists()
    return {
        "work_id": work_dir.name,
        "title": _work_title(work_dir.name, meta),
        "generation_kind": generation_kind,
        "generation_label": GENERATION_LABELS.get(generation_kind, generation_kind),
        "has_final": has_final,
        "can_generate": has_final and not has_draft,
        "input_file": workspace_paths.PROCESSED_FINAL_FILENAME if has_final else "",
    }


def start_reedit_from_final(
    workspace_root: Path, work_id: str, form_data: Mapping[str, str]
) -> dict | None:
    work_dir = resolve_work_dir(workspace_root, work_id)
    if work_dir is None:
        return None

    final_path = work_dir / workspace_paths.PROCESSED_FINAL_FILENAME
    draft_path = work_dir / workspace_paths.PROCESSED_DRAFT_FILENAME
    if not final_path.exists():
        return {"status": "failed", "error": "missing_final"}
    if draft_path.exists() and form_data.get("confirm") != "1":
        return {"status": "conflict", "error": "draft_exists"}

    final_data = read_json(final_path)
    if not isinstance(final_data, dict):
        return {"status": "failed", "error": "invalid_final_json"}

    now = _jst_now_iso()
    draft_data = copy.deepcopy(final_data)
    draft_data["edit_state"] = "draft"
    draft_data["source"] = workspace_paths.PROCESSED_FINAL_FILENAME
    draft_data["updated_at"] = now
    write_json(draft_path, draft_data)
    _update_draft_meta(work_dir, now)
    _write_reedit_editing_session(work_dir, now)

    return {
        "status": "success",
        "source_file": workspace_paths.PROCESSED_FINAL_FILENAME,
        "draft_file": workspace_paths.PROCESSED_DRAFT_FILENAME,
    }


def save_work_draft(
    workspace_root: Path, work_id: str, form_data: Mapping[str, str]
) -> dict | None:
    work_dir = resolve_work_dir(workspace_root, work_id)
    if work_dir is None:
        return None

    selected = select_draft_save_base(work_dir)
    if selected["path"] is None:
        return {
            "status": "failed",
            "error": "draft_save_unavailable",
            "mode": selected["mode"],
            "base_file": selected["filename"],
        }

    source_data = read_json(selected["path"])
    if not isinstance(source_data, dict):
        return {
            "status": "failed",
            "error": "invalid_processed_json",
            "mode": selected["mode"],
            "base_file": selected["filename"],
        }

    draft_data = copy.deepcopy(source_data)
    segments = _processed_segments_for_update(draft_data)
    if segments is None:
        return {
            "status": "failed",
            "error": "missing_segments",
            "mode": selected["mode"],
            "base_file": selected["filename"],
        }

    posted_edits = _audio_edits_from_form(form_data)
    updated_count = 0
    for segment in segments:
        if not isinstance(segment, dict) or _is_image_segment(segment):
            continue
        segment_index = str(segment.get("index", ""))
        if segment_index not in posted_edits:
            continue
        source_audio = str(segment.get("audio") or "")
        try:
            segment["audio"] = edit_text_to_audio(
                posted_edits[segment_index],
                source_audio=source_audio,
            )
        except ValueError as error:
            return {
                "status": "failed",
                "error": "invalid_audio_edit",
                "message": str(error),
                "mode": selected["mode"],
                "base_file": selected["filename"],
                "segment_index": segment_index,
                "posted_edits": posted_edits,
            }
        updated_count += 1

    now = _jst_now_iso()
    draft_data["edit_state"] = "draft"
    draft_data["source"] = selected["filename"]
    draft_data["updated_at"] = now
    write_json(work_dir / workspace_paths.PROCESSED_DRAFT_FILENAME, draft_data)
    _update_draft_meta(work_dir, now)
    _write_draft_editing_session(work_dir, now)

    return {
        "status": "success",
        "mode": selected["mode"],
        "base_file": selected["filename"],
        "draft_file": workspace_paths.PROCESSED_DRAFT_FILENAME,
        "updated_count": updated_count,
    }


def save_work_final(
    workspace_root: Path, work_id: str, form_data: Mapping[str, str]
) -> dict | None:
    work_dir = resolve_work_dir(workspace_root, work_id)
    if work_dir is None:
        return None

    selected = select_final_save_base(work_dir)
    if selected["path"] is None:
        return {
            "status": "failed",
            "error": "final_save_unavailable",
            "mode": selected["mode"],
            "base_file": selected["filename"],
        }

    source_data = read_json(selected["path"])
    if not isinstance(source_data, dict):
        return {
            "status": "failed",
            "error": "invalid_processed_json",
            "mode": selected["mode"],
            "base_file": selected["filename"],
        }

    final_data = copy.deepcopy(source_data)
    segments = _processed_segments_for_update(final_data)
    if segments is None:
        return {
            "status": "failed",
            "error": "missing_segments",
            "mode": selected["mode"],
            "base_file": selected["filename"],
        }

    posted_edits = _audio_edits_from_form(form_data)
    updated_count = 0
    for segment in segments:
        if not isinstance(segment, dict) or _is_image_segment(segment):
            continue
        segment_index = str(segment.get("index", ""))
        if segment_index not in posted_edits:
            continue
        source_audio = str(segment.get("audio") or "")
        try:
            segment["audio"] = edit_text_to_audio(
                posted_edits[segment_index],
                source_audio=source_audio,
            )
        except ValueError as error:
            return {
                "status": "failed",
                "error": "invalid_audio_edit",
                "message": str(error),
                "mode": selected["mode"],
                "base_file": selected["filename"],
                "segment_index": segment_index,
                "posted_edits": posted_edits,
            }
        updated_count += 1

    now = _jst_now_iso()
    final_data["edit_state"] = "final"
    final_data["source"] = selected["filename"]
    final_data["updated_at"] = now
    final_path = work_dir / workspace_paths.PROCESSED_FINAL_FILENAME
    write_json(final_path, final_data)
    if final_path.exists():
        _cleanup_draft_after_final(work_dir)
    _update_final_meta(work_dir, now)

    return {
        "status": "success",
        "mode": selected["mode"],
        "base_file": selected["filename"],
        "final_file": workspace_paths.PROCESSED_FINAL_FILENAME,
        "updated_count": updated_count,
    }


def resolve_work_dir(workspace_root: Path, work_id: str) -> Path | None:
    if not is_safe_work_id(work_id):
        return None
    workspace_root = Path(workspace_root)
    candidate = workspace_root / work_id
    if not candidate.is_dir():
        return None
    workspace_resolved = workspace_root.resolve(strict=False)
    try:
        candidate_resolved = candidate.resolve(strict=True)
    except FileNotFoundError:
        return None
    if candidate_resolved.parent != workspace_resolved:
        return None
    return candidate_resolved


def load_work_delete_confirmation(workspace_root: Path, work_id: str) -> dict | None:
    work_dir = resolve_deletable_work_dir(workspace_root, work_id)
    if work_dir is None:
        return None
    meta = _read_optional_dict(work_dir / workspace_paths.WORK_META_FILENAME)
    return {
        "work_id": work_dir.name,
        "title": _work_title(work_dir.name, meta),
        "created_at": _datetime_label(str(meta.get("created_at") or ""))
        or _work_created_label(work_dir.name),
        "updated_at": _datetime_label(str(meta.get("updated_at") or "")),
    }


def move_work_to_trash(workspace_root: Path, work_id: str) -> dict:
    work_dir = resolve_deletable_work_dir(workspace_root, work_id)
    if work_dir is None:
        return {
            "status": "failed",
            "message": "作品を削除できませんでした。時間をおいてもう一度試してください。",
            "error": "invalid_work_id_or_missing_work",
        }

    trash_root = Path(workspace_root) / "_trash"
    timestamp = datetime.now(workspace_paths.JST).strftime("%Y%m%d-%H%M%S")
    trash_root.mkdir(parents=True, exist_ok=True)
    destination = _unique_trash_destination(trash_root, work_dir.name, timestamp)
    try:
        shutil.move(str(work_dir), str(destination))
    except Exception as error:
        return {
            "status": "failed",
            "message": "作品を削除できませんでした。時間をおいてもう一度試してください。",
            "error": f"{type(error).__name__}: {error}",
        }
    return {
        "status": "success",
        "message": "作品を一覧から削除しました。",
        "trash_path": str(destination),
    }


def resolve_deletable_work_dir(workspace_root: Path, work_id: str) -> Path | None:
    if not is_safe_work_id(work_id) or _is_reserved_work_id(work_id):
        return None
    work_dir = resolve_work_dir(workspace_root, work_id)
    if work_dir is None:
        return None
    workspace_resolved = Path(workspace_root).resolve(strict=False)
    if work_dir.parent != workspace_resolved:
        return None
    if not (work_dir / workspace_paths.WORK_META_FILENAME).is_file():
        return None
    return work_dir


def is_safe_work_id(work_id: str) -> bool:
    if not work_id or work_id in {".", ".."}:
        return False
    if "/" in work_id or "\\" in work_id:
        return False
    return Path(work_id).name == work_id


def select_processed_file(work_dir: Path, view: str | None = None) -> dict:
    final_path = work_dir / workspace_paths.PROCESSED_FINAL_FILENAME
    if view == "final" and final_path.exists():
        return {
            "path": final_path,
            "filename": workspace_paths.PROCESSED_FINAL_FILENAME,
            "mode": "final_readonly",
        }
    candidates = (
        (workspace_paths.PROCESSED_DRAFT_FILENAME, "draft"),
        (workspace_paths.PROCESSED_FINAL_FILENAME, "final_readonly"),
        (workspace_paths.PROCESSED_JSON_FILENAME, "processed"),
    )
    for filename, mode in candidates:
        path = work_dir / filename
        if path.exists():
            return {"path": path, "filename": filename, "mode": mode}
    return {"path": None, "filename": "none", "mode": "unavailable"}


def select_draft_save_base(work_dir: Path) -> dict:
    draft_path = work_dir / workspace_paths.PROCESSED_DRAFT_FILENAME
    if draft_path.exists():
        return {
            "path": draft_path,
            "filename": workspace_paths.PROCESSED_DRAFT_FILENAME,
            "mode": "draft",
        }
    processed_path = work_dir / workspace_paths.PROCESSED_JSON_FILENAME
    if processed_path.exists():
        return {
            "path": processed_path,
            "filename": workspace_paths.PROCESSED_JSON_FILENAME,
            "mode": "processed",
        }
    final_path = work_dir / workspace_paths.PROCESSED_FINAL_FILENAME
    if final_path.exists():
        return {
            "path": None,
            "filename": workspace_paths.PROCESSED_FINAL_FILENAME,
            "mode": "final_readonly",
        }
    return {"path": None, "filename": "none", "mode": "unavailable"}


def select_final_save_base(work_dir: Path) -> dict:
    draft_path = work_dir / workspace_paths.PROCESSED_DRAFT_FILENAME
    if draft_path.exists():
        return {
            "path": draft_path,
            "filename": workspace_paths.PROCESSED_DRAFT_FILENAME,
            "mode": "draft",
        }
    processed_path = work_dir / workspace_paths.PROCESSED_JSON_FILENAME
    if processed_path.exists():
        return {
            "path": processed_path,
            "filename": workspace_paths.PROCESSED_JSON_FILENAME,
            "mode": "processed",
        }
    final_path = work_dir / workspace_paths.PROCESSED_FINAL_FILENAME
    if final_path.exists():
        return {
            "path": None,
            "filename": workspace_paths.PROCESSED_FINAL_FILENAME,
            "mode": "final_readonly",
        }
    return {"path": None, "filename": "none", "mode": "unavailable"}


def build_segment_views(
    processed_data: Any,
    symbol_to_category: dict,
    posted_audio_edits: Mapping[str, str] | None = None,
    segment_errors: Mapping[str, str] | None = None,
) -> list[dict]:
    views: list[dict] = []
    override_map = {
        str(key): str(value) for key, value in (posted_audio_edits or {}).items()
    }
    error_map = {str(key): str(value) for key, value in (segment_errors or {}).items()}
    for segment in _processed_segments(processed_data):
        if _is_image_segment(segment):
            continue
        segment_index = str(segment.get("index", ""))
        source_audio = str(segment.get("audio") or "")
        audio_edit_text = audio_to_edit_text(source_audio)
        if segment_index in override_map:
            audio_edit_text = override_map[segment_index]
            audio_spans = build_audio_edit_spans(audio_edit_text, symbol_to_category)
        else:
            audio_spans = build_audio_edit_spans(
                audio_edit_text,
                symbol_to_category,
                source_audio=source_audio,
            )
        views.append(
            {
                "index": segment.get("index", ""),
                "display": _display_text(segment),
                "audio_edit_text": audio_edit_text,
                "audio_spans": audio_spans,
                "save_error": error_map.get(segment_index, ""),
            }
        )
    return views


def paginate_segments(
    segments: list[dict], *, page: object | None, per_page: object | None
) -> dict:
    selected_per_page = _normalize_per_page(per_page)
    total = len(segments)
    total_pages = max(1, math.ceil(total / selected_per_page)) if total else 1
    selected_page = _normalize_page(page, total_pages)
    start = (selected_page - 1) * selected_per_page
    end = start + selected_per_page
    return {
        "segments": segments[start:end],
        "page": selected_page,
        "per_page": selected_per_page,
        "per_page_options": PER_PAGE_OPTIONS,
        "total": total,
        "total_pages": total_pages,
        "start": start + 1 if total else 0,
        "end": min(end, total),
        "has_prev": selected_page > 1,
        "has_next": selected_page < total_pages,
        "prev_page": max(1, selected_page - 1),
        "next_page": min(total_pages, selected_page + 1),
    }


def workflow_steps(status: str) -> list[dict]:
    labels = [
        "原稿取込",
        "辞書準備",
        "読みと息継ぎ編集",
        "編集完了",
        "EPUB作成",
        "完成",
    ]
    current_index = _workflow_current_index(status)
    steps: list[dict] = []
    for index, label in enumerate(labels):
        if index < current_index:
            state = "done"
            marker = "✓"
        elif index == current_index:
            state = "current"
            marker = "●"
        else:
            state = "pending"
            marker = ""
        steps.append({"label": label, "state": state, "marker": marker})
    return steps


def status_label(status: str) -> str:
    return STATUS_LABELS.get(status, "準備中")


def status_description(status: str, mode: str) -> str:
    if status == "dictionary_review_ready":
        return "作品辞書を読み込みました。辞書確認・修正に進めます。"
    if status == "dictionary_finalized":
        return "作品辞書は作成済みです。編集用データを作成できます。"
    return mode_description(mode)


def mode_label(mode: str) -> str:
    return MODE_LABELS.get(mode, "編集できません")


def mode_description(mode: str) -> str:
    return MODE_DESCRIPTIONS.get(mode, "辞書読み込みに進めます。")


def _normalize_per_page(value: object | None) -> int:
    try:
        number = int(str(value))
    except (TypeError, ValueError):
        return DEFAULT_PER_PAGE
    if number in PER_PAGE_OPTIONS:
        return number
    return DEFAULT_PER_PAGE


def _normalize_page(value: object | None, total_pages: int) -> int:
    try:
        number = int(str(value))
    except (TypeError, ValueError):
        return 1
    return min(max(number, 1), total_pages)


def _workflow_current_index(status: str) -> int:
    if status in {"source_created", "initialized", "pre_processed"}:
        return 0
    if status in {"dictionary_review_ready", "dictionary_finalized"}:
        return 1
    if status in {"processed", "draft_saved"}:
        return 2
    if status in {"finalized", "audio_generated"}:
        return 4
    if status == "exported":
        return 5
    return 0


def _effective_status(work_dir: Path, status: str) -> str:
    if status == "exported":
        return (
            "exported"
            if _has_official_epub(work_dir)
            else _safe_unexported_status(work_dir)
        )
    if status == "audio_generated":
        return (
            "audio_generated"
            if _has_audio_outputs(work_dir)
            else _safe_unexported_status(work_dir)
        )
    return status


def _safe_unexported_status(work_dir: Path) -> str:
    if (work_dir / workspace_paths.PROCESSED_FINAL_FILENAME).exists():
        return "finalized"
    if (work_dir / workspace_paths.PROCESSED_DRAFT_FILENAME).exists():
        return "draft_saved"
    if (work_dir / workspace_paths.PROCESSED_JSON_FILENAME).exists():
        return "processed"
    return "unknown"


def _has_audio_outputs(work_dir: Path) -> bool:
    return (work_dir / "audio" / "audio.mp3").is_file() and (
        work_dir / "audio" / "sync_map.json"
    ).is_file()


def _audio_needs_regeneration(workspace_root: Path, work_dir: Path) -> bool:
    final_path = work_dir / workspace_paths.PROCESSED_FINAL_FILENAME
    audio_path = work_dir / "audio" / "audio.mp3"
    sync_path = work_dir / "audio" / "sync_map.json"
    if not final_path.is_file():
        return False
    if not audio_path.is_file() or not sync_path.is_file():
        return True
    try:
        final_mtime = final_path.stat().st_mtime
        audio_mtime = audio_path.stat().st_mtime
        sync_mtime = sync_path.stat().st_mtime
        settings_path = _effective_tts_settings_path(workspace_root, work_dir)
        settings_mtime = settings_path.stat().st_mtime if settings_path is not None else None
    except OSError:
        return True
    if final_mtime > audio_mtime or final_mtime > sync_mtime:
        return True
    if settings_mtime is not None and (
        settings_mtime > audio_mtime or settings_mtime > sync_mtime
    ):
        return True
    return False


def _has_official_epub(work_dir: Path) -> bool:
    return _latest_official_epub(work_dir) is not None


def _has_current_epub(workspace_root: Path, work_dir: Path) -> bool:
    final_path = work_dir / workspace_paths.PROCESSED_FINAL_FILENAME
    latest_epub = _latest_official_epub(work_dir)
    if latest_epub is None or not final_path.is_file():
        return False
    try:
        latest_epub_mtime = latest_epub.stat().st_mtime
        required_paths = [
            final_path,
            work_dir / "audio" / "audio.mp3",
            work_dir / "audio" / "sync_map.json",
        ]
        settings_path = _effective_tts_settings_path(workspace_root, work_dir)
        if settings_path is not None:
            required_paths.append(settings_path)
        final_data = read_json(final_path)
        required_paths.extend(
            item["path"]
            for item in build_image_items(work_dir, final_data)
            if item.get("exists")
        )
        return all(latest_epub_mtime >= path.stat().st_mtime for path in required_paths)
    except (OSError, TypeError, ValueError):
        return False


def _epub_needs_regeneration(workspace_root: Path, work_dir: Path) -> bool:
    return (work_dir / workspace_paths.PROCESSED_FINAL_FILENAME).is_file() and not _has_current_epub(
        workspace_root, work_dir
    )


def _tts_settings_newer_than_outputs(workspace_root: Path, work_dir: Path) -> bool:
    settings_path = _effective_tts_settings_path(workspace_root, work_dir)
    if settings_path is None:
        return False
    output_paths = [
        work_dir / "audio" / "audio.mp3",
        work_dir / "audio" / "sync_map.json",
    ]
    latest_epub = _latest_official_epub(work_dir)
    if latest_epub is not None:
        output_paths.append(latest_epub)
    try:
        settings_mtime = settings_path.stat().st_mtime
        return any(path.is_file() and settings_mtime > path.stat().st_mtime for path in output_paths)
    except OSError:
        return False


def _effective_tts_settings_path(workspace_root: Path, work_dir: Path) -> Path | None:
    work_settings = work_tts_settings_path(work_dir)
    if work_settings.is_file():
        return work_settings
    workspace_settings = workspace_settings_path(workspace_root)
    if workspace_settings.is_file():
        return workspace_settings
    return None


def _active_generation_lock_info(work_dir: Path) -> dict:
    lock_data = read_generation_lock(work_dir)
    if lock_data is None:
        return {
            "active": False,
            "message": "",
            "operation": "",
            "task_id": "",
            "started_at": "",
            "lock_path": "",
        }
    return {
        "active": True,
        "message": "生成中です。完了後に再読み込みしてください。",
        "details_message": active_generation_lock_message(lock_data),
        "operation": str(lock_data.get("operation") or ""),
        "task_id": str(lock_data.get("task_id") or ""),
        "started_at": str(lock_data.get("started_at") or ""),
        "lock_path": str(generation_lock_path(work_dir)),
    }


def _work_created_label(work_id: str) -> str:
    match = re.search(r"(\d{8}-\d{6})$", work_id)
    if match:
        return datetime.strptime(match.group(1), "%Y%m%d-%H%M%S").strftime(
            "%Y-%m-%d %H:%M:%S"
        )
    return work_id


def _datetime_label(value: str) -> str:
    return value.split("+", 1)[0].split(".", 1)[0].replace("T", " ")


def _sort_works(works: list[dict], sort: str) -> list[dict]:
    if sort in {"updated_desc", "updated_asc"}:
        return _sort_works_by_datetime(works, "updated_at", sort == "updated_desc")
    if sort in {"created_desc", "created_asc"}:
        return _sort_works_by_datetime(works, "created_at", sort == "created_desc")
    if sort in {"title_asc", "title_desc"}:
        return sorted(
            works,
            key=lambda work: (
                str(work.get("title") or work.get("work_id") or ""),
                str(work.get("work_id") or ""),
            ),
            reverse=sort == "title_desc",
        )
    if sort == "status":
        return sorted(
            works,
            key=lambda work: (
                _workflow_current_index(str(work.get("effective_status") or "")),
                str(work.get("status_label") or ""),
                str(work.get("title") or work.get("work_id") or ""),
                str(work.get("work_id") or ""),
            ),
        )
    return works


def _sort_works_by_datetime(
    works: list[dict], field: str, descending: bool
) -> list[dict]:
    present = [
        work for work in works if _parse_datetime_value(work.get(field)) is not None
    ]
    missing = [
        work for work in works if _parse_datetime_value(work.get(field)) is None
    ]
    present_sorted = sorted(
        present,
        key=lambda work: (
            _datetime_order_value(work.get(field)),
            str(work.get("title") or work.get("work_id") or ""),
            str(work.get("work_id") or ""),
        ),
        reverse=descending,
    )
    return present_sorted + missing


def _datetime_order_value(value: object) -> float:
    parsed = _parse_datetime_value(value)
    if parsed is None:
        return float("-inf")
    return parsed.timestamp()


def _parse_datetime_value(value: object) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        pass
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y%m%d-%H%M%S"):
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue
    return None


def _latest_official_epub(work_dir: Path) -> Path | None:
    epub_dir = work_dir / "epub"
    if not epub_dir.is_dir():
        return None
    candidates = [path for path in epub_dir.glob("*.epub") if path.is_file()]
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


def _processed_segments(processed_data: Any) -> list[dict]:
    if isinstance(processed_data, dict):
        for key in ("remastered_data", "segments"):
            value = processed_data.get(key)
            if isinstance(value, list):
                return [item for item in value if isinstance(item, dict)]
    return []


def _processed_segments_for_update(processed_data: dict) -> list[Any] | None:
    for key in ("remastered_data", "segments"):
        value = processed_data.get(key)
        if isinstance(value, list):
            return value
    return None


def _image_item_by_index(work_dir: Path, segment_index: str) -> dict | None:
    summary = build_work_image_summary(work_dir)
    target_index = str(segment_index)
    for item in summary["items"]:
        if str(item.get("index") or "") == target_index:
            return item
    return None


def _safe_image_filename(image_file: str) -> str:
    raw = image_file.strip().replace("\\", "/")
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
    if Path(filename).suffix.lower() not in ALLOWED_IMAGE_UPLOAD_EXTENSIONS:
        return ""
    return filename


def _looks_like_allowed_image(path: Path) -> bool:
    try:
        header = path.read_bytes()[:12]
    except OSError:
        return False
    if header.startswith(b"\x89PNG\r\n\x1a\n"):
        return True
    return header.startswith(b"\xff\xd8\xff")


def _image_upload_result(
    status: str,
    message: str,
    error: str,
    *,
    segment_index: str = "",
    image_file: str = "",
) -> dict:
    return {
        "status": status,
        "message": message,
        "error": "" if status == "success" else error,
        "segment_index": segment_index,
        "image_file": image_file,
    }


def _audio_edits_from_form(form_data: Mapping[str, str]) -> dict[str, str]:
    edits: dict[str, str] = {}
    for key, value in form_data.items():
        if not key.startswith("audio_edit_"):
            continue
        index = key.removeprefix("audio_edit_")
        if index:
            edits[index] = str(value)
    return edits


def _is_image_segment(segment: dict) -> bool:
    return bool(segment.get("is_image")) or segment.get("block_type") == "image"


def _display_text(segment: dict) -> str:
    display = segment.get("display", "")
    if isinstance(display, dict):
        return str(display.get("text") or "")
    return str(display or "")


def _dictionary_review_items_for_display(
    original_items: list[dict],
    *,
    form_items: list[dict] | None = None,
) -> list[dict]:
    overrides = form_items if isinstance(form_items, list) else []
    display_items: list[dict] = []
    for position, original_item in enumerate(original_items):
        if not isinstance(original_item, dict):
            continue
        item = copy.deepcopy(original_item)
        if position < len(overrides) and isinstance(overrides[position], dict):
            override = overrides[position]
            item["reading_final"] = str(override.get("reading_final") or "")
            item["operation"] = _review_operation_value(
                override.get("operation"), override.get("decision")
            )
            item["notes"] = (
                "" if override.get("notes") is None else str(override.get("notes"))
            )
        else:
            item["operation"] = _review_operation_value(None, item.get("decision"))
        item["change_mark"] = _review_change_mark(item)
        item["decision_label"] = DICTIONARY_REVIEW_DECISION_LABELS.get(
            str(item.get("decision") or ""), ""
        )
        display_items.append(item)
    for override in overrides[len(original_items) :]:
        if not isinstance(override, dict):
            continue
        item = {
            "index": _normalize_review_index(override.get("index")),
            "word": str(override.get("word") or ""),
            "reading_suggested": str(
                override.get("reading_suggested")
                or override.get("reading_final")
                or ""
            ),
            "reading_final": str(override.get("reading_final") or ""),
            "meaning": "",
            "difficulty": None,
            "confidence": "manual",
            "decision": str(override.get("decision") or "accept"),
            "operation": _review_operation_value(
                override.get("operation"), override.get("decision")
            ),
            "target_dictionary": "work",
            "promote_to_user_dictionary": False,
            "source": "manual",
            "notes": "" if override.get("notes") is None else str(override.get("notes")),
        }
        item["change_mark"] = _review_change_mark(item)
        item["decision_label"] = DICTIONARY_REVIEW_DECISION_LABELS.get(
            str(item.get("decision") or ""), ""
        )
        display_items.append(item)
    return display_items


def _dictionary_review_form_items(form_data: Mapping[str, str]) -> list[dict]:
    try:
        item_count = int(str(form_data.get("item_count") or "0"))
    except (TypeError, ValueError):
        return (
            form_data.get("items") if isinstance(form_data.get("items"), list) else []
        )

    items: list[dict] = []
    for position in range(max(item_count, 0)):
        items.append(
            {
                "index": form_data.get(f"item_{position}_index"),
                "word": form_data.get(f"item_{position}_word"),
                "reading_suggested": form_data.get(
                    f"item_{position}_reading_suggested"
                ),
                "reading_final": form_data.get(f"item_{position}_reading_final"),
                "decision": form_data.get(f"item_{position}_decision"),
                "operation": form_data.get(f"item_{position}_operation"),
                "notes": form_data.get(f"item_{position}_notes"),
            }
        )
    return items


def _dictionary_review_items_as_form_items(review_data: dict) -> list[dict]:
    items = review_data.get("items")
    if not isinstance(items, list):
        return []
    form_items: list[dict] = []
    for item in items:
        if not isinstance(item, dict):
            form_items.append({})
            continue
        form_items.append(
            {
                "index": item.get("index"),
                "word": item.get("word"),
                "reading_suggested": item.get("reading_suggested"),
                "reading_final": item.get("reading_final"),
                "decision": item.get("decision"),
                "operation": _review_operation_value(
                    item.get("operation"), item.get("decision")
                ),
                "notes": item.get("notes"),
            }
        )
    return form_items


def _apply_dictionary_review_form_update_with_added_items(
    original_review: dict,
    form_items: list[dict],
) -> dict:
    original_items = original_review.get("items")
    if not isinstance(original_items, list):
        return apply_dictionary_review_form_update(original_review, form_items)

    review_with_added_items = copy.deepcopy(original_review)
    review_with_added_items["items"] = []
    update_items: list[dict] = []
    errors: list[dict] = []
    for position, form_item in enumerate(form_items, start=1):
        if not isinstance(form_item, dict):
            errors.append(
                {
                    "code": "invalid_form_item",
                    "message": "form item must be object",
                    "position": position,
                }
            )
            continue
        operation = _review_operation_value(form_item.get("operation"), form_item.get("decision"))
        if operation == "delete":
            continue

        if position <= len(original_items):
            original_item = original_items[position - 1]
            if not isinstance(original_item, dict):
                errors.append(
                    {
                        "code": "invalid_original_item",
                        "message": "original review item must be object",
                        "position": position,
                    }
                )
                continue
            review_with_added_items["items"].append(copy.deepcopy(original_item))
        else:
            before_count = len(review_with_added_items["items"])
            reading = form_item.get("reading_suggested") or form_item.get("reading_final")
            review_with_added_items = append_manual_dictionary_review_item(
                review_with_added_items,
                word=form_item.get("word"),
                reading=reading,
                notes=form_item.get("notes"),
            )
            added_item = review_with_added_items["items"][before_count]
            added_item["index"] = _normalize_review_index(form_item.get("index"))

        update_item = copy.deepcopy(form_item)
        update_item["decision"] = _decision_from_review_operation(operation, form_item)
        update_items.append(update_item)

    if errors:
        raise DictionaryReviewValidationError(errors)
    return apply_dictionary_review_form_update(review_with_added_items, update_items)


def _review_operation_value(operation: object, decision: object) -> str:
    value = str(operation or "").strip()
    if value in {"adopt", "reject", "delete"}:
        return value
    return "reject" if str(decision or "") == "ignore" else "adopt"


def _decision_from_review_operation(operation: str, form_item: dict) -> str:
    if operation == "reject":
        return "ignore"
    reading_suggested = str(form_item.get("reading_suggested") or "").strip()
    reading_final = str(form_item.get("reading_final") or "").strip()
    return "accept" if reading_final == reading_suggested else "edit"


def _review_change_mark(item: dict) -> str:
    reading_suggested = str(item.get("reading_suggested") or "").strip()
    reading_final = str(item.get("reading_final") or "").strip()
    return "＊" if reading_suggested != reading_final else ""


def _next_review_form_index(form_items: list[dict]) -> int:
    indexes = [
        _normalize_review_index(item.get("index"))
        for item in form_items
        if isinstance(item, dict)
    ]
    int_indexes = [index for index in indexes if isinstance(index, int)]
    return (max(int_indexes) if int_indexes else 0) + 1


def _append_unsaved_dictionary_review_item(
    original_review: dict,
    form_items: list[dict],
    *,
    word: object,
    reading: object,
    notes: object,
) -> dict:
    display_review = copy.deepcopy(original_review)
    display_review["items"] = _dictionary_review_items_for_display(
        original_review.get("items", []),
        form_items=form_items,
    )
    updated_review = append_manual_dictionary_review_item(
        display_review,
        word=word,
        reading=reading,
        notes=notes,
    )
    updated_review["items"][-1]["index"] = _next_review_form_index(form_items)
    return updated_review


def _normalize_review_index(value: object) -> object:
    if isinstance(value, int):
        return value
    try:
        return int(str(value))
    except (TypeError, ValueError):
        return value


def _direct_dictionary_finalize_pending_errors(review_data: dict) -> list[dict]:
    items = review_data.get("items")
    if not isinstance(items, list):
        return [
            {
                "code": "invalid_original_items",
                "message": "original review items must be array",
            }
        ]

    errors: list[dict] = []
    for position, item in enumerate(items, start=1):
        if not isinstance(item, dict):
            errors.append(
                {
                    "code": "invalid_original_item",
                    "message": "original review item must be object",
                    "position": position,
                }
            )
            continue
        if str(item.get("decision") or "") == "pending":
            errors.append(
                {
                    "code": "pending_reading_final_required",
                    "message": "pending item requires reading_final before direct finalize",
                    "position": position,
                    "word": str(item.get("word") or ""),
                }
            )
    return errors


def detect_existing_edit_artifacts(work_dir: Path) -> dict:
    existing_files: list[str] = []
    for filename in (
        workspace_paths.PROCESSED_DRAFT_FILENAME,
        workspace_paths.PROCESSED_FINAL_FILENAME,
    ):
        if (work_dir / filename).exists():
            existing_files.append(filename)
    return {
        "has_draft": workspace_paths.PROCESSED_DRAFT_FILENAME in existing_files,
        "has_final": workspace_paths.PROCESSED_FINAL_FILENAME in existing_files,
        "existing_files": existing_files,
        "should_block": bool(existing_files),
    }


def _can_create_empty_dictionary_processed(work_dir: Path, meta: dict) -> bool:
    if str(meta.get("status") or "") != "pre_processed":
        return False
    if (work_dir / workspace_paths.DICTIONARY_REVIEW_FILENAME).exists():
        return False
    if (work_dir / workspace_paths.PROCESSED_JSON_FILENAME).exists():
        return False
    if (work_dir / workspace_paths.WORK_DICTIONARY_FILENAME).exists():
        return False
    if (work_dir / workspace_paths.DICTIONARY_FINALIZE_REPORT_FILENAME).exists():
        return False
    return True


def _dictionary_file_import_block_reason(
    *,
    has_dictionary_review: bool,
    has_draft: bool,
    has_final: bool,
) -> str:
    if has_draft or has_final:
        return "読みと息継ぎの編集結果があります。辞書を変更する場合は、編集済みデータへの反映方法を別途選ぶ必要があります。"
    if has_dictionary_review:
        return ""
    return ""


def _import_type_label(work_dir: Path, meta: dict) -> str:
    import_type = str(meta.get("import_type") or "").strip()
    source_format = str(meta.get("source_format") or "").strip()
    if import_type == "text_import" or source_format == "plain_text":
        return "テキスト原稿"
    if import_type == "legacy_pef_import" or source_format == "pef_legacy":
        return "旧PEF原稿"
    detected_source_format = _work_pre_processed_source_format(work_dir)
    if detected_source_format == "plain_text":
        return "テキスト原稿"
    if detected_source_format == "pef_legacy":
        return "旧PEF原稿"
    return "不明"


def _empty_dictionary_conflict(work_dir: Path, meta: dict) -> dict | None:
    if (work_dir / workspace_paths.DICTIONARY_REVIEW_FILENAME).exists():
        return _dictionary_review_result(
            "failed",
            "辞書なしで編集用データを作成できませんでした。",
            failed_stage="preflight",
            error_type="dictionary_review_exists",
            error_message="01_dictionary_review.json already exists",
            affected_file=workspace_paths.DICTIONARY_REVIEW_FILENAME,
            committed=False,
            meta_status_changed=False,
        )
    if (work_dir / workspace_paths.PROCESSED_JSON_FILENAME).exists():
        return _dictionary_review_result(
            "failed",
            "辞書なしで編集用データを作成できませんでした。",
            failed_stage="preflight",
            error_type="processed_exists",
            error_message="02_processed.json already exists",
            affected_file=workspace_paths.PROCESSED_JSON_FILENAME,
            committed=False,
            meta_status_changed=False,
        )
    existing_step6_files = [
        filename
        for filename in (
            workspace_paths.WORK_DICTIONARY_FILENAME,
            workspace_paths.DICTIONARY_FINALIZE_REPORT_FILENAME,
        )
        if (work_dir / filename).exists()
    ]
    if existing_step6_files:
        return _dictionary_review_result(
            "failed",
            "辞書なしで編集用データを作成できませんでした。",
            failed_stage="preflight",
            error_type="existing_step6_artifacts",
            error_message="workspace already contains Step6 artifacts without review/processed context",
            affected_file=", ".join(existing_step6_files),
            committed=False,
            meta_status_changed=False,
        )
    if str(meta.get("status") or "") not in {"pre_processed", ""}:
        return _dictionary_review_result(
            "failed",
            "辞書なしで編集用データを作成できませんでした。",
            failed_stage="preflight",
            error_type="invalid_status",
            error_message=f"meta.status is not pre_processed: {meta.get('status') or 'unknown'}",
            affected_file=workspace_paths.WORK_META_FILENAME,
            committed=False,
            meta_status_changed=False,
        )
    return None


def _manual_dictionary_review_conflict(work_dir: Path) -> dict | None:
    if not (work_dir / workspace_paths.PRE_PROCESSED_JSON_FILENAME).exists():
        return _dictionary_review_result(
            "failed",
            "手動辞書を作成できませんでした。",
            failed_stage="preflight",
            error_type="missing_pre_processed",
            error_message="00_pre_processed.json does not exist",
            affected_file=workspace_paths.PRE_PROCESSED_JSON_FILENAME,
            committed=False,
            meta_status_changed=False,
        )
    if (work_dir / workspace_paths.DICTIONARY_REVIEW_FILENAME).exists():
        return _dictionary_review_result(
            "failed",
            "手動辞書を作成できませんでした。",
            failed_stage="preflight",
            error_type="dictionary_review_exists",
            error_message="01_dictionary_review.json already exists",
            affected_file=workspace_paths.DICTIONARY_REVIEW_FILENAME,
            committed=False,
            meta_status_changed=False,
        )
    return None


def _ai_dictionary_review_conflict(work_dir: Path) -> dict | None:
    if not (work_dir / workspace_paths.PRE_PROCESSED_JSON_FILENAME).exists():
        return _dictionary_review_result(
            "failed",
            "AI辞書候補の準備に失敗しました。",
            failed_stage="preflight",
            error_type="missing_pre_processed",
            error_message="00_pre_processed.json does not exist",
            affected_file=workspace_paths.PRE_PROCESSED_JSON_FILENAME,
            committed=False,
            meta_status_changed=False,
        )
    if not (work_dir / workspace_paths.SOURCE_ORIGINAL_FILENAME).exists():
        return _dictionary_review_result(
            "failed",
            "AI辞書候補の準備に失敗しました。",
            failed_stage="preflight",
            error_type="missing_source_original",
            error_message="source_original.txt does not exist",
            affected_file=workspace_paths.SOURCE_ORIGINAL_FILENAME,
            committed=False,
            meta_status_changed=False,
        )
    return None


def _run_review_to_processed_pipeline(
    work_dir: Path,
    *,
    meta_path: Path,
    original_meta: dict,
    original_status: str,
    success_message: str,
    form_items: list[dict] | None = None,
) -> dict:
    review_path = workspace_paths.dictionary_review_path(work_dir)
    finalize_report = run_finalize_dictionary(
        source_path=review_path,
        work_dir=work_dir,
        report_path=workspace_paths.dictionary_finalize_report_path(work_dir),
    )
    if finalize_report.get("status") != "success":
        _restore_meta_status(meta_path, original_meta, original_status)
        return _dictionary_review_result(
            "failed",
            (
                "辞書なしで編集用データを作成できませんでした。"
                if not form_items
                else "作品辞書の作成に失敗しました。"
            ),
            failed_stage="step6b_finalize_dictionary",
            error_type="step6b_failed",
            error_message=_report_error_summary(finalize_report),
            affected_file=workspace_paths.WORK_DICTIONARY_FILENAME,
            committed=False,
            meta_status_changed=False,
            report_path=workspace_paths.DICTIONARY_FINALIZE_REPORT_FILENAME,
            form_items=form_items,
        )

    processed_report = run_processed_workspace_full(work_dir)
    if processed_report.get("status") != "success":
        _restore_meta_status(meta_path, original_meta, original_status)
        return _dictionary_review_result(
            "failed",
            (
                "辞書なしで編集用データを作成できませんでした。"
                if not form_items
                else "編集用データの作成に失敗しました。"
            ),
            failed_stage="step6c_build_processed",
            error_type="step6c_failed",
            error_message=_report_error_summary(processed_report),
            affected_file=workspace_paths.PROCESSED_JSON_FILENAME,
            committed=False,
            meta_status_changed=False,
            report_path=workspace_paths.PROCESSED_REPORT_FILENAME,
            form_items=form_items,
        )

    return {
        "status": "success",
        "message": success_message,
        "committed": True,
        "meta_status_changed": True,
    }


def _read_required_dictionary_review(path: Path) -> dict | None:
    try:
        review = read_json(path)
    except Exception:
        return None
    return review if isinstance(review, dict) else None


def _restore_meta_status(
    meta_path: Path, original_meta: dict, original_status: str
) -> None:
    if not meta_path.exists() or not original_meta:
        return
    restored = copy.deepcopy(original_meta)
    restored["status"] = original_status
    write_json(meta_path, restored)


def _backup_existing_files(
    work_dir: Path,
    prefix: str,
    filenames: tuple[str, ...],
) -> dict:
    existing_files = [filename for filename in filenames if (work_dir / filename).exists()]
    if not existing_files:
        return {"backup_dir": None, "filenames": []}

    backup_dir = _unique_backup_dir(work_dir, prefix)
    backup_dir.mkdir(parents=True, exist_ok=False)
    moved_files: list[str] = []
    try:
        for filename in existing_files:
            shutil.move(str(work_dir / filename), str(backup_dir / filename))
            moved_files.append(filename)
    except Exception:
        for filename in reversed(moved_files):
            backup_path = backup_dir / filename
            if backup_path.exists():
                shutil.move(str(backup_path), str(work_dir / filename))
        raise
    return {"backup_dir": backup_dir, "filenames": moved_files}


def _restore_backup_files(
    work_dir: Path,
    backup: dict | None,
    filenames: tuple[str, ...],
) -> None:
    if backup is None:
        return

    for filename in filenames:
        path = work_dir / filename
        if path.exists():
            path.unlink()

    backup_dir = backup.get("backup_dir")
    backup_files = backup.get("filenames", [])
    if backup_dir is None:
        return

    backup_dir = Path(backup_dir)
    for filename in backup_files:
        backup_path = backup_dir / filename
        if backup_path.exists():
            shutil.move(str(backup_path), str(work_dir / filename))
    try:
        backup_dir.rmdir()
    except OSError:
        pass


def _unique_backup_dir(work_dir: Path, prefix: str) -> Path:
    backups_dir = work_dir / "backups"
    timestamp = datetime.now(workspace_paths.JST).strftime("%Y%m%d-%H%M%S")
    candidate = backups_dir / f"{prefix}_{timestamp}"
    if not candidate.exists():
        return candidate
    for suffix in range(1, 100):
        candidate = backups_dir / f"{prefix}_{timestamp}_{suffix}"
        if not candidate.exists():
            return candidate
    raise RuntimeError("backup directory could not be created")


def _report_error_summary(report: object) -> str:
    if not isinstance(report, dict):
        return "unknown report error"
    errors = report.get("errors")
    if not isinstance(errors, list) or not errors:
        return "unknown report error"
    messages: list[str] = []
    for error in errors:
        if isinstance(error, dict):
            message = str(error.get("message") or error.get("code") or "").strip()
            if message:
                messages.append(message)
        else:
            message = str(error).strip()
            if message:
                messages.append(message)
    return "; ".join(messages) if messages else "unknown report error"


def _dictionary_review_validation_summary(errors: list[dict]) -> str:
    messages: list[str] = []
    for error in errors:
        if not isinstance(error, dict):
            continue
        message = str(error.get("message") or error.get("code") or "").strip()
        if message:
            messages.append(message)
    return "; ".join(messages) if messages else "validation failed"


def _dictionary_review_result(
    status: str,
    message: str,
    *,
    failed_stage: str = "",
    error_type: str = "",
    error_message: str = "",
    affected_file: str = "",
    committed: bool = False,
    meta_status_changed: bool = False,
    report_path: str = "",
    errors: list[dict] | None = None,
    form_items: list[dict] | None = None,
    existing_edit_artifacts: list[str] | None = None,
    future_recovery_hint: str = "",
) -> dict:
    return {
        "status": status,
        "message": message,
        "failed_stage": failed_stage,
        "error_type": error_type,
        "error_message": error_message,
        "affected_file": affected_file,
        "committed": committed,
        "meta_status_changed": meta_status_changed,
        "report_path": report_path,
        "errors": list(errors or []),
        "form_items": form_items or [],
        "existing_edit_artifacts": list(existing_edit_artifacts or []),
        "future_recovery_hint": future_recovery_hint,
    }


def _read_optional_dict(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        data = read_json(path)
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def _validate_legacy_uploads(json_upload, txt_upload) -> list[str]:
    if json_upload is None or not getattr(json_upload, "filename", ""):
        return ["json ファイルを選択してください"]
    if txt_upload is None or not getattr(txt_upload, "filename", ""):
        return ["同名の txt ファイルが必要です"]

    json_name = Path(str(json_upload.filename).strip()).name
    txt_name = Path(str(txt_upload.filename).strip()).name
    json_base_stem = _legacy_json_base_stem(Path(json_name).stem)
    if Path(json_name).suffix.lower() != ".json":
        return ["json ファイルを選択してください"]
    if Path(txt_name).suffix.lower() != ".txt":
        return ["txt ファイルを選択してください"]
    if json_base_stem != Path(txt_name).stem:
        return ["json の元作品名と txt のファイル名が一致していません"]
    if _upload_is_empty(json_upload):
        return ["json ファイルが空です"]
    if _upload_is_empty(txt_upload):
        return ["txt ファイルが空です"]
    return []


def _validate_text_upload(title: str, txt_upload) -> tuple[list[str], str, str]:
    if txt_upload is None or not getattr(txt_upload, "filename", ""):
        return ["txt ファイルを選択してください"], "", ""

    txt_name = Path(str(txt_upload.filename).strip()).name
    if Path(txt_name).suffix.lower() != ".txt":
        return ["txt ファイルを選択してください"], "", ""
    if _upload_is_empty(txt_upload):
        return ["txt ファイルが空です"], "", ""

    requested_title = str(title or "").strip()
    txt_stem = Path(txt_name).stem.strip()
    resolved_title = requested_title or txt_stem
    if not resolved_title:
        return ["作品名を入力してください"], "", ""

    input_stem = workspace_paths.sanitize_work_title(resolved_title)
    if not input_stem:
        return ["作品名を入力してください"], "", ""
    return [], resolved_title, input_stem


def _validate_legacy_dictionary_upload(dictionary_upload) -> list[str]:
    if dictionary_upload is None or not getattr(dictionary_upload, "filename", ""):
        return ["辞書ファイルを選択してください。"]
    filename = Path(str(dictionary_upload.filename).strip()).name
    if Path(filename).suffix.lower() != ".json":
        return ["辞書ファイルを選択してください。"]
    if _upload_is_empty(dictionary_upload):
        return ["辞書ファイルが空です。"]
    return []


def _legacy_json_base_stem(json_stem: str) -> str:
    marker = "_[v"
    if marker not in json_stem:
        return json_stem
    return json_stem.split(marker, 1)[0]


def _upload_is_empty(upload) -> bool:
    stream = getattr(upload, "stream", None)
    if stream is None:
        return True
    position = stream.tell()
    stream.seek(0, os.SEEK_END)
    size = stream.tell()
    stream.seek(position)
    return size <= 0


def _upload_bytes(upload) -> bytes:
    stream = upload.stream
    stream.seek(0)
    data = stream.read()
    stream.seek(0)
    return data


def _legacy_dictionary_import_error_message(
    error: LegacyDictionaryImportValidationError,
) -> str:
    if error.code == "invalid_top_level":
        return "正しい辞書フォーマットのファイルを指定してください"
    if error.item_index is None:
        return error.message
    return f"{error.message} item {error.item_index}"


def _import_error_message(error: Exception) -> str:
    if isinstance(error, UnicodeDecodeError):
        return "json を読み込めませんでした"
    if isinstance(error, ValueError):
        message = str(error)
        if "cannot parse JSON object in source text" in message:
            return "json を読み込めませんでした"
        if (
            "unknown source format" in message
            or "cannot parse multiple JSON objects" in message
            or "no segments parsed" in message
        ):
            return "00_pre_processed.json の生成に失敗しました"
        return "json を読み込めませんでした"
    return "00_pre_processed.json の生成に失敗しました"


def _text_import_error_message(error: Exception) -> str:
    if isinstance(error, UnicodeDecodeError):
        return "テキスト原稿を取り込めませんでした。txt を UTF-8 で保存してください。"
    return "テキスト原稿を取り込めませんでした。"


def _unique_generated_work_id(workspace_root: Path, title: str) -> str:
    base_now = datetime.now(workspace_paths.JST)
    for offset in range(0, 60):
        candidate = workspace_paths.generate_work_id(
            title, now=base_now + timedelta(seconds=offset)
        )
        if not (workspace_root / candidate).exists():
            return candidate
    raise RuntimeError("work_id を生成できませんでした")


def _ensure_workspace_runtime_dirs(work_dir: Path) -> None:
    work_dir.mkdir(parents=True, exist_ok=True)
    (work_dir / "audio").mkdir(exist_ok=True)
    (work_dir / "epub").mkdir(exist_ok=True)


def _ensure_workspace_root_layout(workspace_root: Path) -> None:
    (workspace_root / "dictionaries" / "system").mkdir(parents=True, exist_ok=True)
    (workspace_root / "dictionaries" / "user").mkdir(parents=True, exist_ok=True)
    _ensure_default_workspace_dictionaries(workspace_root)


def _ensure_default_workspace_dictionaries(workspace_root: Path) -> None:
    system_dir = workspace_root / "dictionaries" / "system"
    user_dir = workspace_root / "dictionaries" / "user"
    default_files = {
        system_dir / "★システム固定辞書.json": [],
        system_dir / "★標準英単語辞書.json": [],
        system_dir / "★汎用息継ぎ辞書.json": {},
        system_dir
        / "★読み上げ記号ルール.json": {
            "schema_version": "symbol-reading-rules-1",
            "drop": ["「", "」"],
            "pause_s": ["、", "・"],
            "pause_m": ["……"],
            "keep": ["。"],
        },
        user_dir / "★ユーザ辞書.json": [],
    }
    for path, data in default_files.items():
        if not path.exists():
            write_json(path, data)


def _keep_failed_imports_enabled() -> bool:
    return os.environ.get("PEF2_KEEP_IMPORT_FAILED") == "1"


def _write_import_failure_report(work_dir: Path, report: dict) -> None:
    try:
        write_json(work_dir / "import_failed_report.json", report)
    except Exception:
        return


def _is_hidden_workspace_dir(path: Path) -> bool:
    return path.name.startswith("_") or path.name.startswith("_import_tmp_") or path.name.startswith(
        "_import_failed_"
    )


def _is_reserved_work_id(work_id: str) -> bool:
    return (
        work_id in RESERVED_WORK_DIR_NAMES
        or work_id.startswith("_")
        or work_id.startswith("_import_tmp_")
        or work_id.startswith("_import_failed_")
    )


def _unique_trash_destination(trash_root: Path, work_id: str, timestamp: str) -> Path:
    base_name = f"{work_id}_{timestamp}"
    candidate = trash_root / base_name
    if not candidate.exists():
        return candidate
    for index in range(1, 1000):
        candidate = trash_root / f"{base_name}_{index}"
        if not candidate.exists():
            return candidate
    raise RuntimeError("退避先を作成できませんでした。")


def _work_title(work_id: str, meta: dict) -> str:
    for key in ("title", "original_stem", "input_stem"):
        value = meta.get(key)
        if value:
            return str(value)
    source_filename = str(meta.get("source_original_filename") or "")
    if source_filename:
        return Path(source_filename).stem
    return work_id


def _detail_input_stem(work_dir: Path) -> str:
    meta = _read_optional_dict(work_dir / workspace_paths.WORK_META_FILENAME)
    input_stem = str(meta.get("input_stem") or "").strip()
    if input_stem:
        return input_stem
    return _work_title(work_dir.name, meta)


def _work_pre_processed_source_format(work_dir: Path) -> str:
    pre_processed_path = work_dir / workspace_paths.PRE_PROCESSED_JSON_FILENAME
    if not pre_processed_path.exists():
        return ""
    try:
        pre_processed = read_json(pre_processed_path)
    except Exception:
        return ""
    if not isinstance(pre_processed, dict):
        return ""
    source = pre_processed.get("source")
    if not isinstance(source, dict):
        return ""
    return str(source.get("source_format") or "").strip()


def _dictionary_review_debug_info(work_dir: Path) -> dict:
    review_path = work_dir / workspace_paths.DICTIONARY_REVIEW_FILENAME
    if not review_path.exists():
        return {"items": 0, "source": "none"}
    review_data = _read_optional_dict(review_path)
    items = review_data.get("items")
    item_count = len(items) if isinstance(items, list) else 0
    source = str(review_data.get("source") or "none")
    return {"items": item_count, "source": source}


def _gemini_review_debug_info(work_dir: Path) -> dict:
    raw_path = work_dir / STEP5_DIRNAME / GEMINI_REVIEW_RAW_FILENAME
    if not raw_path.exists():
        return {"exists": False}
    raw = _read_optional_dict(raw_path)
    if not raw:
        return {
            "exists": True,
            "raw_path": str(raw_path),
            "readable": False,
        }
    failed_chunks = []
    chunks = raw.get("chunks")
    if isinstance(chunks, list):
        for chunk in chunks:
            if not isinstance(chunk, dict):
                continue
            status = str(chunk.get("status") or "")
            if status not in {"failed", "timeout"}:
                continue
            failed_chunks.append(
                {
                    "chunk_index": chunk.get("chunk_index"),
                    "candidate_count": chunk.get("candidate_count"),
                    "status": status,
                    "attempts": chunk.get("attempts"),
                    "error_type": str(chunk.get("error_type") or ""),
                    "message": str(chunk.get("message") or ""),
                }
            )
    raw_error_responses = raw.get("raw_error_responses")
    return {
        "exists": True,
        "raw_path": str(raw_path),
        "readable": True,
        "api_called": bool(raw.get("gemini_api_called")),
        "skipped": bool(raw.get("skipped")),
        "model": str(raw.get("model") or ""),
        "candidate_count": raw.get("candidate_count"),
        "chunk_size": raw.get("chunk_size"),
        "failed_count": raw.get("failed_count"),
        "timeout_count": raw.get("timeout_count"),
        "raw_error_response_count": len(raw_error_responses)
        if isinstance(raw_error_responses, list)
        else 0,
        "failed_chunks": failed_chunks,
    }


def _update_draft_meta(work_dir: Path, updated_at: str) -> None:
    meta_path = work_dir / workspace_paths.WORK_META_FILENAME
    meta = _read_optional_dict(meta_path)
    meta["status"] = "draft_saved"
    meta["updated_at"] = updated_at
    write_json(meta_path, meta)


def _update_final_meta(work_dir: Path, updated_at: str) -> None:
    meta_path = work_dir / workspace_paths.WORK_META_FILENAME
    meta = _read_optional_dict(meta_path)
    meta["status"] = "finalized"
    meta["updated_at"] = updated_at
    write_json(meta_path, meta)


def _cleanup_draft_after_final(work_dir: Path) -> None:
    for filename in (
        workspace_paths.PROCESSED_DRAFT_FILENAME,
        workspace_paths.EDITING_SESSION_FILENAME,
    ):
        path = work_dir / filename
        if path.exists():
            path.unlink()


def _write_draft_editing_session(work_dir: Path, updated_at: str) -> None:
    write_json(
        work_dir / workspace_paths.EDITING_SESSION_FILENAME,
        {
            "work_id": work_dir.name,
            "status": "draft_saved",
            "updated_at": updated_at,
            "source": "pef_studio",
        },
    )


def _write_reedit_editing_session(work_dir: Path, updated_at: str) -> None:
    write_json(
        work_dir / workspace_paths.EDITING_SESSION_FILENAME,
        {
            "work_id": work_dir.name,
            "status": "draft_saved",
            "updated_at": updated_at,
            "source": "reedit_from_final",
        },
    )


def _jst_now_iso() -> str:
    return datetime.now(workspace_paths.JST).replace(microsecond=0).isoformat()
