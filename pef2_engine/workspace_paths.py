from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone
from pathlib import Path

from pef2_engine.io_utils import read_json, write_json


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_DIR = PROJECT_ROOT / "workspace"
WORKSPACE_DICTIONARIES_DIR = WORKSPACE_DIR / "dictionaries"
WORKSPACE_SYSTEM_DICTIONARIES_DIR = WORKSPACE_DICTIONARIES_DIR / "system"
WORKSPACE_USER_DICTIONARIES_DIR = WORKSPACE_DICTIONARIES_DIR / "user"

WORK_META_FILENAME = "meta.json"
SOURCE_ORIGINAL_FILENAME = "source_original.txt"
WORK_DICTIONARY_DRAFT_FILENAME = "work_dictionary_draft.json"
PRE_PROCESSED_JSON_FILENAME = "00_pre_processed.json"
DICTIONARY_REVIEW_FILENAME = "01_dictionary_review.json"
DICTIONARY_FINALIZE_REPORT_FILENAME = "01_dictionary_finalize_report.json"
WORK_DICTIONARY_FILENAME = "work_dictionary.json"
LEGACY_DICTIONARY_FILENAME = "legacy_dictionary.json"
PROCESSED_JSON_FILENAME = "02_processed.json"
PROCESSED_REPORT_FILENAME = "02_processed_report.json"
PROCESSED_DRAFT_FILENAME = "03_processed_draft.json"
EDITING_SESSION_FILENAME = "03_editing_session.json"
PROCESSED_FINAL_FILENAME = "04_processed_final.json"

WORK_META_SCHEMA_VERSION = "work_meta-1"
JST = timezone(timedelta(hours=9))
FORBIDDEN_TITLE_CHARS = re.compile(r'[\/\\:\*\?"<>\|]')


def workspace_dir(root: Path | None = None) -> Path:
    return (root or PROJECT_ROOT) / "workspace"


def workspace_dictionaries_dir(root: Path | None = None) -> Path:
    return workspace_dir(root) / "dictionaries"


def workspace_system_dictionaries_dir(root: Path | None = None) -> Path:
    return workspace_dictionaries_dir(root) / "system"


def workspace_user_dictionaries_dir(root: Path | None = None) -> Path:
    return workspace_dictionaries_dir(root) / "user"


def sanitize_work_title(title: str, max_length: int = 40) -> str:
    sanitized = FORBIDDEN_TITLE_CHARS.sub("_", str(title).strip())
    if len(sanitized) > max_length:
        sanitized = sanitized[:max_length]
    return sanitized or "untitled"


def generate_work_id(title: str, now: datetime | None = None) -> str:
    timestamp = _jst_datetime(now).strftime("%Y%m%d-%H%M%S")
    return f"{sanitize_work_title(title)}-{timestamp}"


def create_work_meta(
    *,
    work_id: str,
    title: str,
    source_original_filename: str,
    input_stem: str | None = None,
    status: str = "created",
    now: datetime | None = None,
) -> dict:
    timestamp = _jst_datetime(now).isoformat()
    resolved_input_stem = input_stem
    if resolved_input_stem is None:
        resolved_input_stem = Path(source_original_filename).stem or sanitize_work_title(title)
    return {
        "schema_version": WORK_META_SCHEMA_VERSION,
        "work_id": work_id,
        "title": title,
        "input_stem": resolved_input_stem,
        "source_original_filename": source_original_filename,
        "status": status,
        "created_at": timestamp,
        "updated_at": timestamp,
    }


def write_work_meta(work_dir: Path, meta: dict) -> None:
    write_json(work_dir / WORK_META_FILENAME, meta)


def update_work_meta_status(
    work_dir: Path,
    status: str,
    now: datetime | None = None,
) -> dict:
    meta_path = work_dir / WORK_META_FILENAME
    meta = read_json(meta_path)
    if not isinstance(meta, dict):
        raise ValueError("meta.json top-level must be object")
    meta["status"] = status
    meta["updated_at"] = _jst_datetime(now).isoformat()
    write_json(meta_path, meta)
    return meta


def create_work_workspace(
    title: str,
    source_original_filename: str,
    source_text: str | None = None,
    now: datetime | None = None,
    root: Path | None = None,
) -> Path:
    work_id = generate_work_id(title, now)
    ensure_workspace_dictionary_dirs(root)
    work_dir = workspace_dir(root) / work_id
    work_dir.mkdir(parents=True, exist_ok=False)
    (work_dir / "audio").mkdir()
    (work_dir / "epub").mkdir()
    if source_text is not None:
        (work_dir / SOURCE_ORIGINAL_FILENAME).write_text(source_text, encoding="utf-8")
    meta = create_work_meta(
        work_id=work_id,
        title=title,
        source_original_filename=source_original_filename,
        now=now,
    )
    write_work_meta(work_dir, meta)
    return work_dir


def ensure_workspace_dictionary_dirs(root: Path | None = None) -> None:
    workspace_system_dictionaries_dir(root).mkdir(parents=True, exist_ok=True)
    workspace_user_dictionaries_dir(root).mkdir(parents=True, exist_ok=True)


def work_dictionary_draft_path(work_dir: Path) -> Path:
    return work_dir / WORK_DICTIONARY_DRAFT_FILENAME


def pre_processed_path(work_dir: Path) -> Path:
    return work_dir / PRE_PROCESSED_JSON_FILENAME


def dictionary_review_path(work_dir: Path) -> Path:
    return work_dir / DICTIONARY_REVIEW_FILENAME


def work_dictionary_path(work_dir: Path) -> Path:
    return work_dir / WORK_DICTIONARY_FILENAME


def legacy_dictionary_path(work_dir: Path) -> Path:
    return work_dir / LEGACY_DICTIONARY_FILENAME


def dictionary_finalize_report_path(work_dir: Path) -> Path:
    return work_dir / DICTIONARY_FINALIZE_REPORT_FILENAME


def processed_path(work_dir: Path) -> Path:
    return work_dir / PROCESSED_JSON_FILENAME


def processed_report_path(work_dir: Path) -> Path:
    return work_dir / PROCESSED_REPORT_FILENAME


def processed_draft_path(work_dir: Path) -> Path:
    return work_dir / PROCESSED_DRAFT_FILENAME


def editing_session_path(work_dir: Path) -> Path:
    return work_dir / EDITING_SESSION_FILENAME


def processed_final_path(work_dir: Path) -> Path:
    return work_dir / PROCESSED_FINAL_FILENAME


def display_path(path: Path, base: Path) -> str:
    try:
        return path.relative_to(base).as_posix()
    except ValueError:
        return path.as_posix()


def _jst_datetime(now: datetime | None = None) -> datetime:
    value = now or datetime.now(JST)
    if value.tzinfo is None:
        return value.replace(tzinfo=JST)
    return value.astimezone(JST)
