from __future__ import annotations

import json
import os
import re
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any


PROGRESS_DIRNAME = ".progress"
TASK_ID_PATTERN = re.compile(r"^(?P<operation>[a-z][a-z0-9_]*)_[0-9a-f]{32}$")
JST = timezone(timedelta(hours=9))


def create_task_id(operation: str) -> str:
    operation = _normalize_operation(operation)
    return f"{operation}_{uuid.uuid4().hex}"


def is_valid_task_id(task_id: object, *, operation: str | None = None) -> bool:
    if not isinstance(task_id, str):
        return False
    match = TASK_ID_PATTERN.fullmatch(task_id)
    if match is None:
        return False
    if operation is None:
        return True
    return match.group("operation") == _normalize_operation(operation)


def progress_path(work_dir: Path, task_id: str) -> Path | None:
    if not is_valid_task_id(task_id):
        return None
    return Path(work_dir) / PROGRESS_DIRNAME / f"{task_id}.json"


def new_progress(
    work_dir: Path,
    *,
    task_id: str,
    operation: str,
    status: str,
    phase: str,
    message: str,
    lock_started_at: str = "",
    result: dict | None = None,
    error: dict | None = None,
    next_action: str = "",
    next_action_status: str = "",
    now: datetime | None = None,
) -> dict[str, Any]:
    timestamp = _timestamp(now)
    return {
        "task_id": task_id,
        "work_id": Path(work_dir).name,
        "operation": _normalize_operation(operation),
        "status": status,
        "phase": phase,
        "message": message,
        "percent": None,
        "started_at": timestamp,
        "updated_at": timestamp,
        "finished_at": "",
        "lock_started_at": lock_started_at,
        "result": result,
        "error": error,
        "cancellable": False,
        "cancel_requested": False,
        "next_action": next_action,
        "next_action_status": next_action_status,
    }


def read_progress(work_dir: Path, task_id: str) -> dict | None:
    path = progress_path(work_dir, task_id)
    if path is None or not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict):
        return None
    if data.get("task_id") != task_id or data.get("work_id") != Path(work_dir).name:
        return None
    return data


def write_progress(work_dir: Path, progress: dict) -> dict:
    task_id = str(progress.get("task_id") or "")
    path = progress_path(work_dir, task_id)
    if path is None:
        raise ValueError("invalid task_id")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.tmp")
    tmp_path.write_text(
        json.dumps(progress, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(tmp_path, path)
    return progress


def update_progress(work_dir: Path, task_id: str, **updates: Any) -> dict | None:
    progress = read_progress(work_dir, task_id)
    if progress is None:
        return None
    progress.update(updates)
    progress["updated_at"] = _timestamp()
    if progress.get("status") in {"completed", "failed", "abandoned"} and not progress.get("finished_at"):
        progress["finished_at"] = progress["updated_at"]
    return write_progress(work_dir, progress)


def _normalize_operation(operation: str) -> str:
    value = str(operation or "").strip()
    if not re.fullmatch(r"[a-z][a-z0-9_]*", value):
        raise ValueError("invalid operation")
    return value


def _timestamp(now: datetime | None = None) -> str:
    value = now or datetime.now(JST)
    if value.tzinfo is None:
        value = value.replace(tzinfo=JST)
    return value.astimezone(JST).isoformat(timespec="seconds")
