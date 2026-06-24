from __future__ import annotations

import json
import os
import shutil
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any


LOCK_FILENAME = ".generation_lock"
STALE_AFTER = timedelta(hours=6)
JST = timezone(timedelta(hours=9))

STALE_LOCK_CLEARED_MESSAGE = "前回の生成中ロックが残っていたため解除しました。もう一度操作してください。"
ACTIVE_LOCK_MESSAGES = {
    "tts": "この作品は現在、音声を生成中です。生成が終わってからもう一度操作してください。",
    "epub": "この作品は現在、EPUBを生成中です。生成が終わってからもう一度操作してください。",
    "ai_dictionary": "この作品は現在、AI辞書候補を作成中です。作成が終わってからもう一度操作してください。",
}


def generation_lock_path(work_dir: Path) -> Path:
    return Path(work_dir) / LOCK_FILENAME


def read_generation_lock(work_dir: Path) -> dict | None:
    path = generation_lock_path(work_dir)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"invalid": True}
    return data if isinstance(data, dict) else {"invalid": True}


def active_generation_lock_message(lock_data: dict | None = None) -> str:
    if not isinstance(lock_data, dict):
        return "この作品は現在、生成処理中です。生成が終わってからもう一度操作してください。"
    operation = str(lock_data.get("operation") or "")
    return ACTIVE_LOCK_MESSAGES.get(
        operation,
        "この作品は現在、生成処理中です。生成が終わってからもう一度操作してください。",
    )


def is_generation_lock_stale(lock_data: dict | None, now: datetime | None = None) -> bool:
    if not isinstance(lock_data, dict) or lock_data.get("invalid"):
        return True
    started_at = _parse_datetime(lock_data.get("started_at"))
    if started_at is None:
        return True
    current = now or datetime.now(JST)
    if current.tzinfo is None:
        current = current.replace(tzinfo=JST)
    return current - started_at >= STALE_AFTER


def acquire_generation_lock(
    work_dir: Path,
    operation: str,
    *,
    task_id: str = "",
    now: datetime | None = None,
) -> dict[str, Any]:
    work_dir = Path(work_dir)
    lock_path = generation_lock_path(work_dir)
    current = now or datetime.now(JST)
    if current.tzinfo is None:
        current = current.replace(tzinfo=JST)
    lock_data = {
        "work_id": work_dir.name,
        "operation": operation,
        "pid": os.getpid(),
        "started_at": current.isoformat(timespec="seconds"),
    }
    if task_id:
        lock_data["task_id"] = task_id

    try:
        fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError:
        existing = read_generation_lock(work_dir)
        if is_generation_lock_stale(existing, current):
            backup_path = clear_stale_generation_lock(work_dir, current, expected_lock=existing)
            return {
                "status": "stale_cleared",
                "ok": False,
                "message": STALE_LOCK_CLEARED_MESSAGE,
                "lock": existing,
                "backup_path": str(backup_path) if backup_path is not None else "",
            }
        return {
            "status": "locked",
            "ok": False,
            "message": active_generation_lock_message(existing),
            "lock": existing,
        }
    except OSError as error:
        return {
            "status": "failed",
            "ok": False,
            "message": active_generation_lock_message(),
            "error": f"{type(error).__name__}: {error}",
        }

    with os.fdopen(fd, "w", encoding="utf-8") as lock_file:
        json.dump(lock_data, lock_file, ensure_ascii=False, indent=2)
        lock_file.write("\n")
    return {
        "status": "acquired",
        "ok": True,
        "message": "",
        "lock": lock_data,
    }


def release_generation_lock(work_dir: Path, lock_data: dict | None = None) -> None:
    path = generation_lock_path(work_dir)
    if not path.exists():
        return
    if lock_data is not None:
        current = read_generation_lock(work_dir)
        if not _same_lock(current, lock_data):
            return
    try:
        path.unlink()
    except FileNotFoundError:
        return


def clear_stale_generation_lock(
    work_dir: Path,
    now: datetime | None = None,
    *,
    expected_lock: dict | None = None,
) -> Path | None:
    path = generation_lock_path(work_dir)
    if not path.exists():
        return None
    if expected_lock is not None:
        current_lock = read_generation_lock(work_dir)
        if current_lock != expected_lock:
            return None
    current = now or datetime.now(JST)
    if current.tzinfo is None:
        current = current.replace(tzinfo=JST)
    backup_dir = Path(work_dir) / "backups"
    backup_dir.mkdir(parents=True, exist_ok=True)
    backup_path = backup_dir / f"stale_generation_lock_{_timestamp(current)}.json"
    try:
        shutil.move(str(path), str(backup_path))
    except FileNotFoundError:
        return None
    return backup_path


def _parse_datetime(value: object) -> datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=JST)
    return parsed


def _same_lock(current: dict | None, expected: dict) -> bool:
    if not isinstance(current, dict):
        return False
    keys = ("work_id", "operation", "pid", "started_at")
    return all(current.get(key) == expected.get(key) for key in keys)


def _timestamp(value: datetime) -> str:
    return value.astimezone(JST).strftime("%Y%m%d-%H%M%S")
