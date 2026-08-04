from __future__ import annotations

import json
import re
import shutil
from datetime import datetime, timedelta, timezone
from pathlib import Path


JST = timezone(timedelta(hours=9))
ARTIFACT_MAX_AGE = timedelta(hours=24)
TERMINAL_PROGRESS_STATUSES = {"completed", "failed", "cancelled", "abandoned"}
OPERATION_PATTERN = re.compile(r"^[a-z][a-z0-9_]*$")
TIMESTAMPED_DIR_SUFFIX = re.compile(r"^\d{8}-\d{6}(?:_\d+)?$")
TIMESTAMPED_FILE_SUFFIX = re.compile(r"^\d{8}-\d{6}(?:_\d+)?\.json$")
BACKUP_FILE_TIMESTAMP = re.compile(r"^\d{8}-\d{6}$")


def prune_backup_directories(
    work_dir: Path,
    prefix: str,
    *,
    keep: int = 1,
) -> dict:
    backups_dir = Path(work_dir) / "backups"
    candidates = _timestamped_directories(backups_dir, prefix)
    return _delete_oldest(
        sorted(candidates, key=lambda path: _timestamped_directory_sort_key(path, prefix)),
        keep=keep,
    )


def prune_timestamped_backup_files(backups_dir: Path, *, keep: int = 1) -> dict:
    backups_dir = Path(backups_dir)
    result = _result()
    if not _safe_directory(backups_dir):
        return result
    try:
        root_resolved = backups_dir.resolve(strict=True)
        entries = list(backups_dir.iterdir())
    except OSError as error:
        result["errors"].append({"path": str(backups_dir), "reason": f"read_failed:{type(error).__name__}:{error}"})
        return result

    by_timestamp: dict[str, list[Path]] = {}
    for path in entries:
        if path.is_symlink() or not path.is_file() or "_" not in path.name:
            continue
        timestamp = path.name.split("_", 1)[0]
        if BACKUP_FILE_TIMESTAMP.fullmatch(timestamp) is None:
            continue
        try:
            if path.resolve(strict=True).parent != root_resolved:
                continue
        except OSError as error:
            result["errors"].append({"path": str(path), "reason": f"stat_failed:{type(error).__name__}:{error}"})
            continue
        by_timestamp.setdefault(timestamp, []).append(path)

    keep = max(0, int(keep))
    timestamps = sorted(by_timestamp)
    for timestamp in timestamps[: max(0, len(timestamps) - keep)]:
        for path in by_timestamp[timestamp]:
            try:
                path.unlink()
            except OSError as error:
                result["errors"].append({"path": str(path), "reason": f"delete_failed:{type(error).__name__}:{error}"})
            else:
                result["deleted"].append(str(path))
    return result


def prune_stale_lock_backups(work_dir: Path, *, keep: int = 2) -> dict:
    backups_dir = Path(work_dir) / "backups"
    result = _result()
    candidates: list[Path] = []
    if not _safe_directory(backups_dir):
        return result
    try:
        entries = list(backups_dir.iterdir())
    except OSError as error:
        result["errors"].append({"path": str(backups_dir), "reason": f"read_failed:{type(error).__name__}:{error}"})
        return result
    for path in entries:
        if path.is_symlink() or not path.is_file():
            continue
        if not path.name.startswith("stale_generation_lock_"):
            continue
        suffix = path.name.removeprefix("stale_generation_lock_")
        if TIMESTAMPED_FILE_SUFFIX.fullmatch(suffix):
            candidates.append(path)
    _merge_result(
        result,
        _delete_oldest(sorted(candidates, key=lambda path: path.name), keep=keep),
    )
    return result


def prune_terminal_progress(work_dir: Path, *, keep_per_operation: int = 1) -> dict:
    progress_dir = Path(work_dir) / ".progress"
    result = _result()
    if not _safe_directory(progress_dir):
        return result

    by_operation: dict[str, list[tuple[str, Path]]] = {}
    try:
        entries = list(progress_dir.iterdir())
    except OSError as error:
        result["errors"].append({"path": str(progress_dir), "reason": f"read_failed:{type(error).__name__}:{error}"})
        return result

    for path in entries:
        if path.is_symlink() or not path.is_file() or path.suffix.lower() != ".json":
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            result["skipped"].append({"path": str(path), "reason": f"unreadable:{type(error).__name__}"})
            continue
        if not isinstance(data, dict):
            result["skipped"].append({"path": str(path), "reason": "invalid_progress"})
            continue
        operation = str(data.get("operation") or "")
        status = str(data.get("status") or "")
        if not OPERATION_PATTERN.fullmatch(operation) or status not in TERMINAL_PROGRESS_STATUSES:
            result["skipped"].append({"path": str(path), "reason": "not_terminal_progress"})
            continue
        sort_key = str(data.get("finished_at") or data.get("updated_at") or data.get("started_at") or "")
        by_operation.setdefault(operation, []).append((sort_key, path))

    for candidates in by_operation.values():
        ordered = [path for _, path in sorted(candidates, key=lambda item: (item[0], item[1].name))]
        _merge_result(result, _delete_oldest(ordered, keep=keep_per_operation))
    return result


def cleanup_generation_artifacts(
    work_dir: Path,
    category: str,
    *,
    now: datetime | None = None,
) -> dict:
    work_dir = Path(work_dir)
    if category == "audio":
        roots = ((work_dir / "audio", ("_build_tmp_", "_commit_tmp_", "_build_failed_")),)
    elif category == "epub":
        roots = ((work_dir / "epub", ("_build_tmp_", "_build_failed_")),)
    else:
        raise ValueError("unsupported cleanup category")
    return _cleanup_aged_directories(roots, now=now)


def cleanup_preview_failures(preview_dir: Path, *, now: datetime | None = None) -> dict:
    return _cleanup_aged_directories(
        ((Path(preview_dir), ("_preview_failed_",)),),
        now=now,
    )


def _cleanup_aged_directories(
    roots: tuple[tuple[Path, tuple[str, ...]], ...],
    *,
    now: datetime | None,
) -> dict:
    result = _result()
    current = now or datetime.now(JST)
    if current.tzinfo is None:
        current = current.replace(tzinfo=JST)
    cutoff = current.timestamp() - ARTIFACT_MAX_AGE.total_seconds()

    for root, prefixes in roots:
        if not _safe_directory(root):
            continue
        try:
            root_resolved = root.resolve(strict=True)
            entries = list(root.iterdir())
        except OSError as error:
            result["errors"].append({"path": str(root), "reason": f"read_failed:{type(error).__name__}:{error}"})
            continue
        for path in entries:
            if path.is_symlink() or not path.is_dir():
                continue
            if not _matches_timestamped_directory(path.name, prefixes):
                continue
            try:
                resolved = path.resolve(strict=True)
                modified = path.stat().st_mtime
            except OSError as error:
                result["errors"].append({"path": str(path), "reason": f"stat_failed:{type(error).__name__}"})
                continue
            if resolved.parent != root_resolved:
                result["skipped"].append({"path": str(path), "reason": "outside_root"})
                continue
            if modified > cutoff:
                result["skipped"].append({"path": str(path), "reason": "younger_than_24_hours"})
                continue
            try:
                shutil.rmtree(path)
            except OSError as error:
                result["errors"].append({"path": str(path), "reason": f"delete_failed:{type(error).__name__}:{error}"})
            else:
                result["deleted"].append(str(path))
    return result


def _timestamped_directories(backups_dir: Path, prefix: str) -> list[Path]:
    if not _safe_directory(backups_dir):
        return []
    candidates: list[Path] = []
    marker = f"{prefix}_"
    try:
        entries = list(backups_dir.iterdir())
    except OSError:
        return []
    for path in entries:
        if path.is_symlink() or not path.is_dir() or not path.name.startswith(marker):
            continue
        if TIMESTAMPED_DIR_SUFFIX.fullmatch(path.name.removeprefix(marker)):
            candidates.append(path)
    return candidates


def _matches_timestamped_directory(name: str, prefixes: tuple[str, ...]) -> bool:
    for prefix in prefixes:
        if name.startswith(prefix) and TIMESTAMPED_DIR_SUFFIX.fullmatch(name.removeprefix(prefix)):
            return True
    return False


def _timestamped_directory_sort_key(path: Path, prefix: str) -> tuple[str, int]:
    suffix = path.name.removeprefix(f"{prefix}_")
    timestamp = suffix[:15]
    collision_suffix = suffix[16:] if len(suffix) > 15 else ""
    return timestamp, int(collision_suffix) if collision_suffix.isdigit() else 0


def _safe_directory(path: Path) -> bool:
    try:
        return path.is_dir() and not path.is_symlink()
    except OSError:
        return False


def _delete_oldest(candidates: list[Path], *, keep: int) -> dict:
    result = _result()
    keep = max(0, int(keep))
    ordered = list(candidates)
    for path in ordered[: max(0, len(ordered) - keep)]:
        try:
            if path.is_symlink():
                result["skipped"].append({"path": str(path), "reason": "symlink"})
            elif path.is_dir():
                shutil.rmtree(path)
                result["deleted"].append(str(path))
            elif path.is_file():
                path.unlink()
                result["deleted"].append(str(path))
            else:
                result["skipped"].append({"path": str(path), "reason": "unsupported_type"})
        except OSError as error:
            result["errors"].append({"path": str(path), "reason": f"delete_failed:{type(error).__name__}:{error}"})
    return result


def _result() -> dict:
    return {"deleted": [], "skipped": [], "errors": []}


def _merge_result(target: dict, source: dict) -> None:
    for key in ("deleted", "skipped", "errors"):
        target[key].extend(source.get(key, []))
