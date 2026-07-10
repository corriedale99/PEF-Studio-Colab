from __future__ import annotations

from pathlib import Path
from threading import Event, Lock, Thread
from typing import Any

from pef2_engine import workspace_paths
from pef2_engine.epub_builder import EPUB_BUILD_REPORT_FILENAME, generate_epub_for_work
from pef2_engine.gemini_dictionary_review import AIDictionaryReviewCancelled
from pef2_engine.generation_lock import acquire_generation_lock, release_generation_lock
from pef2_engine.image_alt_generator import ImageAltGenerationCancelled, run_image_alt_generation
from pef2_engine.image_paths import ImagePathError, image_filename, resolve_existing_image
from pef2_engine.io_utils import read_json, write_json
from pef2_engine.tts_generator import (
    AUDIO_FILENAME,
    SYNC_MAP_FILENAME,
    TTS_BUILD_REPORT_FILENAME,
    VOICE_PREVIEW_DIRNAME,
    VOICE_PREVIEW_FILENAME,
    WORKSPACE_TEMP_DIRNAME,
    TTSGenerationCancelled,
    generate_voice_preview_for_work,
    generate_workspace_voice_preview,
    generate_tts_for_work,
)
from pef2_engine.tts_settings import (
    resolve_tts_settings,
    work_tts_settings_path,
    workspace_settings_path,
)
from pef2_studio.generation_progress import (
    PROGRESS_DIRNAME,
    create_task_id,
    is_valid_task_id,
    new_progress,
    read_progress,
    update_progress,
    write_progress,
)
from pef2_studio.workspace_view import (
    create_ai_dictionary_review_submission,
    load_ai_dictionary_review_confirmation,
    resolve_work_dir,
)


FINAL_REQUIRED_MESSAGE = "まだ原稿が確定していないため、音声を生成できません。先に編集画面で「編集完了として確定」を押してください。"
EPUB_FINAL_REQUIRED_MESSAGE = "まだ原稿が確定していないため、EPUBを生成できません。先に編集画面で「編集完了として確定」を押してください。"
EPUB_REEDITING_MESSAGE = "この作品は再編集中です。EPUBを生成するには、先に編集内容を「編集完了として確定」してください。"
TTS_FAILED_MESSAGE = "音声の生成に失敗したため、EPUB生成を中止しました。詳細は音声生成レポートを確認してください。"
VOICEVOX_CONNECTION_FAILED_MESSAGE = "音声生成に失敗しました。VOICEVOX（音声生成エンジン）を起動してから、もう一度EPUB生成を実行してください。"
VOICE_PREVIEW_CONNECTION_FAILED_MESSAGE = "VOICEVOX（音声生成エンジン）を起動してください。"
EPUB_FAILED_MESSAGE = "EPUBの生成に失敗しました。既存のEPUBがある場合は保持されています。詳細はEPUB生成レポートを確認してください。"
MISSING_IMAGES_CANCELLED_MESSAGE = "画像ファイルが足りないため、EPUB生成を中止しました。画像を追加してから、もう一度EPUB生成を実行してください。"
MISSING_IMAGES_ALLOWED_MESSAGE = "EPUBを生成しました。ただし、一部の画像が見つからなかったため、本文に代替表示を入れています。"
MISSING_IMAGES_MESSAGE = "画像ファイルが足りませんが、そのままEPUBを生成しますか？"
AI_DICTIONARY_RUNNING_MESSAGE = "辞書候補を生成中"
AI_DICTIONARY_FAILED_MESSAGE = "辞書候補生成に失敗しました。時間をおいてもう一度試すか、手動で辞書項目を追加してください。"
TTS_CANCELLED_MESSAGE = "音声生成をキャンセルしました。"
TTS_CANCEL_UNAVAILABLE_MESSAGE = "音声生成はすでに完了しているため、キャンセルできません。"
AI_DICTIONARY_CANCELLED_MESSAGE = "辞書候補生成をキャンセルしました。"
AI_DICTIONARY_CANCEL_UNAVAILABLE_MESSAGE = "辞書候補生成はすでに完了しているため、キャンセルできません。"
IMAGE_ALT_RUNNING_MESSAGE = "画像alt生成中"
IMAGE_ALT_FAILED_MESSAGE = "画像alt生成に失敗しました。"
IMAGE_ALT_CANCELLED_MESSAGE = "画像alt生成をキャンセルしました。"
IMAGE_ALT_CANCEL_UNAVAILABLE_MESSAGE = "画像alt生成はすでに完了しているため、キャンセルできません。"

_TTS_TASKS: dict[str, dict[str, Any]] = {}
_TTS_TASKS_LOCK = Lock()
_AI_DICTIONARY_TASKS: dict[str, dict[str, Any]] = {}
_AI_DICTIONARY_TASKS_LOCK = Lock()
_IMAGE_ALT_TASKS: dict[str, dict[str, Any]] = {}
_IMAGE_ALT_TASKS_LOCK = Lock()


def run_tts_generation(workspace_root: Path, work_id: str) -> dict | None:
    work_dir = resolve_work_dir(workspace_root, work_id)
    if work_dir is None:
        return None

    final_path = work_dir / workspace_paths.PROCESSED_FINAL_FILENAME
    if not final_path.exists():
        return _preflight_result("tts", work_dir, FINAL_REQUIRED_MESSAGE, ["04_processed_final.json がありません。"])

    lock_result = acquire_generation_lock(work_dir, "tts")
    if not lock_result.get("ok"):
        return _lock_result("tts", work_dir, lock_result)

    try:
        try:
            report = generate_tts_for_work(
                work_dir,
                workspace_root,
                speaker_id=_studio_speaker_id(workspace_root, work_dir),
            )
        except Exception as error:
            return _exception_result("tts", work_dir, error)
    finally:
        release_generation_lock(work_dir, lock_result.get("lock"))

    return _tts_generation_result(workspace_root, work_dir, report)


def start_tts_generation_task(
    workspace_root: Path,
    work_id: str,
    *,
    next_action: str = "",
) -> dict | None:
    work_dir = resolve_work_dir(workspace_root, work_id)
    if work_dir is None:
        return None
    next_action = "epub" if next_action == "epub" else ""

    final_path = work_dir / workspace_paths.PROCESSED_FINAL_FILENAME
    if not final_path.exists():
        return _task_start_failed(
            _preflight_result("tts", work_dir, FINAL_REQUIRED_MESSAGE, ["04_processed_final.json がありません。"])
        )

    task_id = create_task_id("tts")
    lock_result = acquire_generation_lock(work_dir, "tts", task_id=task_id)
    if not lock_result.get("ok"):
        return _task_start_failed(_lock_result("tts", work_dir, lock_result))

    lock_data = lock_result.get("lock")
    cancel_event = Event()
    _register_tts_task(task_id, work_dir.name, cancel_event)
    progress = new_progress(
        work_dir,
        task_id=task_id,
        operation="tts",
        status="running",
        phase="音声生成中",
        message="音声生成中",
        lock_started_at=str(lock_data.get("started_at") or "") if isinstance(lock_data, dict) else "",
        next_action=next_action,
        next_action_status="pending" if next_action else "",
    )
    progress["cancellable"] = True
    write_progress(work_dir, progress)

    thread = Thread(
        target=_run_tts_generation_task,
        name=f"pef2-tts-{task_id}",
        args=(Path(workspace_root), work_dir, task_id, lock_data, next_action, cancel_event),
        daemon=True,
    )
    try:
        thread.start()
    except Exception as error:
        update_progress(
            work_dir,
            task_id,
            status="failed",
            phase="音声生成失敗",
            message="音声生成に失敗しました。",
            error={"message": f"{type(error).__name__}: {error}"},
        )
        release_generation_lock(work_dir, lock_data if isinstance(lock_data, dict) else None)
        _unregister_tts_task(task_id)
        return {
            "ok": False,
            "status": "failed",
            "message": "音声生成に失敗しました。",
            "task_id": task_id,
            "progress": read_progress(work_dir, task_id),
        }

    return {
        "ok": True,
        "status": "started",
        "message": "音声生成中",
        "task_id": task_id,
        "progress": progress,
    }


def load_tts_generation_progress(workspace_root: Path, work_id: str, task_id: str) -> dict | None:
    if not is_valid_task_id(task_id, operation="tts"):
        return None
    work_dir = resolve_work_dir(workspace_root, work_id)
    if work_dir is None:
        return None
    return read_progress(work_dir, task_id)


def cancel_tts_generation_task(workspace_root: Path, work_id: str, task_id: str) -> dict | None:
    if not is_valid_task_id(task_id, operation="tts"):
        return None
    work_dir = resolve_work_dir(workspace_root, work_id)
    if work_dir is None:
        return None
    progress = read_progress(work_dir, task_id)
    if progress is None:
        return None

    with _TTS_TASKS_LOCK:
        task = _TTS_TASKS.get(task_id)
        if task is None or task.get("work_id") != work_dir.name:
            return _cancel_unavailable_result(task_id, progress)
        state = str(task.get("state") or "")
        if state == "cancelling":
            return {
                "ok": True,
                "status": "cancelling",
                "message": "キャンセル中",
                "task_id": task_id,
                "progress": progress,
            }
        if state != "running":
            return _cancel_unavailable_result(task_id, progress)
        task["state"] = "cancelling"
        progress = update_progress(
            work_dir,
            task_id,
            status="cancelling",
            phase="キャンセル中",
            message="キャンセル中",
            cancellable=False,
            cancel_requested=True,
        ) or progress
        cancel_event = task.get("event")
        if isinstance(cancel_event, Event):
            cancel_event.set()
    return {
        "ok": True,
        "status": "cancelling",
        "message": "キャンセル中",
        "task_id": task_id,
        "progress": progress,
    }


def start_ai_dictionary_review_task(workspace_root: Path, work_id: str) -> dict | None:
    work_dir = resolve_work_dir(workspace_root, work_id)
    if work_dir is None:
        return None

    confirmation = load_ai_dictionary_review_confirmation(workspace_root, work_id)
    if confirmation is None:
        return None
    if confirmation.get("status") == "blocked":
        result = confirmation.get("result") or {}
        return _task_start_failed(
            _preflight_result(
                "ai_dictionary",
                work_dir,
                str(result.get("message") or AI_DICTIONARY_FAILED_MESSAGE),
                _ai_dictionary_preflight_lines(result),
            )
        )

    task_id = create_task_id("ai_dictionary")
    lock_result = acquire_generation_lock(work_dir, "ai_dictionary", task_id=task_id)
    if not lock_result.get("ok"):
        return _task_start_failed(_lock_result("ai_dictionary", work_dir, lock_result))

    lock_data = lock_result.get("lock")
    cancel_event = Event()
    _register_ai_dictionary_task(task_id, work_dir.name, cancel_event)
    progress = new_progress(
        work_dir,
        task_id=task_id,
        operation="ai_dictionary",
        status="running",
        phase="辞書候補生成中",
        message=AI_DICTIONARY_RUNNING_MESSAGE,
        lock_started_at=str(lock_data.get("started_at") or "") if isinstance(lock_data, dict) else "",
    )
    progress["cancellable"] = True
    write_progress(work_dir, progress)

    thread = Thread(
        target=_run_ai_dictionary_review_task,
        name=f"pef2-ai-dictionary-{task_id}",
        args=(Path(workspace_root), work_dir, task_id, lock_data, cancel_event),
        daemon=True,
    )
    try:
        thread.start()
    except Exception as error:
        update_progress(
            work_dir,
            task_id,
            status="failed",
            phase="AI辞書候補作成失敗",
            message=AI_DICTIONARY_FAILED_MESSAGE,
            error={"message": f"{type(error).__name__}: {error}"},
        )
        release_generation_lock(work_dir, lock_data if isinstance(lock_data, dict) else None)
        _unregister_ai_dictionary_task(task_id)
        return {
            "ok": False,
            "status": "failed",
            "message": AI_DICTIONARY_FAILED_MESSAGE,
            "task_id": task_id,
            "progress": read_progress(work_dir, task_id),
        }

    return {
        "ok": True,
        "status": "started",
        "message": AI_DICTIONARY_RUNNING_MESSAGE,
        "task_id": task_id,
        "progress": progress,
    }


def load_ai_dictionary_review_progress(workspace_root: Path, work_id: str, task_id: str) -> dict | None:
    if not is_valid_task_id(task_id, operation="ai_dictionary"):
        return None
    work_dir = resolve_work_dir(workspace_root, work_id)
    if work_dir is None:
        return None
    return read_progress(work_dir, task_id)


def cancel_ai_dictionary_review_task(workspace_root: Path, work_id: str, task_id: str) -> dict | None:
    if not is_valid_task_id(task_id, operation="ai_dictionary"):
        return None
    work_dir = resolve_work_dir(workspace_root, work_id)
    if work_dir is None:
        return None
    progress = read_progress(work_dir, task_id)
    if progress is None:
        return None

    with _AI_DICTIONARY_TASKS_LOCK:
        task = _AI_DICTIONARY_TASKS.get(task_id)
        if task is None or task.get("work_id") != work_dir.name:
            return _ai_dictionary_cancel_unavailable_result(task_id, progress)
        state = str(task.get("state") or "")
        if state == "cancelling":
            return {
                "ok": True,
                "status": "cancelling",
                "message": "キャンセル中",
                "task_id": task_id,
                "progress": progress,
            }
        if state != "running":
            return _ai_dictionary_cancel_unavailable_result(task_id, progress)
        task["state"] = "cancelling"
        progress = update_progress(
            work_dir,
            task_id,
            status="cancelling",
            phase="キャンセル中",
            message="キャンセル中",
            cancellable=False,
            cancel_requested=True,
        ) or progress
        cancel_event = task.get("event")
        if isinstance(cancel_event, Event):
            cancel_event.set()
    return {
        "ok": True,
        "status": "cancelling",
        "message": "キャンセル中",
        "task_id": task_id,
        "progress": progress,
    }


def start_image_alt_generation_task(
    workspace_root: Path,
    work_id: str,
    *,
    segment_index: str = "",
) -> dict | None:
    work_dir = resolve_work_dir(workspace_root, work_id)
    if work_dir is None:
        return None

    task_id = create_task_id("image_alt")
    lock_result = acquire_generation_lock(work_dir, "image_alt", task_id=task_id)
    if not lock_result.get("ok"):
        return _task_start_failed(_lock_result("image_alt", work_dir, lock_result))

    lock_data = lock_result.get("lock")
    cancel_event = Event()
    _register_image_alt_task(task_id, work_dir.name, cancel_event)
    progress = new_progress(
        work_dir,
        task_id=task_id,
        operation="image_alt",
        status="running",
        phase="画像alt生成中",
        message=IMAGE_ALT_RUNNING_MESSAGE,
        lock_started_at=str(lock_data.get("started_at") or "") if isinstance(lock_data, dict) else "",
        result={"segment_index": str(segment_index or "")},
    )
    progress["cancellable"] = True
    write_progress(work_dir, progress)

    thread = Thread(
        target=_run_image_alt_generation_task,
        name=f"pef2-image-alt-{task_id}",
        args=(Path(workspace_root), work_dir, task_id, lock_data, cancel_event, str(segment_index or "")),
        daemon=True,
    )
    try:
        thread.start()
    except Exception as error:
        update_progress(
            work_dir,
            task_id,
            status="failed",
            phase="画像alt生成失敗",
            message=IMAGE_ALT_FAILED_MESSAGE,
            error={"message": f"{type(error).__name__}: {error}"},
            cancellable=False,
        )
        release_generation_lock(work_dir, lock_data if isinstance(lock_data, dict) else None)
        _unregister_image_alt_task(task_id)
        return {
            "ok": False,
            "status": "failed",
            "message": IMAGE_ALT_FAILED_MESSAGE,
            "task_id": task_id,
            "progress": read_progress(work_dir, task_id),
        }

    return {
        "ok": True,
        "status": "started",
        "message": IMAGE_ALT_RUNNING_MESSAGE,
        "task_id": task_id,
        "progress": progress,
    }


def load_image_alt_generation_progress(workspace_root: Path, work_id: str, task_id: str) -> dict | None:
    if not is_valid_task_id(task_id, operation="image_alt"):
        return None
    work_dir = resolve_work_dir(workspace_root, work_id)
    if work_dir is None:
        return None
    return read_progress(work_dir, task_id)


def cancel_image_alt_generation_task(workspace_root: Path, work_id: str, task_id: str) -> dict | None:
    if not is_valid_task_id(task_id, operation="image_alt"):
        return None
    work_dir = resolve_work_dir(workspace_root, work_id)
    if work_dir is None:
        return None
    progress = read_progress(work_dir, task_id)
    if progress is None:
        return None

    with _IMAGE_ALT_TASKS_LOCK:
        task = _IMAGE_ALT_TASKS.get(task_id)
        if task is None or task.get("work_id") != work_dir.name:
            return _image_alt_cancel_unavailable_result(task_id, progress)
        state = str(task.get("state") or "")
        if state == "cancelling":
            return {
                "ok": True,
                "status": "cancelling",
                "message": "キャンセル中",
                "task_id": task_id,
                "progress": progress,
            }
        if state != "running":
            return _image_alt_cancel_unavailable_result(task_id, progress)
        task["state"] = "cancelling"
        progress = update_progress(
            work_dir,
            task_id,
            status="cancelling",
            phase="キャンセル中",
            message="キャンセル中",
            cancellable=False,
            cancel_requested=True,
        ) or progress
        cancel_event = task.get("event")
        if isinstance(cancel_event, Event):
            cancel_event.set()
    return {
        "ok": True,
        "status": "cancelling",
        "message": "キャンセル中",
        "task_id": task_id,
        "progress": progress,
    }


def load_latest_blocked_generation_result(workspace_root: Path, work_id: str) -> dict | None:
    work_dir = resolve_work_dir(workspace_root, work_id)
    if work_dir is None:
        return None
    progress_dir = work_dir / PROGRESS_DIRNAME
    if not progress_dir.is_dir():
        return None
    latest_epub_mtime = _latest_official_epub_mtime(work_dir)
    candidates = sorted(
        progress_dir.glob("tts_*.json"),
        key=lambda path: path.stat().st_mtime if path.exists() else 0,
        reverse=True,
    )
    for path in candidates:
        try:
            if latest_epub_mtime is not None and path.stat().st_mtime <= latest_epub_mtime:
                continue
        except OSError:
            continue
        progress = read_progress(work_dir, path.stem)
        if not _is_blocked_epub_progress(progress):
            continue
        result = progress.get("result")
        if isinstance(result, dict):
            return _generation_result_from_progress(work_dir, result)
    return None


def _run_tts_generation_task(
    workspace_root: Path,
    work_dir: Path,
    task_id: str,
    lock_data: object,
    next_action: str,
    cancel_event: Event,
) -> None:
    lock_released = False
    try:
        try:
            report = generate_tts_for_work(
                work_dir,
                workspace_root,
                speaker_id=_studio_speaker_id(workspace_root, work_dir),
                cancel_event=cancel_event,
                before_commit=lambda: _begin_tts_commit(work_dir, task_id),
            )
            result = _tts_generation_result(workspace_root, work_dir, report)
        except TTSGenerationCancelled:
            _set_tts_task_state(task_id, "cancelled")
            update_progress(
                work_dir,
                task_id,
                status="cancelled",
                phase="音声生成キャンセル",
                message=TTS_CANCELLED_MESSAGE,
                result=None,
                error=None,
                cancellable=False,
                cancel_requested=True,
                next_action_status="",
            )
            return
        except Exception as error:
            result = _exception_result("tts", work_dir, error)

        status = "completed" if result.get("ok") else "failed"
        _set_tts_task_state(task_id, status)
        update_progress(
            work_dir,
            task_id,
            status=status,
            phase="音声生成完了" if status == "completed" else "音声生成失敗",
            message=str(result.get("message") or ""),
            result=_progress_result(result),
            error=None if status == "completed" else _progress_error(result),
            next_action_status="pending" if status == "completed" and next_action else "",
            cancellable=False,
        )
        if status != "completed" or next_action != "epub":
            return

        release_generation_lock(work_dir, lock_data if isinstance(lock_data, dict) else None)
        lock_released = True

        update_progress(
            work_dir,
            task_id,
            status="running",
            phase="EPUB生成中",
            message="音声生成が完了しました。EPUBを生成しています。",
            next_action_status="running",
            cancellable=False,
            finished_at="",
        )
        epub_result = run_epub_generation(workspace_root, work_dir.name)
        if epub_result is None:
            epub_result = _preflight_result("epub", work_dir, "作品が見つかりません。", ["work directory was not found."])
        next_status = _next_action_status(epub_result)
        progress_status = "completed" if next_status in {"completed", "blocked"} else "failed"
        update_progress(
            work_dir,
            task_id,
            status=progress_status,
            phase=_next_action_phase(next_status),
            message=str(epub_result.get("message") or ""),
            result=_progress_result(epub_result),
            error=None if next_status in {"completed", "blocked"} else _progress_error(epub_result),
            next_action_status=next_status,
        )
    finally:
        if not lock_released:
            release_generation_lock(work_dir, lock_data if isinstance(lock_data, dict) else None)
        _unregister_tts_task(task_id)


def _register_tts_task(task_id: str, work_id: str, cancel_event: Event) -> None:
    with _TTS_TASKS_LOCK:
        _TTS_TASKS[task_id] = {
            "work_id": work_id,
            "event": cancel_event,
            "state": "running",
        }


def _unregister_tts_task(task_id: str) -> None:
    with _TTS_TASKS_LOCK:
        _TTS_TASKS.pop(task_id, None)


def _set_tts_task_state(task_id: str, state: str) -> None:
    with _TTS_TASKS_LOCK:
        task = _TTS_TASKS.get(task_id)
        if task is not None:
            task["state"] = state


def _begin_tts_commit(work_dir: Path, task_id: str) -> bool:
    with _TTS_TASKS_LOCK:
        task = _TTS_TASKS.get(task_id)
        if task is None or task.get("state") != "running":
            return False
        cancel_event = task.get("event")
        if isinstance(cancel_event, Event) and cancel_event.is_set():
            task["state"] = "cancelling"
            return False
        task["state"] = "committing"
    update_progress(
        work_dir,
        task_id,
        status="committing",
        phase="音声保存中",
        message="音声を保存しています。",
        cancellable=False,
        cancel_requested=False,
    )
    return True


def _cancel_unavailable_result(task_id: str, progress: dict) -> dict:
    return {
        "ok": False,
        "status": "not_cancellable",
        "message": TTS_CANCEL_UNAVAILABLE_MESSAGE,
        "task_id": task_id,
        "progress": progress,
    }


def _ai_dictionary_cancel_unavailable_result(task_id: str, progress: dict) -> dict:
    return {
        "ok": False,
        "status": "not_cancellable",
        "message": AI_DICTIONARY_CANCEL_UNAVAILABLE_MESSAGE,
        "task_id": task_id,
        "progress": progress,
    }


def _image_alt_cancel_unavailable_result(task_id: str, progress: dict) -> dict:
    return {
        "ok": False,
        "status": "not_cancellable",
        "message": IMAGE_ALT_CANCEL_UNAVAILABLE_MESSAGE,
        "task_id": task_id,
        "progress": progress,
    }


def _run_ai_dictionary_review_task(
    workspace_root: Path,
    work_dir: Path,
    task_id: str,
    lock_data: object,
    cancel_event: Event,
) -> None:
    try:
        try:
            result = create_ai_dictionary_review_submission(
                workspace_root,
                work_dir.name,
                cancel_event=cancel_event,
                before_commit=lambda: _begin_ai_dictionary_commit(work_dir, task_id),
            )
            if result is None:
                result = _preflight_result(
                    "ai_dictionary",
                    work_dir,
                    "作品が見つかりません。",
                    ["work directory was not found."],
                )
        except AIDictionaryReviewCancelled:
            _set_ai_dictionary_task_state(task_id, "cancelled")
            update_progress(
                work_dir,
                task_id,
                status="cancelled",
                phase="辞書候補生成キャンセル",
                message=AI_DICTIONARY_CANCELLED_MESSAGE,
                result=None,
                error=None,
                cancellable=False,
                cancel_requested=True,
            )
            return
        except Exception as error:
            result = _exception_result("ai_dictionary", work_dir, error)

        status = "completed" if result.get("status") == "success" or result.get("ok") else "failed"
        _set_ai_dictionary_task_state(task_id, status)
        update_progress(
            work_dir,
            task_id,
            status=status,
            phase="辞書候補生成完了" if status == "completed" else "辞書候補生成失敗",
            message=str(result.get("message") or (AI_DICTIONARY_RUNNING_MESSAGE if status == "completed" else AI_DICTIONARY_FAILED_MESSAGE)),
            result=_ai_dictionary_progress_result(work_dir, result),
            error=None if status == "completed" else _ai_dictionary_progress_error(result),
            cancellable=False,
        )
    finally:
        release_generation_lock(work_dir, lock_data if isinstance(lock_data, dict) else None)
        _unregister_ai_dictionary_task(task_id)


def _register_ai_dictionary_task(task_id: str, work_id: str, cancel_event: Event) -> None:
    with _AI_DICTIONARY_TASKS_LOCK:
        _AI_DICTIONARY_TASKS[task_id] = {
            "work_id": work_id,
            "event": cancel_event,
            "state": "running",
        }


def _unregister_ai_dictionary_task(task_id: str) -> None:
    with _AI_DICTIONARY_TASKS_LOCK:
        _AI_DICTIONARY_TASKS.pop(task_id, None)


def _set_ai_dictionary_task_state(task_id: str, state: str) -> None:
    with _AI_DICTIONARY_TASKS_LOCK:
        task = _AI_DICTIONARY_TASKS.get(task_id)
        if task is not None:
            task["state"] = state


def _run_image_alt_generation_task(
    workspace_root: Path,
    work_dir: Path,
    task_id: str,
    lock_data: object,
    cancel_event: Event,
    segment_index: str,
) -> None:
    try:
        try:
            result = run_image_alt_generation(
                work_dir,
                workspace_paths.PROJECT_ROOT,
                segment_index=segment_index,
                cancel_event=cancel_event,
                progress_callback=lambda event: _update_image_alt_running_progress(work_dir, task_id, event),
            )
        except ImageAltGenerationCancelled:
            _set_image_alt_task_state(task_id, "cancelled")
            update_progress(
                work_dir,
                task_id,
                status="cancelled",
                phase="画像alt生成キャンセル",
                message=IMAGE_ALT_CANCELLED_MESSAGE,
                result=None,
                error=None,
                cancellable=False,
                cancel_requested=True,
            )
            return
        except Exception as error:
            result = _exception_result("image_alt", work_dir, error)

        status = "completed" if result.get("ok") else "failed"
        _set_image_alt_task_state(task_id, status)
        update_progress(
            work_dir,
            task_id,
            status=status,
            phase="画像alt生成完了" if status == "completed" else "画像alt生成失敗",
            message=str(result.get("message") or (IMAGE_ALT_RUNNING_MESSAGE if status == "completed" else IMAGE_ALT_FAILED_MESSAGE)),
            result=_image_alt_progress_result(work_dir, result),
            error=None if status == "completed" else _image_alt_progress_error(result),
            cancellable=False,
        )
    finally:
        release_generation_lock(work_dir, lock_data if isinstance(lock_data, dict) else None)
        _unregister_image_alt_task(task_id)


def _register_image_alt_task(task_id: str, work_id: str, cancel_event: Event) -> None:
    with _IMAGE_ALT_TASKS_LOCK:
        _IMAGE_ALT_TASKS[task_id] = {
            "work_id": work_id,
            "event": cancel_event,
            "state": "running",
        }


def _unregister_image_alt_task(task_id: str) -> None:
    with _IMAGE_ALT_TASKS_LOCK:
        _IMAGE_ALT_TASKS.pop(task_id, None)


def _set_image_alt_task_state(task_id: str, state: str) -> None:
    with _IMAGE_ALT_TASKS_LOCK:
        task = _IMAGE_ALT_TASKS.get(task_id)
        if task is not None:
            task["state"] = state


def _update_image_alt_running_progress(work_dir: Path, task_id: str, event: dict[str, Any]) -> None:
    current = _safe_int(event.get("current"))
    total = _safe_int(event.get("total"))
    percent = int(current * 100 / total) if total else None
    update_progress(
        work_dir,
        task_id,
        status="running",
        phase="画像alt生成中",
        message=str(event.get("message") or IMAGE_ALT_RUNNING_MESSAGE),
        percent=percent,
        result={
            "current": current,
            "total": total,
            "current_image": str(event.get("image_path") or ""),
        },
        cancellable=True,
    )


def _begin_ai_dictionary_commit(work_dir: Path, task_id: str) -> bool:
    with _AI_DICTIONARY_TASKS_LOCK:
        task = _AI_DICTIONARY_TASKS.get(task_id)
        if task is None or task.get("state") != "running":
            return False
        cancel_event = task.get("event")
        if isinstance(cancel_event, Event) and cancel_event.is_set():
            task["state"] = "cancelling"
            return False
        task["state"] = "committing"
    update_progress(
        work_dir,
        task_id,
        status="running",
        phase="辞書候補保存中",
        message="辞書候補を保存しています。",
        cancellable=False,
        cancel_requested=False,
    )
    return True


def _tts_generation_result(workspace_root: Path, work_dir: Path, report: dict) -> dict:
    report_path = work_dir / "audio" / TTS_BUILD_REPORT_FILENAME
    if report.get("ok"):
        meta_status = _meta_status(work_dir)
        if meta_status != "audio_generated":
            return _report_result(
                "tts",
                work_dir,
                "failed",
                "音声生成後の状態確認に失敗しました。",
                report,
                report_path,
                [f"meta.status が audio_generated ではありません: {meta_status or '不明'}"],
            )
        return _report_result(
            "tts",
            work_dir,
            "success",
            "音声生成が完了しました。",
            report,
            report_path,
            [
                _rel_work_path(work_dir, "audio", AUDIO_FILENAME),
                _rel_work_path(work_dir, "audio", SYNC_MAP_FILENAME),
                _rel_work_path(work_dir, "audio", TTS_BUILD_REPORT_FILENAME),
            ],
        )

    return _report_result(
        "tts",
        work_dir,
        "failed",
        _tts_failure_message(report, "音声生成に失敗しました。"),
        report,
        report_path,
        _report_error_lines(report),
    )


def _task_start_failed(result: dict) -> dict:
    return {
        "ok": False,
        "status": result.get("status") or "failed",
        "message": result.get("message") or "",
        "task_id": "",
        "progress": None,
        "errors": result.get("dev_log", []),
    }


def _progress_result(result: dict) -> dict:
    return {
        "generation_kind": result.get("generation_kind"),
        "status": result.get("status"),
        "ok": bool(result.get("ok")),
        "message": result.get("message") or "",
        "output_paths": result.get("output_paths") or [],
        "report_path": result.get("report_path") or "",
        "failed_build_dir": result.get("failed_build_dir") or "",
        "missing_files": result.get("missing_files") or [],
        "missing_images": result.get("missing_images") or [],
        "needs_confirmation": bool(result.get("needs_confirmation")),
        "download_ready": bool(result.get("download_ready")),
        "dev_log": list(result.get("dev_log") or [])[:8],
    }


def _progress_error(result: dict) -> dict:
    return {
        "message": result.get("message") or "音声生成に失敗しました。",
        "details": list(result.get("dev_log") or [])[:5],
    }


def _ai_dictionary_progress_result(work_dir: Path, result: dict) -> dict:
    raw_path = work_dir / "step5" / "gemini_review_raw.json"
    raw = read_json(raw_path, default={}) if raw_path.is_file() else {}
    raw_chunks = raw.get("chunks") if isinstance(raw, dict) else []
    warnings = result.get("warnings") or []
    return {
        "generation_kind": "ai_dictionary",
        "status": result.get("status"),
        "ok": result.get("status") == "success" or bool(result.get("ok")),
        "message": result.get("message") or "",
        "candidate_count": _safe_int(result.get("candidate_count"), _safe_int(raw.get("candidate_count") if isinstance(raw, dict) else 0)),
        "chunk_count": _safe_int(result.get("chunk_count"), len(raw_chunks) if isinstance(raw_chunks, list) else 0),
        "draft_count": _safe_int(result.get("draft_count")),
        "api_called": bool(result.get("api_called") if "api_called" in result else raw.get("gemini_api_called") if isinstance(raw, dict) else False),
        "skip_reason": result.get("skip_reason") or (raw.get("skip_reason") if isinstance(raw, dict) else ""),
        "failed_count": _safe_int(result.get("failed_count"), _safe_int(raw.get("failed_count") if isinstance(raw, dict) else 0)),
        "timeout_count": _safe_int(result.get("timeout_count"), _safe_int(raw.get("timeout_count") if isinstance(raw, dict) else 0)),
        "raw_path": _rel_path(raw_path) if raw_path.exists() else "",
        "draft_path": _rel_work_path(work_dir, workspace_paths.WORK_DICTIONARY_DRAFT_FILENAME) if (work_dir / workspace_paths.WORK_DICTIONARY_DRAFT_FILENAME).exists() else "",
        "log_path": _rel_work_path(work_dir, "step5", "gemini_review_log.jsonl") if (work_dir / "step5" / "gemini_review_log.jsonl").exists() else "",
        "backup_dir": str(result.get("backup_dir") or ""),
        "warnings": warnings[:5] if isinstance(warnings, list) else [],
    }


def _ai_dictionary_progress_error(result: dict) -> dict:
    details = []
    for key in ("failed_stage", "error_type", "affected_file"):
        value = result.get(key)
        if value:
            details.append(f"{key}: {value}")
    return {
        "message": result.get("message") or AI_DICTIONARY_FAILED_MESSAGE,
        "error_type": result.get("error_type") or "",
        "details": details[:5],
    }


def _image_alt_progress_result(work_dir: Path, result: dict) -> dict:
    return {
        "generation_kind": "image_alt",
        "status": result.get("status"),
        "ok": bool(result.get("ok")),
        "message": result.get("message") or "",
        "model": result.get("model") or "",
        "request_mode": result.get("request_mode") or "",
        "target_count": _safe_int(result.get("target_count")),
        "success_count": _safe_int(result.get("success_count")),
        "failed_count": _safe_int(result.get("failed_count")),
        "skipped_count": _safe_int(result.get("skipped_count")),
        "review_path": _rel_work_path(work_dir, workspace_paths.IMAGE_ALT_REVIEW_FILENAME),
        "skipped_items": list(result.get("skipped_items") or [])[:8],
    }


def _image_alt_progress_error(result: dict) -> dict:
    return {
        "message": result.get("message") or IMAGE_ALT_FAILED_MESSAGE,
        "details": list(result.get("errors") or [])[:5],
    }


def _ai_dictionary_preflight_lines(result: dict) -> list[str]:
    lines = ["preflightで停止しました。"]
    for key in ("failed_stage", "error_type", "affected_file"):
        value = result.get(key)
        if value:
            lines.append(f"{key}: {value}")
    return lines


def _safe_int(value: object, default: int = 0) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return default


def _is_blocked_epub_progress(progress: object) -> bool:
    if not isinstance(progress, dict):
        return False
    return (
        progress.get("operation") == "tts"
        and progress.get("next_action") == "epub"
        and progress.get("next_action_status") == "blocked"
    )


def _generation_result_from_progress(work_dir: Path, result: dict) -> dict:
    return {
        "generation_kind": result.get("generation_kind") or "epub",
        "status": result.get("status") or "needs_confirmation",
        "ok": bool(result.get("ok")),
        "message": result.get("message") or "",
        "output_paths": result.get("output_paths") or [],
        "report_path": result.get("report_path") or "",
        "failed_build_dir": result.get("failed_build_dir") or "",
        "missing_files": result.get("missing_files") or [],
        "missing_images": result.get("missing_images") or [],
        "needs_confirmation": bool(result.get("needs_confirmation")),
        "download_ready": bool(result.get("download_ready")),
        "dev_log": result.get("dev_log") or [],
        "work_id": work_dir.name,
    }


def _next_action_status(result: dict) -> str:
    if result.get("ok"):
        return "completed"
    if result.get("needs_confirmation"):
        return "blocked"
    return "failed"


def _next_action_phase(next_action_status: str) -> str:
    if next_action_status == "completed":
        return "EPUB生成完了"
    if next_action_status == "blocked":
        return "EPUB生成確認待ち"
    return "EPUB生成失敗"


def run_voice_preview_generation(workspace_root: Path, work_id: str, *, speaker_id: int | str | None = None) -> dict | None:
    work_dir = resolve_work_dir(workspace_root, work_id)
    if work_dir is None:
        return None

    report = generate_voice_preview_for_work(
        work_dir,
        workspace_root,
        speaker_id=speaker_id if speaker_id is not None else _studio_speaker_id(workspace_root, work_dir),
    )
    output_path = work_dir / "audio" / VOICE_PREVIEW_DIRNAME / VOICE_PREVIEW_FILENAME
    return _voice_preview_result(report, output_path, work_dir.name)


def run_workspace_voice_preview_generation(workspace_root: Path, *, speaker_id: int | str | None = None) -> dict:
    workspace_root = Path(workspace_root)
    report = generate_workspace_voice_preview(workspace_root, speaker_id=speaker_id)
    output_path = workspace_root / WORKSPACE_TEMP_DIRNAME / VOICE_PREVIEW_FILENAME
    return _voice_preview_result(report, output_path, "workspace")


def _voice_preview_result(report: dict, output_path: Path, work_id: str) -> dict:
    if report.get("ok") and output_path.exists():
        return {
            "status": "success",
            "ok": True,
            "message": "試聴音声を生成しました。",
            "speaker_id": report.get("speaker_id"),
            "output_path": _rel_path(output_path),
            "dev_log": _report_summary_lines(report),
            "work_id": work_id,
        }
    return {
        "status": "failed",
        "ok": False,
        "message": (
            VOICE_PREVIEW_CONNECTION_FAILED_MESSAGE
            if _looks_like_voicevox_connection_failure(_report_error_lines(report))
            else "試聴音声の生成に失敗しました。"
        ),
        "speaker_id": report.get("speaker_id"),
        "output_path": "",
        "dev_log": _report_error_lines(report),
        "work_id": work_id,
    }


def run_epub_generation(workspace_root: Path, work_id: str, *, allow_missing_images: bool = True) -> dict | None:
    work_dir = resolve_work_dir(workspace_root, work_id)
    if work_dir is None:
        return None

    original_meta = _read_meta(work_dir)
    preflight = _epub_preflight_before_audio(work_dir)
    if preflight is not None:
        return preflight

    lock_result = acquire_generation_lock(work_dir, "epub")
    if not lock_result.get("ok"):
        return _lock_result("epub", work_dir, lock_result)
    try:
        tts_result = _ensure_audio_for_epub(workspace_root, work_dir)
        if tts_result is not None:
            _restore_meta_after_failed_epub(work_dir, original_meta)
            return tts_result

        preflight = _epub_preflight_after_audio(work_dir, allow_missing_images=allow_missing_images)
        if preflight is not None:
            _restore_meta_after_failed_epub(work_dir, original_meta)
            return preflight

        try:
            report = generate_epub_for_work(work_dir, workspace_root, allow_missing_images=allow_missing_images)
        except Exception as error:
            _restore_meta_after_failed_epub(work_dir, original_meta)
            return _exception_result("epub", work_dir, error)
    finally:
        release_generation_lock(work_dir, lock_result.get("lock"))

    report_path = work_dir / "epub" / EPUB_BUILD_REPORT_FILENAME
    if _epub_completed(work_dir, report, report_path):
        meta_status = _meta_status(work_dir)
        if meta_status != "exported":
            _restore_meta_after_failed_epub(work_dir, original_meta)
            return _report_result(
                "epub",
                work_dir,
                "failed",
                "EPUB生成後の状態確認に失敗しました。",
                report,
                report_path,
                [f"meta.status が exported ではありません: {meta_status or '不明'}"],
            )
        output_epub = str(report.get("output_epub") or "")
        output_paths = []
        if output_epub:
            output_paths.append(_rel_work_path(work_dir, *output_epub.split("/")))
        output_paths.append(_rel_work_path(work_dir, "epub", EPUB_BUILD_REPORT_FILENAME))
        return _report_result(
            "epub",
            work_dir,
            "success",
            _epub_success_message(report),
            report,
            report_path,
            output_paths,
            download_ready=True,
        )

    _restore_meta_after_failed_epub(work_dir, original_meta)
    return _report_result(
        "epub",
        work_dir,
        "failed",
        EPUB_FAILED_MESSAGE,
        report,
        report_path,
        _report_error_lines(report),
    )


def _epub_preflight_before_audio(work_dir: Path) -> dict | None:
    final_path = work_dir / workspace_paths.PROCESSED_FINAL_FILENAME
    if not final_path.exists():
        return _preflight_result("epub", work_dir, EPUB_FINAL_REQUIRED_MESSAGE, ["04_processed_final.json がありません。"])
    if (work_dir / workspace_paths.PROCESSED_DRAFT_FILENAME).exists():
        return _preflight_result(
            "epub",
            work_dir,
            EPUB_REEDITING_MESSAGE,
            ["03_processed_draft.json があるため、再編集中として停止しました。"],
        )
    return None


def _epub_preflight_after_audio(work_dir: Path, *, allow_missing_images: bool) -> dict | None:
    audio_path = work_dir / "audio" / AUDIO_FILENAME
    sync_path = work_dir / "audio" / SYNC_MAP_FILENAME
    missing_audio = [path for path in (audio_path, sync_path) if not path.exists()]
    if missing_audio:
        return _preflight_result(
            "epub",
            work_dir,
            TTS_FAILED_MESSAGE,
            [f"足りないファイル: {_rel_path(path)}" for path in missing_audio],
            missing_files=[_rel_path(path) for path in missing_audio],
        )

    final_path = work_dir / workspace_paths.PROCESSED_FINAL_FILENAME
    try:
        processed = read_json(final_path)
    except Exception as error:
        return _preflight_result(
            "epub",
            work_dir,
            "04_processed_final.json を読み込めませんでした。",
            [f"{type(error).__name__}: {error}"],
        )

    missing_images = _missing_images(work_dir, processed)
    if missing_images and not allow_missing_images:
        return _preflight_result(
            "epub",
            work_dir,
            MISSING_IMAGES_MESSAGE,
            [f"index {item['index']}: {item['image_file']} ({item['searched_path']})" for item in missing_images],
            missing_images=missing_images,
            needs_confirmation=True,
        )
    return None


def _ensure_audio_for_epub(workspace_root: Path, work_dir: Path) -> dict | None:
    audio_path = work_dir / "audio" / AUDIO_FILENAME
    sync_path = work_dir / "audio" / SYNC_MAP_FILENAME
    if not _audio_needs_regeneration(workspace_root, work_dir, audio_path, sync_path):
        return None

    try:
        report = generate_tts_for_work(
            work_dir,
            workspace_root,
            speaker_id=_studio_speaker_id(workspace_root, work_dir),
        )
    except Exception as error:
        return _exception_result("tts", work_dir, error)

    report_path = work_dir / "audio" / TTS_BUILD_REPORT_FILENAME
    if report.get("ok") and audio_path.exists() and sync_path.exists():
        return None

    return _report_result(
        "tts",
        work_dir,
        "failed",
        _tts_failure_message(report, TTS_FAILED_MESSAGE),
        report,
        report_path,
        _report_error_lines(report) or [
            f"足りないファイル: {_rel_path(path)}"
            for path in (audio_path, sync_path)
            if not path.exists()
        ],
    )


def _audio_needs_regeneration(
    workspace_root: Path, work_dir: Path, audio_path: Path, sync_path: Path
) -> bool:
    final_path = work_dir / workspace_paths.PROCESSED_FINAL_FILENAME
    if not audio_path.exists() or not sync_path.exists():
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


def _studio_speaker_id(workspace_root: Path, work_dir: Path) -> int:
    return int(resolve_tts_settings(workspace_root, work_dir)["voice"]["speaker_id"])


def _effective_tts_settings_path(workspace_root: Path, work_dir: Path) -> Path | None:
    work_settings = work_tts_settings_path(work_dir)
    if work_settings.is_file():
        return work_settings
    workspace_settings = workspace_settings_path(workspace_root)
    if workspace_settings.is_file():
        return workspace_settings
    return None


def _epub_completed(work_dir: Path, report: dict, report_path: Path) -> bool:
    if not (report.get("ok") and report.get("committed")):
        return False
    if not report_path.exists():
        return False
    try:
        saved_report = read_json(report_path)
    except Exception:
        return False
    if not isinstance(saved_report, dict):
        return False
    if not (saved_report.get("ok") and saved_report.get("committed")):
        return False
    output_epub = str(saved_report.get("output_epub") or report.get("output_epub") or "")
    output_path = _official_epub_path(work_dir, output_epub)
    return output_path is not None and output_path.exists()


def _official_epub_path(work_dir: Path, output_epub: str) -> Path | None:
    parts = [part for part in output_epub.replace("\\", "/").split("/") if part]
    if len(parts) != 2 or parts[0] != "epub":
        return None
    filename = parts[1]
    if filename in {".", ".."} or "/" in filename or "\\" in filename or ".." in filename:
        return None
    path = work_dir / "epub" / filename
    return path if path.suffix == ".epub" else None


def _missing_images(work_dir: Path, processed: Any) -> list[dict]:
    segments = []
    if isinstance(processed, dict):
        value = processed.get("segments")
        if not isinstance(value, list) and isinstance(processed.get("remastered_data"), list):
            value = processed.get("remastered_data")
        if isinstance(value, list):
            segments = [item for item in value if isinstance(item, dict)]

    missing: list[dict] = []
    images_dir = work_dir / "images"
    for segment in segments:
        if not (segment.get("is_image") or segment.get("block_type") == "image"):
            continue
        image_file = str(segment.get("image_file") or "").strip()
        try:
            filename = image_filename(image_file)
            resolved = resolve_existing_image(images_dir, image_file)
            searched_path = resolved.path if resolved is not None else images_dir / filename
        except ImagePathError:
            filename = ""
            resolved = None
            searched_path = images_dir
        if not filename or resolved is None:
            missing.append(
                {
                    "index": segment.get("index", ""),
                    "image_file": image_file,
                    "searched_path": str(searched_path),
                }
            )
    return missing


def _preflight_result(
    generation_kind: str,
    work_dir: Path,
    message: str,
    dev_lines: list[str],
    *,
    missing_files: list[str] | None = None,
    missing_images: list[dict] | None = None,
    needs_confirmation: bool = False,
) -> dict:
    return {
        "generation_kind": generation_kind,
        "status": "needs_confirmation" if needs_confirmation else "preflight_failed",
        "ok": False,
        "message": message,
        "output_paths": [],
        "report_path": "",
        "failed_build_dir": "",
        "missing_files": missing_files or [],
        "missing_images": missing_images or [],
        "needs_confirmation": needs_confirmation,
        "download_ready": False,
        "dev_log": ["preflightで停止しました。", *dev_lines],
        "work_id": work_dir.name,
    }


def _lock_result(generation_kind: str, work_dir: Path, lock_result: dict) -> dict:
    return {
        "generation_kind": generation_kind,
        "status": str(lock_result.get("status") or "locked"),
        "ok": False,
        "message": str(lock_result.get("message") or ""),
        "output_paths": [],
        "report_path": "",
        "failed_build_dir": "",
        "missing_files": [],
        "missing_images": [],
        "needs_confirmation": False,
        "download_ready": False,
        "dev_log": _lock_dev_log(lock_result),
        "work_id": work_dir.name,
    }


def build_generation_notice_result(work_id: str, notice: str | None) -> dict | None:
    if notice != "missing_images_cancelled":
        return None
    return {
        "generation_kind": "epub",
        "status": "cancelled",
        "ok": False,
        "message": MISSING_IMAGES_CANCELLED_MESSAGE,
        "output_paths": [],
        "report_path": "",
        "failed_build_dir": "",
        "missing_files": [],
        "missing_images": [],
        "needs_confirmation": False,
        "download_ready": False,
        "dev_log": [],
        "work_id": work_id,
    }


def _lock_dev_log(lock_result: dict) -> list[str]:
    status = str(lock_result.get("status") or "locked")
    lines = [f"generation lock: {status}"]
    lock_data = lock_result.get("lock")
    if isinstance(lock_data, dict):
        operation = lock_data.get("operation")
        started_at = lock_data.get("started_at")
        if operation:
            lines.append(f"operation: {operation}")
        if started_at:
            lines.append(f"started_at: {started_at}")
    backup_path = str(lock_result.get("backup_path") or "")
    if backup_path:
        lines.append(f"backup_path: {backup_path}")
    error = str(lock_result.get("error") or "")
    if error:
        lines.append(error)
    return lines


def _report_result(
    generation_kind: str,
    work_dir: Path,
    status: str,
    message: str,
    report: dict,
    report_path: Path,
    detail_lines: list[str],
    *,
    download_ready: bool = False,
) -> dict:
    failed_build_dir = str(report.get("failed_build_dir") or "")
    dev_log = [
        f"ok: {bool(report.get('ok'))}",
        f"committed: {bool(report.get('committed'))}",
        f"report: {_rel_path(report_path)}",
    ]
    if failed_build_dir:
        dev_log.append(f"failed_build_dir: {failed_build_dir}")
    dev_log.extend(_report_summary_lines(report))
    dev_log.extend(detail_lines)
    return {
        "generation_kind": generation_kind,
        "status": status,
        "ok": status == "success",
        "message": message,
        "output_paths": detail_lines if status == "success" else [],
        "report_path": _rel_path(report_path) if report_path.exists() else "",
        "failed_build_dir": failed_build_dir,
        "missing_files": [],
        "missing_images": [],
        "needs_confirmation": False,
        "download_ready": download_ready,
        "dev_log": dev_log,
        "work_id": work_dir.name,
    }


def _exception_result(generation_kind: str, work_dir: Path, error: Exception) -> dict:
    message = "生成処理でエラーが発生しました。"
    if generation_kind == "tts" and _looks_like_voicevox_connection_failure([f"{type(error).__name__}: {error}"]):
        message = VOICEVOX_CONNECTION_FAILED_MESSAGE
    return {
        "generation_kind": generation_kind,
        "status": "failed",
        "ok": False,
        "message": message,
        "output_paths": [],
        "report_path": "",
        "failed_build_dir": "",
        "missing_files": [],
        "missing_images": [],
        "needs_confirmation": False,
        "download_ready": False,
        "dev_log": [f"{type(error).__name__}: {error}"],
        "work_id": work_dir.name,
    }


def _report_summary_lines(report: dict) -> list[str]:
    lines: list[str] = []
    for key in ("backend", "speaker_id", "segments", "tts_units", "sync_map_count", "duration_seconds", "output_epub"):
        if key in report and report.get(key) not in (None, ""):
            lines.append(f"{key}: {report.get(key)}")
    lines.extend(_report_error_lines(report))
    warning_count = len(report.get("warnings", [])) if isinstance(report.get("warnings"), list) else 0
    if warning_count:
        lines.append(f"warnings: {warning_count}")
    return lines


def _report_error_lines(report: dict) -> list[str]:
    errors = report.get("errors", [])
    if not isinstance(errors, list) or not errors:
        return []
    lines = [f"errors: {len(errors)}"]
    for item in errors[:5]:
        if isinstance(item, dict):
            code = item.get("code", "")
            message = item.get("message", "")
            index = item.get("index", item.get("segment_index", ""))
            prefix = f"{code}: "
            if index != "":
                prefix = f"{code}: index={index}: "
            lines.append(f"{prefix}{message}")
        else:
            lines.append(str(item))
    if len(errors) > 5:
        lines.append("errors は先頭5件のみ表示しています。")
    return lines


def _tts_failure_message(report: dict, fallback: str) -> str:
    lines = _report_error_lines(report)
    if _looks_like_voicevox_connection_failure(lines):
        return VOICEVOX_CONNECTION_FAILED_MESSAGE
    return fallback


def _looks_like_voicevox_connection_failure(lines: list[str]) -> bool:
    text = "\n".join(lines).lower()
    if "voicevox" not in text and "localhost:50021" not in text and "127.0.0.1:50021" not in text:
        return False
    connection_tokens = (
        "connection refused",
        "connection reset",
        "failed to establish a new connection",
        "max retries exceeded",
        "urlopen error",
        "connectionerror",
        "httperror",
        "httpconnectionpool",
        "timeout",
        "timed out",
        "operation not permitted",
        "nodename nor servname",
        "name or service not known",
    )
    return any(token in text for token in connection_tokens)


def _meta_status(work_dir: Path) -> str:
    return str(_read_meta(work_dir).get("status") or "")


def _epub_success_message(report: dict) -> str:
    if _has_warning_code(report, "missing_images"):
        return MISSING_IMAGES_ALLOWED_MESSAGE
    return "EPUB生成が完了しました。"


def _has_warning_code(report: dict, code: str) -> bool:
    warnings = report.get("warnings", [])
    if not isinstance(warnings, list):
        return False
    return any(isinstance(item, dict) and item.get("code") == code for item in warnings)


def _read_meta(work_dir: Path) -> dict:
    try:
        meta = read_json(work_dir / workspace_paths.WORK_META_FILENAME)
    except Exception:
        return {}
    return meta if isinstance(meta, dict) else {}


def _restore_meta_after_failed_epub(work_dir: Path, original_meta: dict) -> None:
    meta_path = work_dir / workspace_paths.WORK_META_FILENAME
    if not original_meta or not meta_path.exists():
        return
    restored = dict(original_meta)
    if restored.get("status") == "exported" and not _has_official_epub(work_dir):
        restored["status"] = "finalized" if (work_dir / workspace_paths.PROCESSED_FINAL_FILENAME).exists() else "processed"
    write_json(meta_path, restored)


def _has_official_epub(work_dir: Path) -> bool:
    epub_dir = work_dir / "epub"
    if not epub_dir.is_dir():
        return False
    return any(path.is_file() and path.suffix == ".epub" for path in epub_dir.glob("*.epub"))


def _latest_official_epub_mtime(work_dir: Path) -> float | None:
    epub_dir = work_dir / "epub"
    if not epub_dir.is_dir():
        return None
    mtimes: list[float] = []
    for path in epub_dir.glob("*.epub"):
        try:
            if path.is_file():
                mtimes.append(path.stat().st_mtime)
        except OSError:
            continue
    return max(mtimes) if mtimes else None


def _rel_work_path(work_dir: Path, *parts: str) -> str:
    return str(Path("workspace") / work_dir.name / Path(*parts))


def _rel_path(path: Path) -> str:
    path = Path(path)
    parts = path.parts
    if "workspace" in parts:
        index = parts.index("workspace")
        return str(Path(*parts[index:]))
    return str(path)
