from __future__ import annotations

from pathlib import Path
from typing import Any

from pef2_engine import workspace_paths
from pef2_engine.epub_builder import EPUB_BUILD_REPORT_FILENAME, generate_epub_for_work
from pef2_engine.io_utils import read_json, write_json
from pef2_engine.tts_generator import (
    AUDIO_FILENAME,
    SYNC_MAP_FILENAME,
    TTS_BUILD_REPORT_FILENAME,
    VOICE_PREVIEW_DIRNAME,
    VOICE_PREVIEW_FILENAME,
    WORKSPACE_TEMP_DIRNAME,
    generate_voice_preview_for_work,
    generate_workspace_voice_preview,
    generate_tts_for_work,
)
from pef2_engine.tts_settings import (
    resolve_tts_settings,
    work_tts_settings_path,
    workspace_settings_path,
)
from pef2_studio.workspace_view import resolve_work_dir


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


def run_tts_generation(workspace_root: Path, work_id: str) -> dict | None:
    work_dir = resolve_work_dir(workspace_root, work_id)
    if work_dir is None:
        return None

    final_path = work_dir / workspace_paths.PROCESSED_FINAL_FILENAME
    if not final_path.exists():
        return _preflight_result("tts", work_dir, FINAL_REQUIRED_MESSAGE, ["04_processed_final.json がありません。"])

    try:
        report = generate_tts_for_work(
            work_dir,
            workspace_root,
            speaker_id=_studio_speaker_id(workspace_root, work_dir),
        )
    except Exception as error:
        return _exception_result("tts", work_dir, error)

    report_path = work_dir / "audio" / TTS_BUILD_REPORT_FILENAME
    if report.get("ok"):
        meta_status = _meta_status(work_dir)
        if meta_status != "audio_generated":
            return _report_result(
                "tts",
                work_dir,
                "failed",
                "TTS生成後の状態確認に失敗しました。",
                report,
                report_path,
                [f"meta.status が audio_generated ではありません: {meta_status or '不明'}"],
            )
        return _report_result(
            "tts",
            work_dir,
            "success",
            "TTS生成が完了しました。",
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
        _tts_failure_message(report, "TTS生成に失敗しました。"),
        report,
        report_path,
        _report_error_lines(report),
    )


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


def run_epub_generation(workspace_root: Path, work_id: str, *, allow_missing_images: bool = False) -> dict | None:
    work_dir = resolve_work_dir(workspace_root, work_id)
    if work_dir is None:
        return None

    original_meta = _read_meta(work_dir)
    preflight = _epub_preflight_before_audio(work_dir)
    if preflight is not None:
        return preflight

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
        filename = _image_filename(image_file)
        searched_path = images_dir / filename if filename else images_dir
        if not filename or not searched_path.is_file():
            missing.append(
                {
                    "index": segment.get("index", ""),
                    "image_file": image_file,
                    "searched_path": str(searched_path),
                }
            )
    return missing


def _image_filename(image_file: str) -> str:
    raw = image_file.replace("\\", "/")
    parts = [part for part in raw.split("/") if part]
    if parts and parts[0] == "images":
        parts = parts[1:]
    if len(parts) != 1 or parts[0] in {".", ".."} or ".." in parts[0]:
        return ""
    return parts[0]


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


def _rel_work_path(work_dir: Path, *parts: str) -> str:
    return str(Path("workspace") / work_dir.name / Path(*parts))


def _rel_path(path: Path) -> str:
    path = Path(path)
    parts = path.parts
    if "workspace" in parts:
        index = parts.index("workspace")
        return str(Path(*parts[index:]))
    return str(path)
