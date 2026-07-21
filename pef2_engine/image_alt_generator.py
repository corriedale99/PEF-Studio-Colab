from __future__ import annotations

import mimetypes
import os
import base64
import json
import time
import urllib.error
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path
from threading import Event
from typing import Any, Callable

from pef2_engine.image_alt_review import (
    DEFAULT_IMAGE_ALT_LENGTH_TARGET,
    image_alt_review_settings,
    normalize_alt_length_target,
    save_image_alt_review,
    sync_image_alt_review,
)
from pef2_engine.image_paths import ImagePathError, resolve_existing_image


IMAGE_ALT_MODEL = "gemini-3.1-flash-lite"
IMAGE_ALT_PROMPT_TEMPLATE = (
    "この画像のEPUB用代替テキストを日本語で作成してください。\n"
    "{alt_length_target}文字程度で、画像の情景が浮かぶように簡潔に表現してください。\n"
    "画像から分かる内容だけを書き、推測はしないでください。\n"
    "代替テキスト本文のみを出力してください。"
)
IMAGE_ALT_PROMPT = IMAGE_ALT_PROMPT_TEMPLATE.format(
    alt_length_target=DEFAULT_IMAGE_ALT_LENGTH_TARGET
)
IMAGE_ALT_MAX_CONCURRENCY = 1
IMAGE_ALT_THROTTLE_SECONDS = 5
IMAGE_ALT_MAX_RETRIES = 4
IMAGE_ALT_REQUEST_MODE = "single_image"
IMAGE_ALT_REQUEST_TIMEOUT = 60
IMAGE_SIGNATURES = {
    ".png": b"\x89PNG\r\n\x1a\n",
    ".jpg": b"\xff\xd8",
    ".jpeg": b"\xff\xd8",
}
JST = timezone(timedelta(hours=9))


class ImageAltGenerationCancelled(Exception):
    pass


ProgressCallback = Callable[[dict[str, Any]], None]


def run_image_alt_generation(
    work_dir: Path,
    project_root: Path,
    *,
    segment_index: str = "",
    cancel_event: Event | None = None,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, Any]:
    review = sync_image_alt_review(work_dir)
    alt_length_target = image_alt_review_settings(review)["alt_length_target"]
    targets, skipped_items = select_image_alt_generation_targets(work_dir, review, segment_index=segment_index)
    result = {
        "generation_kind": "image_alt",
        "status": "failed",
        "ok": False,
        "message": "",
        "model": IMAGE_ALT_MODEL,
        "request_mode": IMAGE_ALT_REQUEST_MODE,
        "max_concurrency": IMAGE_ALT_MAX_CONCURRENCY,
        "throttle_seconds": IMAGE_ALT_THROTTLE_SECONDS,
        "max_retries": IMAGE_ALT_MAX_RETRIES,
        "alt_length_target": alt_length_target,
        "target_count": len(targets),
        "skipped_count": len(skipped_items),
        "success_count": 0,
        "failed_count": 0,
        "skipped_items": skipped_items,
        "errors": [],
    }
    if not targets:
        result["message"] = "生成対象の画像がありません。"
        return result

    api_key = _load_api_key(project_root)
    if not api_key:
        message = "AI機能を使うには、AI APIキーの設定が必要です。"
        _mark_targets_error(work_dir, targets, message)
        result["message"] = message
        result["failed_count"] = len(targets)
        result["errors"] = [message]
        return result

    last_api_call_at = 0.0
    for current, target in enumerate(targets, start=1):
        _raise_if_cancelled(cancel_event)
        if progress_callback is not None:
            progress_callback(
                {
                    "current": current,
                    "total": len(targets),
                    "image_path": target["image_path"],
                    "message": f"{target['image_path']} のaltを生成しています。",
                }
            )
        try:
            alt_text, last_api_call_at = _generate_alt_with_retries(
                api_key=api_key,
                image_path=target["path"],
                alt_length_target=alt_length_target,
                cancel_event=cancel_event,
                last_api_call_at=last_api_call_at,
            )
        except ImageAltGenerationCancelled:
            raise
        except Exception as error:
            message = _safe_error_message(error, api_key)
            _update_item_error(work_dir, target["segment_index"], message)
            result["failed_count"] += 1
            result["errors"].append(message)
            continue
        _update_item_success(work_dir, target["segment_index"], alt_text)
        result["success_count"] += 1

    if result["failed_count"]:
        result["status"] = "completed_with_errors" if result["success_count"] else "failed"
        result["ok"] = bool(result["success_count"])
        result["message"] = "一部の画像でalt生成に失敗しました。" if result["success_count"] else "画像alt生成に失敗しました。"
    else:
        result["status"] = "success"
        result["ok"] = True
        result["message"] = "画像alt生成が完了しました。"
    return result


def select_image_alt_generation_targets(
    work_dir: Path,
    review: dict[str, Any],
    *,
    segment_index: str = "",
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    targets: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    requested = _segment_index_text(segment_index)
    for item in review.get("items") or []:
        if not isinstance(item, dict):
            continue
        item_index = _segment_index_text(item.get("segment_index"))
        if requested and item_index != requested:
            continue
        if not requested and item.get("send_to_ai") is not True:
            skipped.append(_skip_item(item, "send_to_ai_false"))
            continue
        if not requested and str(item.get("gemini_alt_ja") or "").strip():
            skipped.append(_skip_item(item, "already_generated"))
            continue
        path = _item_image_path(work_dir, item)
        if path is None or not path.is_file():
            skipped.append(_skip_item(item, "missing_image"))
            continue
        if not _looks_like_allowed_image(path):
            skipped.append(_skip_item(item, "invalid_image"))
            continue
        targets.append(
            {
                "segment_index": item_index,
                "image_path": str(item.get("image_path") or ""),
                "path": path,
            }
        )
    return targets, skipped


def _generate_alt_with_retries(
    *,
    api_key: str,
    image_path: Path,
    alt_length_target: int,
    cancel_event: Event | None,
    last_api_call_at: float,
) -> tuple[str, float]:
    last_error: Exception | None = None
    for attempt in range(1, IMAGE_ALT_MAX_RETRIES + 2):
        _raise_if_cancelled(cancel_event)
        last_api_call_at = _throttle_api_call(last_api_call_at, cancel_event)
        try:
            response = _generate_content_rest(api_key, image_path, alt_length_target=alt_length_target)
            text = _extract_response_text(response).strip()
            if not text:
                raise RuntimeError("Gemini APIの応答が空でした。")
            return text, last_api_call_at
        except Exception as error:
            last_error = error
            if attempt > IMAGE_ALT_MAX_RETRIES or _retry_wait(error) is None:
                break
            _wait_or_cancel(_retry_wait(error) or IMAGE_ALT_THROTTLE_SECONDS, cancel_event)
    raise RuntimeError(f"Gemini API呼び出しに失敗しました: {last_error}")


def build_image_alt_prompt(alt_length_target: Any = None) -> str:
    return IMAGE_ALT_PROMPT_TEMPLATE.format(
        alt_length_target=normalize_alt_length_target(alt_length_target)
    )


def _generate_content_rest(api_key: str, image_path: Path, *, alt_length_target: int) -> dict[str, Any]:
    image = _image_part(image_path)
    body = {
        "contents": [
            {
                "parts": [
                    {"text": build_image_alt_prompt(alt_length_target)},
                    {
                        "inline_data": {
                            "mime_type": image["mime_type"],
                            "data": base64.b64encode(image["data"]).decode("ascii"),
                        }
                    },
                ]
            }
        ],
        "generationConfig": {"temperature": 0.2},
    }
    request = urllib.request.Request(
        f"https://generativelanguage.googleapis.com/v1beta/models/{IMAGE_ALT_MODEL}:generateContent?key={api_key}",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=IMAGE_ALT_REQUEST_TIMEOUT) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        response_body = _safe_api_error_body(error.read().decode("utf-8", errors="replace"), api_key)
        raise RuntimeError(f"Gemini API HTTP {error.code}: {response_body}") from error
    except urllib.error.URLError as error:
        raise RuntimeError(f"Gemini API接続エラー: {error.reason}") from error


def _image_part(path: Path) -> dict[str, Any]:
    mime_type = mimetypes.guess_type(path.name)[0] or ""
    if mime_type not in {"image/png", "image/jpeg"}:
        raise RuntimeError("PNGまたはJPEG画像のみ送信できます。")
    return {"mime_type": mime_type, "data": path.read_bytes()}


def _extract_response_text(response: Any) -> str:
    if isinstance(response, dict):
        parts: list[str] = []
        for candidate in response.get("candidates") or []:
            if not isinstance(candidate, dict):
                continue
            content = candidate.get("content") or {}
            if not isinstance(content, dict):
                continue
            for part in content.get("parts") or []:
                if isinstance(part, dict) and part.get("text"):
                    parts.append(str(part["text"]))
        return "\n".join(parts)
    text = getattr(response, "text", "")
    if text:
        return str(text)
    candidates = getattr(response, "candidates", None) or []
    parts: list[str] = []
    for candidate in candidates:
        content = getattr(candidate, "content", None)
        for part in getattr(content, "parts", None) or []:
            value = getattr(part, "text", "")
            if value:
                parts.append(str(value))
    return "\n".join(parts)


def _update_item_success(work_dir: Path, segment_index: str, alt_text: str) -> None:
    review = sync_image_alt_review(work_dir)
    item = _item_by_segment_index(review, segment_index)
    if item is None:
        return
    item["gemini_alt_ja"] = alt_text
    item["error_message"] = ""
    item["generated_at"] = _timestamp()
    item["updated_at"] = item["generated_at"]
    if str(item.get("user_alt_ja") or "").strip():
        item["status"] = "edited"
    else:
        item["status"] = "generated"
    save_image_alt_review(work_dir, review)


def _update_item_error(work_dir: Path, segment_index: str, message: str) -> None:
    review = sync_image_alt_review(work_dir)
    item = _item_by_segment_index(review, segment_index)
    if item is None:
        return
    item["status"] = "error"
    item["error_message"] = message
    item["updated_at"] = _timestamp()
    save_image_alt_review(work_dir, review)


def _mark_targets_error(work_dir: Path, targets: list[dict[str, Any]], message: str) -> None:
    for target in targets:
        _update_item_error(work_dir, _segment_index_text(target.get("segment_index")), message)


def _item_by_segment_index(review: dict[str, Any], segment_index: str) -> dict[str, Any] | None:
    target = _segment_index_text(segment_index)
    for item in review.get("items") or []:
        if isinstance(item, dict) and _segment_index_text(item.get("segment_index")) == target:
            return item
    return None


def _segment_index_text(value: Any) -> str:
    return "" if value is None else str(value).strip()


def _item_image_path(work_dir: Path, item: dict[str, Any]) -> Path | None:
    image_path = str(item.get("image_path") or "").strip()
    if not image_path:
        filename = str(item.get("filename") or "").strip()
        image_path = f"images/{filename}" if filename else ""
    try:
        resolved = resolve_existing_image(work_dir / "images", image_path)
    except ImagePathError:
        return None
    return resolved.path if resolved is not None else None


def _looks_like_allowed_image(path: Path) -> bool:
    try:
        header = path.read_bytes()[:16]
    except OSError:
        return False
    signature = IMAGE_SIGNATURES.get(path.suffix.lower())
    return bool(signature and header.startswith(signature))


def _skip_item(item: dict[str, Any], reason: str) -> dict[str, Any]:
    return {
        "segment_index": item.get("segment_index"),
        "image_path": item.get("image_path") or "",
        "reason": reason,
    }


def _load_api_key(project_root: Path) -> str:
    env_value = os.getenv("GEMINI_API_KEY", "").strip()
    if env_value:
        return env_value
    env_path = project_root / ".env"
    if not env_path.exists():
        return ""
    for line in env_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        if key.strip() != "GEMINI_API_KEY":
            continue
        return _strip_env_value_comment(value).strip().strip("'\"")
    return ""


def _strip_env_value_comment(value: str) -> str:
    in_single_quote = False
    in_double_quote = False
    for index, char in enumerate(value):
        if char == "'" and not in_double_quote:
            in_single_quote = not in_single_quote
        elif char == '"' and not in_single_quote:
            in_double_quote = not in_double_quote
        elif char == "#" and not in_single_quote and not in_double_quote:
            return value[:index]
    return value


def _retry_wait(error: Exception) -> float | None:
    message = str(error).lower()
    if "429" in message or "resource_exhausted" in message:
        return 15.0
    if "503" in message or "overloaded" in message or "unavailable" in message:
        return 5.0
    if "timeout" in message or "deadline" in message:
        return 5.0
    return None


def _throttle_api_call(last_api_call_at: float, cancel_event: Event | None) -> float:
    elapsed = time.monotonic() - last_api_call_at
    if last_api_call_at and elapsed < IMAGE_ALT_THROTTLE_SECONDS:
        _wait_or_cancel(IMAGE_ALT_THROTTLE_SECONDS - elapsed, cancel_event)
    return time.monotonic()


def _wait_or_cancel(seconds: float, cancel_event: Event | None) -> None:
    if seconds <= 0:
        return
    if cancel_event is not None and cancel_event.wait(seconds):
        raise ImageAltGenerationCancelled()
    if cancel_event is None:
        time.sleep(seconds)


def _raise_if_cancelled(cancel_event: Event | None) -> None:
    if cancel_event is not None and cancel_event.is_set():
        raise ImageAltGenerationCancelled()


def _safe_error_message(error: Exception, api_key: str) -> str:
    message = f"{type(error).__name__}: {error}"
    return message.replace(api_key, "[REDACTED]") if api_key else message


def _safe_api_error_body(body: str, api_key: str) -> str:
    redacted = body.replace(api_key, "[REDACTED]") if api_key else body
    return redacted[:1000]


def _timestamp() -> str:
    return datetime.now(JST).isoformat(timespec="seconds")
