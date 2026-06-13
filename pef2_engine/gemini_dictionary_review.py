from __future__ import annotations

import json
import os
import re
import signal
import threading
import time
from pathlib import Path

from pef2_engine.io_utils import read_json, write_json


GEMINI_MODEL = "gemini-3.1-flash-lite"
THROTTLE_TIME = 4.2
WAIT_429 = 15.0
WAIT_503 = 5.0
MAX_RETRIES = 5
REQUEST_TIMEOUT = 60
OUTER_TIMEOUT = REQUEST_TIMEOUT + 15
CHUNK_SIZE = 50
MAX_REVIEW_TERMS = 3000
TEMPERATURE = 0.1
RESPONSE_MIME_TYPE = "application/json"
API_KEY_PATTERN = re.compile(r"AIza[0-9A-Za-z_-]{10,}")


class GeminiCallTimeout(Exception):
    pass


class GeminiResponseParseError(Exception):
    def __init__(self, message: str, *, text: str) -> None:
        super().__init__(message)
        self.text = text


def run_gemini_review(
    *,
    ai_review_terms_path: Path,
    raw_path: Path,
    draft_path: Path,
    log_path: Path,
    project_root: Path,
    run_gemini: bool = False,
    max_terms: int = MAX_REVIEW_TERMS,
    chunk_size: int = CHUNK_SIZE,
) -> dict:
    ai_review_terms = read_json(ai_review_terms_path, default={})
    candidates = select_candidates(ai_review_terms, max_terms=max_terms)
    logs = [_build_log("start", candidate_count=len(candidates), model=GEMINI_MODEL)]

    if not run_gemini:
        logs.append(_build_log("skipped_api_disabled"))
        return _write_skipped_outputs(
            ai_review_terms_path,
            raw_path,
            draft_path,
            log_path,
            logs,
            candidates,
            "api_disabled",
            chunk_size,
        )

    if not candidates:
        logs.append(_build_log("skipped_no_candidates"))
        return _write_skipped_outputs(
            ai_review_terms_path,
            raw_path,
            draft_path,
            log_path,
            logs,
            candidates,
            "no_candidates",
            chunk_size,
        )

    api_key = _load_api_key(project_root)
    api_key_available = bool(api_key)
    logs.append(_build_log("api_key_available", available=api_key_available))
    if not api_key:
        logs.append(_build_log("skipped_missing_api_key"))
        return _write_skipped_outputs(
            ai_review_terms_path,
            raw_path,
            draft_path,
            log_path,
            logs,
            candidates,
            "missing_api_key",
            chunk_size,
        )

    logs.append(_build_log("api_import_start"))
    try:
        import google.generativeai  # noqa: F401
    except ImportError as error:
        logs.append(
            _build_log(
                "api_import_error",
                error_type=type(error).__name__,
                message=_safe_error(error, api_key),
            )
        )
        logs.append(_build_log("skipped_import_error"))
        return _write_skipped_outputs(
            ai_review_terms_path,
            raw_path,
            draft_path,
            log_path,
            logs,
            candidates,
            "import_error",
            chunk_size,
        )
    logs.append(_build_log("api_import_success"))

    chunks = chunk_candidates(candidates, chunk_size)
    raw_responses = []
    raw_error_responses = []
    chunk_summaries = []
    draft_items = []
    allowed_surfaces = {candidate["surface"] for candidate in candidates}
    seen_draft_surfaces = set()

    for chunk_index, chunk in enumerate(chunks):
        logs.append(
            _build_log(
                "chunk_start",
                chunk_index=chunk_index,
                candidate_count=len(chunk),
            )
        )
        chunk_result = _run_chunk_with_retries(chunk, chunk_index, api_key, logs)
        chunk_summaries.append(chunk_result["summary"])
        raw_error_responses.extend(chunk_result.get("raw_error_responses", []))
        if chunk_result["raw_response"]:
            raw_responses.append(chunk_result["raw_response"])
            normalized = normalize_gemini_items(
                chunk_result["raw_response"].get("parsed", []),
                allowed_surfaces,
            )
            for item in normalized:
                if item["単語原文"] in seen_draft_surfaces:
                    continue
                draft_items.append(item)
                seen_draft_surfaces.add(item["単語原文"])

    timeout_count = sum(
        1 for summary in chunk_summaries if summary.get("status") == "timeout"
    )
    failed_count = sum(
        1 for summary in chunk_summaries if summary.get("status") == "failed"
    )
    logs.append(
        _build_log(
            "finish",
            api_called=True,
            draft_count=len(draft_items),
            failed_count=failed_count,
            timeout_count=timeout_count,
        )
    )
    raw_result = _build_raw_result(
        ai_review_terms_path,
        candidates,
        chunk_summaries,
        raw_responses,
        raw_error_responses,
        api_called=True,
        skipped=False,
        chunk_size=chunk_size,
    )
    write_json(raw_path, raw_result)
    write_json(draft_path, draft_items)
    _write_jsonl(log_path, logs)
    return {
        "status": "completed" if failed_count == 0 and timeout_count == 0 else "completed_with_errors",
        "api_called": True,
        "candidate_count": len(candidates),
        "chunk_count": len(chunks),
        "timeout_count": timeout_count,
        "failed_count": failed_count,
        "draft_count": len(draft_items),
        "raw_path": str(raw_path),
        "draft_path": str(draft_path),
        "log_path": str(log_path),
    }


def select_candidates(
    ai_review_terms: dict,
    max_terms: int = MAX_REVIEW_TERMS,
) -> list[dict]:
    candidates = []
    for item in ai_review_terms.get("items", []):
        if item.get("target_dictionary") != "work":
            continue
        if item.get("promote_to_user_dictionary") is not False:
            continue
        if item.get("decision") != "pending":
            continue
        if not item.get("surface"):
            continue
        candidates.append(item)
        if len(candidates) >= max_terms:
            break
    return candidates


def chunk_candidates(
    candidates: list[dict],
    chunk_size: int = CHUNK_SIZE,
) -> list[list[dict]]:
    return [
        candidates[index : index + chunk_size]
        for index in range(0, len(candidates), chunk_size)
    ]


def build_system_instruction() -> str:
    return (
        "あなたは日本語TTS向けの読み辞書候補を作る辞書編纂者です。"
        "対象は作品辞書であり、全作品共通のユーザ辞書ではありません。"
        "辞書は短いほどよいです。100%読み間違えそうな語、初見で読みが止まりそうな語、"
        "作品固有語、人名、地名、当て字、難読語を優先してください。"
        "普通に読める一般語は返さないでください。"
        "候補語を勝手に分解しないでください。入力候補にない語を新規追加せず、"
        "入力候補と同じ表記の語だけを単語原文に入れてください。"
        "複合語や英語フレーズが候補に含まれる場合もむやみに分解しないでください。"
        "読みはカタカナを基本にし、スペースを入れないでください。"
        "JSON配列だけを返し、Markdownコードフェンスや説明文を返さないでください。"
        "返答は1個のJSON配列だけにしてください。空の場合は[]だけを1回だけ返してください。"
        "前置き、後書き、説明文、複数のJSON配列は禁止です。"
        "指定キー以外の余計なキーは返さないでください。"
        "target_dictionaryは返す場合でもworkにしてください。"
    )


def build_user_prompt(chunk: list[dict]) -> str:
    candidates = []
    for item in chunk:
        candidate = {
            "surface": item.get("surface", ""),
            "current_reading": item.get("current_reading", ""),
            "reasons": item.get("reasons", []),
            "count": item.get("count", 0),
        }
        if "source_terms" in item:
            candidate["source_terms"] = item["source_terms"]
        candidates.append(candidate)

    return (
        "以下の候補語について、日本語TTS向けの読みを確認してください。"
        "作品辞書ドラフトに入れる前提で、必要な候補だけをJSON配列で返してください。\n"
        "100%読み間違えそうな語、初見で読みが止まりそうな語を優先してください。\n"
        "普通に読める一般語は返さないでください。辞書は短いほどよいです。\n"
        "候補語を分解しないでください。入力候補と同じ表記だけ返してください。\n"
        "読みにはスペースを入れないでください。JSON配列だけ返してください。\n"
        "返答は1個のJSON配列だけです。空の場合は[]だけを1回だけ返してください。\n"
        "前置き、後書き、説明文、複数のJSON配列は禁止です。\n"
        "期待する形式:\n"
        "[{\"単語原文\":\"犍陀多\",\"読み\":\"カンダタ\",\"意味\":\"作品固有の人名\","
        "\"難易度\":5,\"confidence\":\"high\",\"target_dictionary\":\"work\"}]\n"
        f"候補:\n{json.dumps(candidates, ensure_ascii=False, indent=2)}"
    )


def call_gemini_chunk(
    chunk: list[dict],
    chunk_index: int,
    attempt: int,
    api_key: str,
    logs: list[dict],
) -> dict:
    import google.generativeai as genai

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(
        GEMINI_MODEL,
        system_instruction=build_system_instruction(),
    )
    user_prompt = build_user_prompt(chunk)
    generation_config = {
        "response_mime_type": RESPONSE_MIME_TYPE,
        "temperature": TEMPERATURE,
    }
    request_options = {"timeout": REQUEST_TIMEOUT}
    logs.append(_build_log("api_call_start", chunk_index=chunk_index, attempt=attempt))
    response = _generate_content_with_outer_timeout(
        model,
        user_prompt,
        generation_config,
        request_options,
    )
    logs.append(_build_log("api_call_returned", chunk_index=chunk_index, attempt=attempt))
    text = getattr(response, "text", "") or ""
    try:
        parsed, parse_note = _parse_gemini_response_text(text)
    except json.JSONDecodeError as error:
        raise GeminiResponseParseError(str(error), text=text) from error
    raw_response = {
        "chunk_index": chunk_index,
        "text": text,
        "parsed": parsed,
    }
    if parse_note:
        raw_response["parse_note"] = parse_note
    return raw_response


def _parse_gemini_response_text(text: str) -> tuple[list, str]:
    stripped = text.strip()
    if not stripped:
        return [], ""
    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError:
        parsed = _parse_multiple_json_arrays(stripped)
        return parsed, "multiple_json_arrays"
    return parsed if isinstance(parsed, list) else [], ""


def _parse_multiple_json_arrays(text: str) -> list:
    decoder = json.JSONDecoder()
    index = 0
    values = []
    while index < len(text):
        while index < len(text) and text[index].isspace():
            index += 1
        if index >= len(text):
            break
        value, next_index = decoder.raw_decode(text, index)
        if not isinstance(value, list):
            raise json.JSONDecodeError("Expected JSON array", text, index)
        values.extend(value)
        index = next_index
    if not values and not text:
        return []
    return values


def normalize_gemini_items(
    parsed_items: list[dict],
    allowed_surfaces: set[str],
) -> list[dict]:
    normalized = []
    seen_surfaces = set()

    for item in parsed_items:
        if not isinstance(item, dict):
            continue
        surface = str(item.get("単語原文", "")).strip()
        reading = str(item.get("読み", "")).replace(" ", "").strip()
        if not surface or not reading:
            continue
        if surface not in allowed_surfaces or surface in seen_surfaces:
            continue

        normalized.append(
            {
                "単語原文": surface,
                "読み": reading,
                "意味": str(item.get("意味", "")).strip(),
                "難易度": item.get("難易度", ""),
                "confidence": str(item.get("confidence", "")).strip(),
                "target_dictionary": "work",
                "source": "gemini",
            }
        )
        seen_surfaces.add(surface)
    return normalized


def _run_chunk_with_retries(
    chunk: list[dict],
    chunk_index: int,
    api_key: str,
    logs: list[dict],
) -> dict:
    last_error = ""
    last_error_type = ""
    last_status = "failed"
    last_raw_response = None
    raw_error_responses = []
    attempts = 0
    for attempt in range(MAX_RETRIES):
        attempts = attempt + 1
        time.sleep(THROTTLE_TIME + attempt * 2.0)
        try:
            logs.append(
                _build_log(
                    "api_call_prepare",
                    chunk_index=chunk_index,
                    attempt=attempts,
                    model=GEMINI_MODEL,
                )
            )
            raw_response = call_gemini_chunk(chunk, chunk_index, attempts, api_key, logs)
            response_count = len(raw_response.get("parsed", []))
            logs.append(
                _build_log(
                    "chunk_success",
                    chunk_index=chunk_index,
                    attempts=attempts,
                    response_count=response_count,
                )
            )
            return {
                "summary": {
                    "chunk_index": chunk_index,
                    "candidate_count": len(chunk),
                    "api_called": True,
                    "status": "success",
                    "attempts": attempts,
                    "response_count": response_count,
                },
                "raw_response": raw_response,
                "raw_error_responses": raw_error_responses,
            }
        except Exception as error:
            last_error = _safe_error(error, api_key)
            last_error_type = type(error).__name__
            if isinstance(error, GeminiResponseParseError):
                last_raw_response = {
                    "chunk_index": chunk_index,
                    "attempt": attempts,
                    "text": error.text,
                    "parsed": [],
                    "parse_error": {
                        "error_type": last_error_type,
                        "message": last_error,
                    },
                }
                raw_error_responses.append(last_raw_response)
            if isinstance(error, GeminiCallTimeout):
                last_status = "timeout"
                logs.append(
                    _build_log(
                        "api_call_timeout",
                        chunk_index=chunk_index,
                        attempt=attempts,
                        outer_timeout=OUTER_TIMEOUT,
                    )
                )
            else:
                last_status = "failed"
                logs.append(
                    _build_log(
                        "api_call_exception",
                        chunk_index=chunk_index,
                        attempt=attempts,
                        error_type=last_error_type,
                        message=last_error,
                    )
                )
            retry_wait = _get_retry_wait(error)
            logs.append(
                _build_log(
                    "chunk_retry",
                    chunk_index=chunk_index,
                    attempt=attempts,
                    error_type=last_error_type,
                    message=last_error,
                    retry_wait=retry_wait,
                )
            )
            if retry_wait is None or attempts >= MAX_RETRIES:
                break
            time.sleep(retry_wait)

    logs.append(
        _build_log(
            "chunk_failed",
            chunk_index=chunk_index,
            status=last_status,
            error_type=last_error_type,
            message=last_error,
        )
    )
    summary = {
        "chunk_index": chunk_index,
        "candidate_count": len(chunk),
        "api_called": True,
        "status": last_status,
        "attempts": attempts,
        "response_count": 0,
    }
    if last_status == "timeout":
        summary["outer_timeout"] = OUTER_TIMEOUT
    if last_error:
        summary["error_type"] = last_error_type
        summary["message"] = last_error
    return {
        "summary": summary,
        "raw_response": last_raw_response,
        "raw_error_responses": raw_error_responses,
    }


def _generate_content_with_outer_timeout(
    model,
    user_prompt: str,
    generation_config: dict,
    request_options: dict,
):
    if not hasattr(signal, "SIGALRM") or not hasattr(signal, "setitimer"):
        return model.generate_content(
            user_prompt,
            generation_config=generation_config,
            request_options=request_options,
        )
    if threading.current_thread() is not threading.main_thread():
        return model.generate_content(
            user_prompt,
            generation_config=generation_config,
            request_options=request_options,
        )

    previous_handler = signal.getsignal(signal.SIGALRM)
    previous_timer = signal.setitimer(signal.ITIMER_REAL, OUTER_TIMEOUT)
    signal.signal(signal.SIGALRM, _timeout_handler)
    try:
        return model.generate_content(
            user_prompt,
            generation_config=generation_config,
            request_options=request_options,
        )
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)
        if previous_timer[0] > 0:
            signal.setitimer(signal.ITIMER_REAL, *previous_timer)


def _write_skipped_outputs(
    ai_review_terms_path: Path,
    raw_path: Path,
    draft_path: Path,
    log_path: Path,
    logs: list[dict],
    candidates: list[dict],
    skip_reason: str,
    chunk_size: int,
) -> dict:
    logs.append(
        _build_log(
            "finish",
            api_called=False,
            draft_count=0,
            failed_count=0,
            timeout_count=0,
        )
    )
    raw_result = _build_raw_result(
        ai_review_terms_path,
        candidates,
        [],
        [],
        [],
        api_called=False,
        skipped=True,
        skip_reason=skip_reason,
        chunk_size=chunk_size,
    )
    write_json(raw_path, raw_result)
    write_json(draft_path, [])
    _write_jsonl(log_path, logs)
    return {
        "status": skip_reason,
        "api_called": False,
        "candidate_count": len(candidates),
        "chunk_count": 0,
        "timeout_count": 0,
        "failed_count": 0,
        "draft_count": 0,
        "raw_path": str(raw_path),
        "draft_path": str(draft_path),
        "log_path": str(log_path),
    }


def _build_raw_result(
    input_path: Path,
    candidates: list[dict],
    chunks: list[dict],
    raw_responses: list[dict],
    raw_error_responses: list[dict],
    api_called: bool,
    skipped: bool,
    chunk_size: int,
    skip_reason: str | None = None,
) -> dict:
    timeout_count = sum(1 for chunk in chunks if chunk.get("status") == "timeout")
    failed_count = sum(1 for chunk in chunks if chunk.get("status") == "failed")
    result = {
        "schema_version": "step5c-4",
        "purpose": "gemini_review_raw",
        "gemini_api_called": api_called,
        "skipped": skipped,
        "model": GEMINI_MODEL,
        "input_path": str(input_path),
        "candidate_count": len(candidates),
        "chunk_size": chunk_size,
        "timeout_count": timeout_count,
        "failed_count": failed_count,
        "chunks": chunks,
        "raw_responses": raw_responses,
    }
    if raw_error_responses:
        result["raw_error_responses"] = raw_error_responses
    if skip_reason:
        result["skip_reason"] = skip_reason
    return result


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


def _timeout_handler(signum, frame) -> None:
    raise GeminiCallTimeout("Gemini API call exceeded outer timeout")


def _get_retry_wait(error: Exception) -> float | None:
    if isinstance(error, GeminiCallTimeout):
        return 3.0
    if isinstance(error, GeminiResponseParseError):
        return 3.0
    message = str(error).lower()
    if "429" in message or "resource_exhausted" in message:
        return WAIT_429
    if "503" in message or "overloaded" in message or "unavailable" in message:
        return WAIT_503
    if "timeout" in message or "deadline" in message:
        return 3.0
    return None


def _build_log(event: str, **values) -> dict:
    return {"stage": "gemini_review", "event": event, **values}


def _safe_error(error: Exception, api_key: str = "") -> str:
    message = str(error)
    if api_key:
        message = message.replace(api_key, "[redacted]")
    message = API_KEY_PATTERN.sub("[redacted]", message)
    message = " ".join(message.split())
    return message[:200]


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(record, ensure_ascii=False) for record in records]
    content = "\n".join(lines)
    if content:
        content += "\n"
    path.write_text(content, encoding="utf-8")
