from __future__ import annotations

import unicodedata
from pathlib import Path
from threading import Event
from typing import Callable

from pef2_engine import workspace_paths
from pef2_engine.dictionary_candidate import (
    CandidateRecord,
    aggregate_suspicious_terms,
    build_ai_review_terms,
    build_gemini_prompt_preview,
    extract_english_terms,
    extract_suspicious_terms,
    split_into_sentences,
    tokenize_for_dictionary_candidates,
)
from pef2_engine.dictionary_loader import find_dictionary_matches, load_reading_dictionaries
from pef2_engine.gemini_dictionary_review import (
    CHUNK_SIZE,
    MAX_REVIEW_TERMS,
    AIDictionaryReviewCancelled,
    run_gemini_review,
)
from pef2_engine.io_utils import read_text, write_json


STEP5_DIRNAME = "step5"
SUSPICIOUS_TERMS_FILENAME = "suspicious_terms.json"
AI_REVIEW_TERMS_FILENAME = "ai_review_terms.json"
GEMINI_PROMPT_PREVIEW_FILENAME = "gemini_prompt_preview.json"
GEMINI_REVIEW_RAW_FILENAME = "gemini_review_raw.json"
GEMINI_REVIEW_LOG_FILENAME = "gemini_review_log.jsonl"


def run_step5_dictionary_draft(
    *,
    work_dir: Path,
    project_root: Path | None = None,
    run_gemini: bool = False,
    max_terms: int = MAX_REVIEW_TERMS,
    chunk_size: int = CHUNK_SIZE,
    cancel_event: Event | None = None,
    before_commit: Callable[[], bool] | None = None,
) -> dict:
    project_root = project_root or workspace_paths.PROJECT_ROOT
    work_dir = Path(work_dir)
    workspace_root = work_dir.parent
    source_path = work_dir / workspace_paths.SOURCE_ORIGINAL_FILENAME
    step5_dir = work_dir / STEP5_DIRNAME
    suspicious_terms_path = step5_dir / SUSPICIOUS_TERMS_FILENAME
    ai_review_terms_path = step5_dir / AI_REVIEW_TERMS_FILENAME
    prompt_preview_path = step5_dir / GEMINI_PROMPT_PREVIEW_FILENAME
    raw_path = step5_dir / GEMINI_REVIEW_RAW_FILENAME
    log_path = step5_dir / GEMINI_REVIEW_LOG_FILENAME
    draft_path = workspace_paths.work_dictionary_draft_path(work_dir)

    if not source_path.exists():
        raise FileNotFoundError(source_path)

    _raise_if_cancelled(cancel_event)
    source_text = read_text(source_path)
    dictionary_entries = _load_workspace_dictionaries(work_dir, workspace_root)
    records, warnings = _build_candidate_records(source_text, dictionary_entries)
    _raise_if_cancelled(cancel_event)
    suspicious_terms = aggregate_suspicious_terms(records)
    ai_review_terms = build_ai_review_terms(
        suspicious_terms,
        records,
        max_terms=max_terms,
    )
    prompt_preview = build_gemini_prompt_preview(ai_review_terms)
    _raise_if_cancelled(cancel_event)
    if warnings:
        ai_review_terms["warnings"] = sorted(set(warnings))
        prompt_preview["warnings"] = sorted(set(warnings))

    write_json(suspicious_terms_path, suspicious_terms)
    write_json(ai_review_terms_path, ai_review_terms)
    write_json(prompt_preview_path, prompt_preview)
    review_result = run_gemini_review(
        ai_review_terms_path=ai_review_terms_path,
        raw_path=raw_path,
        draft_path=draft_path,
        log_path=log_path,
        project_root=project_root,
        run_gemini=run_gemini,
        max_terms=max_terms,
        chunk_size=chunk_size,
        cancel_event=cancel_event,
        before_commit=before_commit,
    )

    _raise_if_cancelled(cancel_event)
    if (work_dir / workspace_paths.WORK_META_FILENAME).exists():
        workspace_paths.update_work_meta_status(work_dir, "dictionary_draft")

    return {
        "status": review_result["status"],
        "work_dir": str(work_dir),
        "source": str(source_path),
        "step5_dir": str(step5_dir),
        "suspicious_terms": str(suspicious_terms_path),
        "ai_review_terms": str(ai_review_terms_path),
        "gemini_prompt_preview": str(prompt_preview_path),
        "gemini_review_raw": str(raw_path),
        "gemini_review_log": str(log_path),
        "work_dictionary_draft": str(draft_path),
        "suspicious_term_count": len(suspicious_terms),
        "candidate_count": review_result["candidate_count"],
        "draft_count": review_result["draft_count"],
        "api_called": review_result["api_called"],
        "warnings": sorted(set(warnings)),
    }


def _build_candidate_records(
    source_text: str,
    dictionary_entries: list[dict],
) -> tuple[list[CandidateRecord], list[str]]:
    records: list[CandidateRecord] = []
    warnings: list[str] = []
    for index, raw_sentence in enumerate(split_into_sentences(source_text)):
        tokens, tokenizer_warnings = tokenize_for_dictionary_candidates(raw_sentence.text)
        warnings.extend(tokenizer_warnings)
        dictionary_matches = find_dictionary_matches(raw_sentence.text, dictionary_entries)
        suspicious_terms = extract_english_terms(raw_sentence.text, dictionary_matches)
        suspicious_terms.extend(extract_suspicious_terms(tokens, dictionary_matches))
        records.append(
            CandidateRecord(
                index=index,
                paragraph_index=raw_sentence.paragraph_index,
                line_index=raw_sentence.line_index,
                sentence_index=raw_sentence.sentence_index,
                tokens={"sudachi": tokens},
                dictionary_matches=dictionary_matches,
                suspicious_terms=suspicious_terms,
            )
        )
    return records, warnings


def _load_workspace_dictionaries(work_dir: Path, workspace_root: Path) -> list[dict]:
    system_dir = workspace_root / "dictionaries" / "system"
    user_dir = workspace_root / "dictionaries" / "user"
    system_dict_path = _resolve_optional_child(system_dir, "★システム固定辞書.json")
    standard_english_dict_path = _resolve_optional_child(system_dir, "★標準英単語辞書.json")
    user_dict_path = _resolve_optional_child(user_dir, "★ユーザ辞書.json")
    work_dict_path = workspace_paths.work_dictionary_path(work_dir)
    return load_reading_dictionaries(
        system_dict_path,
        standard_english_dict_path,
        user_dict_path,
        work_dict_path,
    )


def _resolve_optional_child(parent: Path, filename: str) -> Path:
    path = parent / filename
    if path.exists():
        return path
    if not parent.exists():
        return path
    target_name = unicodedata.normalize("NFC", filename)
    for child in parent.iterdir():
        if unicodedata.normalize("NFC", child.name) == target_name:
            return child
    return path


def _raise_if_cancelled(cancel_event: Event | None) -> None:
    if cancel_event is not None and cancel_event.is_set():
        raise AIDictionaryReviewCancelled()
