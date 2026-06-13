from __future__ import annotations

from copy import deepcopy


SCHEMA_VERSION = "dictionary_review-1"
SOURCE = "work_dictionary_draft"
DEFAULT_GENERATED_FROM = "work_dictionary_draft.json"
EMPTY_SOURCE = "empty_dictionary"
EMPTY_GENERATED_FROM = "none"
MANUAL_SOURCE = "manual_dictionary"
MANUAL_GENERATED_FROM = "manual"
DECISIONS = {"pending", "accept", "edit", "ignore"}


def build_dictionary_review(
    draft_data: object,
    input_stem: str = "unknown",
    generated_from: str = DEFAULT_GENERATED_FROM,
) -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "source": SOURCE,
        "input_stem": input_stem or "unknown",
        "generated_from": generated_from,
        "items": [
            _build_review_item(raw_item, fallback_index)
            for fallback_index, raw_item in enumerate(_draft_items(draft_data), start=1)
            if isinstance(raw_item, dict)
        ],
    }


def build_empty_dictionary_review(input_stem: str = "unknown") -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "source": EMPTY_SOURCE,
        "input_stem": input_stem or "unknown",
        "generated_from": EMPTY_GENERATED_FROM,
        "items": [],
    }


def build_manual_dictionary_review(input_stem: str = "unknown") -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "source": MANUAL_SOURCE,
        "input_stem": input_stem or "unknown",
        "generated_from": MANUAL_GENERATED_FROM,
        "items": [],
    }


def append_manual_dictionary_review_item(
    original_review: object,
    *,
    word: object,
    reading: object,
    notes: object = "",
) -> dict:
    if not isinstance(original_review, dict):
        raise DictionaryReviewValidationError(
            [{"code": "invalid_original_review", "message": "original review must be object"}]
        )

    original_items = original_review.get("items")
    if not isinstance(original_items, list):
        raise DictionaryReviewValidationError(
            [{"code": "invalid_original_items", "message": "original review items must be array"}]
        )

    normalized_word = _normalize_text(word)
    normalized_reading = _normalize_text(reading)
    errors: list[dict] = []
    if normalized_word == "" or normalized_reading == "":
        errors.append(
            {
                "code": "missing_manual_word_or_reading",
                "message": "単語と読みを入力してください。",
            }
        )
    if "\n" in normalized_word or "\r" in normalized_word:
        errors.append(
            {
                "code": "manual_word_has_newline",
                "message": "単語と読みを入力してください。",
            }
        )
    if "\n" in normalized_reading or "\r" in normalized_reading:
        errors.append(
            {
                "code": "manual_reading_has_newline",
                "message": "単語と読みを入力してください。",
            }
        )

    for item in original_items:
        if isinstance(item, dict) and str(item.get("word") or "") == normalized_word:
            errors.append(
                {
                    "code": "duplicate_manual_word",
                    "message": "同じ単語がすでにあります。既存の項目を修正してください。",
                    "word": normalized_word,
                }
            )
            break

    if errors:
        raise DictionaryReviewValidationError(errors)

    next_index = _next_review_item_index(original_items)
    updated_review = deepcopy(original_review)
    updated_items = deepcopy(original_items)
    updated_items.append(
        {
            "index": next_index,
            "word": normalized_word,
            "reading_suggested": normalized_reading,
            "reading_final": normalized_reading,
            "meaning": "",
            "difficulty": None,
            "confidence": "manual",
            "decision": "accept",
            "target_dictionary": "work",
            "promote_to_user_dictionary": False,
            "source": "manual",
            "notes": _stringify(notes),
        }
    )
    updated_review["items"] = updated_items
    return updated_review


def _draft_items(draft_data: object) -> list:
    if isinstance(draft_data, list):
        return draft_data
    return []


def _build_review_item(raw_item: dict, fallback_index: int) -> dict:
    reading = raw_item.get("読み", "")
    return {
        "index": raw_item.get("index", fallback_index),
        "word": raw_item.get("単語原文", ""),
        "reading_suggested": reading,
        "reading_final": reading,
        "meaning": raw_item.get("意味", ""),
        "difficulty": raw_item.get("難易度", None),
        "confidence": raw_item.get("confidence", ""),
        "decision": "pending",
        "target_dictionary": raw_item.get("target_dictionary", "work"),
        "promote_to_user_dictionary": raw_item.get("promote_to_user_dictionary", False),
        "source": raw_item.get("source", SOURCE),
        "notes": raw_item.get("notes", ""),
    }


class DictionaryReviewValidationError(ValueError):
    def __init__(self, errors: list[dict]):
        super().__init__("dictionary review validation failed")
        self.errors = errors


def apply_dictionary_review_form_update(original_review: object, form_items: object) -> dict:
    errors: list[dict] = []
    if not isinstance(original_review, dict):
        raise DictionaryReviewValidationError(
            [{"code": "invalid_original_review", "message": "original review must be object"}]
        )

    original_items = original_review.get("items")
    if not isinstance(original_items, list):
        raise DictionaryReviewValidationError(
            [{"code": "invalid_original_items", "message": "original review items must be array"}]
        )
    if not isinstance(form_items, list):
        raise DictionaryReviewValidationError(
            [{"code": "invalid_form_items", "message": "form items must be array"}]
        )
    if len(form_items) != len(original_items):
        raise DictionaryReviewValidationError(
            [
                {
                    "code": "item_count_mismatch",
                    "message": "item count does not match original review",
                    "expected": len(original_items),
                    "actual": len(form_items),
                }
            ]
        )

    updated_review = deepcopy(original_review)
    updated_items: list[dict] = []
    for position, (original_item, form_item) in enumerate(zip(original_items, form_items), start=1):
        if not isinstance(original_item, dict):
            errors.append(
                {
                    "code": "invalid_original_item",
                    "message": "original review item must be object",
                    "position": position,
                }
            )
            continue
        if not isinstance(form_item, dict):
            errors.append(
                {
                    "code": "invalid_form_item",
                    "message": "form item must be object",
                    "position": position,
                }
            )
            continue

        original_index = original_item.get("index")
        form_index = _normalize_index(form_item.get("index"))
        if form_index != original_index:
            errors.append(
                {
                    "code": "index_mismatch",
                    "message": "index does not match original review",
                    "position": position,
                    "expected": original_index,
                    "actual": form_index,
                }
            )

        original_word = str(original_item.get("word") or "")
        form_word = str(form_item.get("word") or "")
        if form_word != original_word:
            errors.append(
                {
                    "code": "word_mismatch",
                    "message": "word does not match original review",
                    "position": position,
                    "expected": original_word,
                    "actual": form_word,
                }
            )

        decision = str(form_item.get("decision") or "")
        if decision not in DECISIONS:
            errors.append(
                {
                    "code": "invalid_decision",
                    "message": "decision must be pending/accept/edit/ignore",
                    "position": position,
                    "decision": decision,
                }
            )

        reading_final = _normalize_text(form_item.get("reading_final"))
        if "\n" in reading_final or "\r" in reading_final:
            errors.append(
                {
                    "code": "reading_final_has_newline",
                    "message": "reading_final must not contain newline",
                    "position": position,
                }
            )

        resolved_decision = _resolve_finalize_decision(
            decision=decision,
            reading_suggested=_normalize_text(original_item.get("reading_suggested")),
            reading_final=reading_final,
        )
        if resolved_decision in {"accept", "edit"} and reading_final == "":
            errors.append(
                {
                    "code": "missing_reading_final",
                    "message": "accept/edit requires reading_final",
                    "position": position,
                }
            )

        updated_item = deepcopy(original_item)
        updated_item["reading_final"] = reading_final
        updated_item["decision"] = resolved_decision
        updated_item["notes"] = _stringify(form_item.get("notes"))
        updated_items.append(updated_item)

    if errors:
        raise DictionaryReviewValidationError(errors)

    updated_review["items"] = updated_items
    return updated_review


def _normalize_index(value: object) -> int | object:
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value
    try:
        return int(str(value))
    except (TypeError, ValueError):
        return value


def _next_review_item_index(items: list) -> int:
    indexes = [
        item.get("index")
        for item in items
        if isinstance(item, dict) and isinstance(item.get("index"), int)
    ]
    return (max(indexes) if indexes else 0) + 1


def _normalize_text(value: object) -> str:
    return str(value or "").strip()


def _resolve_finalize_decision(
    *,
    decision: str,
    reading_suggested: str,
    reading_final: str,
) -> str:
    if decision == "ignore":
        return "ignore"
    if decision in {"accept", "edit"}:
        return decision
    if reading_final == "":
        return "pending"
    if reading_final == reading_suggested:
        return "accept"
    return "edit"


def _stringify(value: object) -> str:
    if value is None:
        return ""
    return str(value)
