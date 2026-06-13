from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pef2_engine.io_utils import read_json, write_json
from pef2_engine import workspace_paths


REPORT_SCHEMA_VERSION = "dictionary_finalize_report-1"
SOURCE_FILENAME = workspace_paths.DICTIONARY_REVIEW_FILENAME
REPORT_FILENAME = workspace_paths.DICTIONARY_FINALIZE_REPORT_FILENAME
VALID_DECISIONS = {"accept", "edit", "pending", "ignore"}
WRITE_DECISIONS = {"accept", "edit"}


def load_dictionary_review(path: Path) -> object:
    return read_json(path)


def finalize_work_dictionary(
    review_data: object,
    source_path: Path,
    work_dir: Path,
) -> tuple[list[dict] | None, dict, Path | None]:
    report = _base_report(source_path, work_dir)

    if not isinstance(review_data, dict):
        report["status"] = "failed"
        report["errors"].append(
            {
                "code": "invalid_top_level",
                "message": "dictionary_review top-level must be object",
            }
        )
        return None, report, None

    input_stem = review_data.get("input_stem")
    output_path = _work_dictionary_path(input_stem, work_dir)
    report["input_stem"] = input_stem if isinstance(input_stem, str) else ""
    report["output"] = workspace_paths.display_path(output_path, work_dir) if output_path is not None else ""

    if not isinstance(input_stem, str) or input_stem.strip() == "" or input_stem == "unknown":
        report["errors"].append(
            {
                "code": "invalid_input_stem",
                "input_stem": input_stem if input_stem is not None else "",
                "message": "input_stem is missing, empty, or unknown",
            }
        )
        output_path = None
        report["output"] = ""

    if "items" not in review_data:
        report["errors"].append(
            {
                "code": "missing_items",
                "message": "dictionary_review items is missing",
            }
        )
        items: list[Any] = []
    else:
        raw_items = review_data.get("items")
        if not isinstance(raw_items, list):
            report["errors"].append(
                {
                    "code": "invalid_items",
                    "message": "dictionary_review items must be array",
                }
            )
            items = []
        else:
            items = raw_items

    entries: list[dict] = []
    word_indexes: dict[str, list[int]] = {}

    for fallback_index, item in enumerate(items):
        if not isinstance(item, dict):
            report["warnings"].append(
                {
                    "code": "invalid_item",
                    "index": fallback_index,
                    "word": "",
                    "reason": "item is not object",
                    "message": "dictionary_review item is not object",
                }
            )
            continue

        decision = item.get("decision")
        word = _text_value(item.get("word"))
        index = item.get("index", fallback_index)

        if decision == "accept":
            report["accepted_count"] += 1
        elif decision == "edit":
            report["edited_count"] += 1
        elif decision == "ignore":
            report["ignored_count"] += 1
        elif decision == "pending":
            report["pending_count"] += 1
        else:
            report["warnings"].append(
                {
                    "code": "unknown_decision",
                    "index": index,
                    "word": word,
                    "reason": "decision is not accept/edit/pending/ignore",
                    "message": "unknown decision; item was not written",
                }
            )
            continue

        if decision not in WRITE_DECISIONS:
            continue

        reading_final = _text_value(item.get("reading_final"))
        if item.get("promote_to_user_dictionary") is True:
            report["warnings"].append(
                {
                    "code": "promote_to_user_dictionary",
                    "index": index,
                    "word": word,
                    "reason": "Step6-B does not update user dictionary",
                    "message": "promote_to_user_dictionary is true; user dictionary was not updated",
                }
            )

        if word == "":
            report["errors"].append(
                {
                    "code": "missing_word",
                    "index": index,
                    "word": word,
                    "message": "accept/edit item has empty word",
                }
            )
        if reading_final == "":
            report["errors"].append(
                {
                    "code": "missing_reading_final",
                    "index": index,
                    "word": word,
                    "message": "accept/edit item has empty reading_final",
                }
            )
        if word == "" or reading_final == "":
            continue

        word_indexes.setdefault(word, []).append(index)
        entries.append(
            {
                "単語原文": word,
                "読み": reading_final,
            }
        )

    for word, indexes in word_indexes.items():
        if len(indexes) > 1:
            report["errors"].append(
                {
                    "code": "duplicate_word",
                    "word": word,
                    "indexes": indexes,
                    "message": "duplicate word in accepted/edited items",
                }
            )

    if report["errors"]:
        report["status"] = "failed"
        report["written_count"] = 0
        return None, report, None

    report["status"] = "success"
    report["written_count"] = len(entries)
    return entries, report, output_path


def run_finalize_dictionary(
    source_path: Path,
    work_dir: Path,
    report_path: Path,
) -> dict:
    try:
        review_data = load_dictionary_review(source_path)
    except FileNotFoundError:
        report = _base_report(source_path, work_dir)
        report["status"] = "failed"
        report["errors"].append(
            {
                "code": "missing_dictionary_review",
                "message": f"{workspace_paths.display_path(source_path, work_dir)} does not exist",
            }
        )
        write_json(report_path, report)
        return report
    except json.JSONDecodeError as error:
        report = _base_report(source_path, work_dir)
        report["status"] = "failed"
        report["errors"].append(
            {
                "code": "invalid_json",
                "message": f"dictionary_review JSON parse failed: {error.msg}",
            }
        )
        write_json(report_path, report)
        return report

    entries, report, output_path = finalize_work_dictionary(review_data, source_path, work_dir)
    if report["status"] == "success" and entries is not None and output_path is not None:
        write_json(output_path, entries)
        if (work_dir / workspace_paths.WORK_META_FILENAME).exists():
            workspace_paths.update_work_meta_status(work_dir, "dictionary_finalized")
    write_json(report_path, report)
    return report


def _base_report(source_path: Path, work_dir: Path) -> dict:
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "input_stem": "",
        "source": workspace_paths.display_path(source_path, work_dir),
        "output": "",
        "status": "failed",
        "accepted_count": 0,
        "edited_count": 0,
        "ignored_count": 0,
        "pending_count": 0,
        "written_count": 0,
        "warnings": [],
        "errors": [],
    }


def _work_dictionary_path(input_stem: object, work_dir: Path) -> Path | None:
    if not isinstance(input_stem, str) or input_stem.strip() == "" or input_stem == "unknown":
        return None
    return workspace_paths.work_dictionary_path(work_dir)


def _text_value(value: object) -> str:
    if value is None:
        return ""
    return str(value).strip()
