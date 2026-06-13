from __future__ import annotations

from pef2_engine.review import validate_decision
from version import VERSION


def build_work_dictionary(review_items: list[dict]) -> list[dict]:
    entries: list[dict] = []
    for item in review_items:
        decision = validate_decision(str(item.get("decision") or "pending"))
        if decision not in {"accept", "edit"}:
            continue
        reading = _accepted_reading(item, decision)
        if not item.get("term") or not reading:
            continue
        entry = {
            "単語原文": item["term"],
            "読み": reading,
            "generator_version": VERSION,
        }
        if item.get("meaning"):
            entry["意味"] = item["meaning"]
        if item.get("difficulty") != "":
            entry["難易度"] = item["difficulty"]
        for key in ("confidence", "source", "decision", "target_dictionary", "notes"):
            if item.get(key):
                entry[key] = item[key]
        if item.get("tts_alias"):
            entry["tts_alias"] = item["tts_alias"]
        if "voicevox_katakana" in item:
            entry["voicevox_katakana"] = bool(item["voicevox_katakana"])
        entries.append(entry)
    return entries


def build_promote_candidates(review_items: list[dict]) -> list[dict]:
    return [
        {**item, "promote_to_user_dictionary": True}
        for item in review_items
        if validate_decision(str(item.get("decision") or "")) == "promote"
    ]


def _accepted_reading(item: dict, decision: str) -> str:
    if decision == "edit":
        return str(item.get("edited_reading") or "").strip()
    return str(item.get("reading") or "").strip()
