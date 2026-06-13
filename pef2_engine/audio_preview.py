from __future__ import annotations

import re

from pef2_engine.dictionary_loader import build_text_ssml, find_dictionary_matches
from pef2_engine.ruby import strip_aozora_ruby
from pef2_engine.voicevox_alias import build_voicevox_alias
from version import VERSION


CHOKING_THRESHOLD = 20
TE_DE_THRESHOLD = 15
HEAVY_THRESHOLD = 12
DISTANCE_THRESHOLD = 6
STOP_MARKS = ("、", "　", "。", "！", "？", "「", "『", "」", "』")
STOP_PATTERN = r"[、。！？」』]"
SUFFIX_FRAGMENTS = (
    "よっ",
    "よる",
    "より",
    "よれ",
    "関し",
    "対し",
    "基づ",
    "通じ",
    "おいて",
    "あたっ",
    "とって",
    "とっ",
)
END_PROTECTION_VERBS = ("ある", "あった", "いる", "いた", "する", "した", "なる", "なった")
_BUDOUX_PARSER = None


class BudouxUnavailableError(RuntimeError):
    pass


def build_audio_preview(
    processed_data: object,
    dictionary_entries: list[dict],
    breath_rules: dict,
    breath_settings: dict | None = None,
) -> dict:
    segments = _segments(processed_data)
    preview_segments = [
        build_audio_preview_for_segment(segment, dictionary_entries, breath_rules, breath_settings)
        for segment in segments
    ]
    return {
        "schema_version": "pef2-audio-preview-0.1",
        "generator_version": VERSION,
        "segments": preview_segments,
    }


def build_audio_preview_for_segment(
    segment: dict,
    dictionary_entries: list[dict],
    breath_rules: dict,
    breath_settings: dict | None = None,
) -> dict:
    if is_image_segment(segment):
        return build_image_audio_segment(segment)

    display_text = strip_aozora_ruby(_display_text(segment))
    is_chapter = bool(segment.get("is_chapter") or segment.get("block_type") == "title")
    audio_fields = build_audio_fields_for_text(
        display_text,
        dictionary_entries,
        breath_rules,
        is_chapter,
        breath_settings,
    )

    return {
        "index": segment.get("index"),
        "segment_id": segment.get("segment_id", ""),
        "block_type": segment.get("block_type", "paragraph"),
        "display": display_text,
        **audio_fields,
    }


def build_audio_fields_for_text(
    display_text: str,
    dictionary_entries: list[dict],
    breath_rules: dict,
    is_chapter: bool = False,
    breath_settings: dict | None = None,
) -> dict:
    matches = find_dictionary_matches(display_text, dictionary_entries)
    text_for_breath, breath_marks = apply_machine_breath(
        display_text,
        breath_rules,
        is_chapter,
        breath_settings=breath_settings,
    )
    remapped_matches = find_dictionary_matches(text_for_breath, dictionary_entries)
    text_ssml, substitutions = build_text_ssml(text_for_breath, remapped_matches)
    text_plain_for_tts, replacements = build_text_plain_for_tts(text_for_breath, remapped_matches)

    return {
        "audio": text_ssml,
        "text_ssml": text_ssml,
        "text_plain_for_tts": text_plain_for_tts,
        "breath_marks": breath_marks,
        "dictionary_applied": _dictionary_applied(substitutions, replacements),
        "source_dictionary_matches": matches,
    }


def build_image_audio_segment(segment: dict) -> dict:
    return {
        "index": segment.get("index"),
        "segment_id": segment.get("segment_id", ""),
        "block_type": "image",
        "display": "",
        "audio": "",
        "text_ssml": "",
        "text_plain_for_tts": "",
        "is_image": True,
        "image_file": segment.get("image_file", ""),
        "audio_policy": segment.get("audio_policy", "pause_only"),
        "pause_type": segment.get("pause_type", "M-PAUSE"),
        "sync": {
            "include_in_audio_timeline": bool(
                segment.get("sync", {}).get("include_in_audio_timeline", True)
            ),
            "include_in_highlight": bool(
                segment.get("sync", {}).get("include_in_highlight", True)
            ),
        },
        "breath_marks": [],
        "dictionary_applied": [],
        "source_dictionary_matches": [],
    }


def is_image_segment(segment: dict) -> bool:
    return bool(segment.get("is_image")) or segment.get("block_type") == "image"


def build_text_plain_for_tts(text: str, matches: list[dict]) -> tuple[str, list[dict]]:
    text_plain = text
    replacements: list[dict] = []
    for match in sorted(matches, key=lambda item: item.get("match_start", -1), reverse=True):
        start = match.get("match_start")
        end = match.get("match_end")
        word = match.get("word", "")
        reading = match.get("reading", "")
        if not _valid_match(text, word, start, end) or not reading:
            continue
        alias = build_voicevox_alias(match, reading)
        text_plain = text_plain[:start] + alias + text_plain[end:]
        replacements.append({**match, "replacement": alias})
    replacements.sort(key=lambda item: item["match_start"])
    return text_plain, replacements


def apply_machine_breath(
    text: str,
    breath_rules: dict,
    is_chapter: bool = False,
    breath_settings: dict | None = None,
) -> tuple[str, list[dict]]:
    if not text:
        return text, []
    if is_chapter:
        return text, []

    resolved_breath_settings = _resolve_breath_settings(breath_settings)
    choking_threshold = resolved_breath_settings["choking_threshold"]
    distance_threshold = resolved_breath_settings["distance_threshold"]
    chunks = _parse_chunks(text)
    processed_chunks: list[str] = []
    breath_marks: list[dict] = []
    chars_since_pause = 0

    for index, chunk in enumerate(chunks):
        if any(mark in chunk for mark in STOP_MARKS):
            processed_chunks.append(chunk)
            _append_existing_comma_marks(chunk, breath_marks)
            chars_since_pause = 0
            continue

        add_comma = False
        reason = ""
        chars_since_pause += len(chunk)

        if chunk in breath_rules.get("conjunctions", []):
            add_comma = True
            reason = "Conj"
        else:
            is_conj_part = any(
                chunk.endswith(p) for p in breath_rules.get("conjunctive_particles", ["て", "で"])
            )
            is_heavy = any(chunk.endswith(p) for p in breath_rules.get("heavy_particles", []))
            is_light = any(chunk.endswith(p) for p in breath_rules.get("light_particles", []))
            want_to_pause = False
            if is_conj_part and chars_since_pause >= TE_DE_THRESHOLD:
                want_to_pause = True
                reason = "Te-Form"
            elif is_heavy and chars_since_pause >= HEAVY_THRESHOLD:
                want_to_pause = True
                reason = "Heavy"
            elif is_light and chars_since_pause >= choking_threshold:
                want_to_pause = True
                reason = "Light"

            if want_to_pause and index + 1 < len(chunks):
                next_chunk = chunks[index + 1]
                is_suffix = next_chunk.startswith(
                    tuple(breath_rules.get("functional_suffixes", []))
                ) or next_chunk.startswith(SUFFIX_FRAGMENTS)
                is_verb_protected = any(next_chunk.startswith(v) for v in END_PROTECTION_VERBS)
                if not is_suffix and not is_verb_protected and _remaining_len_to_stop(chunks, index + 1) > distance_threshold:
                    add_comma = True

        if add_comma:
            processed_chunks.append(chunk + "、")
            breath_marks.append(
                {
                    "type": "machine_inserted",
                    "source": "machine",
                    "after": chunk,
                    "reason": reason,
                    "chars_since_pause": chars_since_pause,
                    "editable": True,
                }
            )
            chars_since_pause = 0
        else:
            processed_chunks.append(chunk)

    return "".join(processed_chunks).replace("、、", "、"), breath_marks


def _resolve_breath_settings(settings: dict | None) -> dict:
    choking_threshold = CHOKING_THRESHOLD
    distance_threshold = DISTANCE_THRESHOLD
    if isinstance(settings, dict):
        candidate_choking = settings.get("choking_threshold")
        if isinstance(candidate_choking, int) and not isinstance(candidate_choking, bool) and candidate_choking > 0:
            choking_threshold = candidate_choking
        candidate_distance = settings.get("distance_threshold")
        if isinstance(candidate_distance, int) and not isinstance(candidate_distance, bool) and candidate_distance >= 0:
            distance_threshold = candidate_distance
    return {
        "choking_threshold": choking_threshold,
        "distance_threshold": distance_threshold,
    }


def _segments(processed_data: object) -> list[dict]:
    if isinstance(processed_data, dict):
        value = processed_data.get("segments", processed_data.get("remastered_data", []))
        return _non_empty_segments(value) if isinstance(value, list) else []
    if isinstance(processed_data, list):
        return _non_empty_segments(processed_data)
    return []


def _non_empty_segments(value: list) -> list[dict]:
    segments: list[dict] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        if is_image_segment(item) or _display_text(item):
            segments.append(item)
    return segments


def _display_text(segment: dict) -> str:
    display = segment.get("display", "")
    if isinstance(display, dict):
        return str(display.get("text") or "")
    return str(display or segment.get("text_processed") or segment.get("text_raw") or "")


def _parse_chunks(text: str) -> list[str]:
    global _BUDOUX_PARSER
    try:
        import budoux
    except ImportError as error:
        raise BudouxUnavailableError(
            "BudouX is required for automatic breath insertion; install budoux from requirements.txt"
        ) from error
    if _BUDOUX_PARSER is None:
        _BUDOUX_PARSER = budoux.load_default_japanese_parser()
    return _BUDOUX_PARSER.parse(text)


def _fallback_chunks(text: str) -> list[str]:
    parts = re.split(r"(、|。|！|？|「|」|『|』|\s+)", text)
    chunks: list[str] = []
    for part in parts:
        if not part:
            continue
        if chunks and part in STOP_MARKS:
            chunks[-1] += part
        else:
            chunks.append(part)
    return chunks


def _append_existing_comma_marks(chunk: str, breath_marks: list[dict]) -> None:
    if "、" not in chunk:
        return
    before = chunk.split("、", 1)[0]
    breath_marks.append(
        {
            "type": "existing_comma",
            "source": "original",
            "after": before,
            "editable": True,
        }
    )


def _remaining_len_to_stop(chunks: list[str], start_index: int) -> int:
    remaining_len = 0
    for chunk in chunks[start_index:]:
        match = re.search(STOP_PATTERN, chunk)
        if match:
            remaining_len += len(chunk[: match.start()])
            break
        remaining_len += len(chunk)
    return remaining_len


def _valid_match(text: str, word: str, start: object, end: object) -> bool:
    if not isinstance(start, int) or not isinstance(end, int):
        return False
    return 0 <= start < end <= len(text) and text[start:end] == word


def _dictionary_applied(substitutions: list[dict], replacements: list[dict]) -> list[dict]:
    plain_by_word = {item.get("word"): item.get("replacement") for item in replacements}
    applied: list[dict] = []
    for item in substitutions:
        word = item.get("word", "")
        applied.append(
            {
                "surface": word,
                "reading": item.get("reading", ""),
                "plain_reading": plain_by_word.get(word, item.get("reading", "")),
                "source": item.get("source", "dictionary"),
                "method": "sub_alias",
                "confirmed": True,
            }
        )
    return applied
