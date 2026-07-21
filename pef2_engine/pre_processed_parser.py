from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path

from pef2_engine.processed_builder import (
    DEFAULT_IMAGE_AUDIO_POLICY,
    DEFAULT_IMAGE_PAUSE_TYPE,
    DEFAULT_INCLUDE_IMAGE_IN_AUDIO_TIMELINE,
    DEFAULT_INCLUDE_IMAGE_IN_HIGHLIGHT,
    parse_image_marker,
)
from version import VERSION


SCHEMA_VERSION = "pef2-pre-processed-0.1"
PURPOSE = "pef2_step6_pre_processed"
NLM_SIMPLE_SCHEMA_PREFIX = "pef2-nlm-simple"
LEGACY_TITLE_MAX_LENGTH = 100
HD_TITLE_DISPLAY_PATTERN = re.compile(r"^(?:[一二三四五六七八九十]|第[0-9０-９一二三四五六七八九十]+章)$")
LEGACY_KANJI_TITLE_PATTERN = re.compile(r"^[一二三四五六七八九十百壱弐参]+$")
LEGACY_NUMBERED_TITLE_PATTERN = re.compile(r"^(第)?[0-9１２３４５６７８９０一二三四五六七八九十百]+[章節回集巻].*$")
LEGACY_SENTENCE_SPLIT_PATTERN = re.compile(
    r"(?<=[。！？][」』）］\)])|(?<=[。！？])(?![」』）］\)])"
)
PLAIN_TEXT_AUDIO_SPACE_PATTERN = re.compile(r"[ \u3000]+")


def normalize_plain_text_audio_spaces(text: str) -> str:
    """Normalize U+0020/U+3000 separators for a new plain-text audio seed."""

    def replace_space_run(match: re.Match[str]) -> str:
        left = text[match.start() - 1] if match.start() > 0 else ""
        right = text[match.end()] if match.end() < len(text) else ""
        if _is_japanese_text_character(left) or _is_japanese_text_character(right):
            return ""
        return " "

    return PLAIN_TEXT_AUDIO_SPACE_PATTERN.sub(replace_space_run, text)


def _is_japanese_text_character(character: str) -> bool:
    if not character:
        return False
    codepoint = ord(character)
    # Kana ranges cover hiragana, full/half-width katakana, and kana extensions.
    if (
        0x3040 <= codepoint <= 0x30FF
        or 0x31F0 <= codepoint <= 0x31FF
        or 0x1AFF0 <= codepoint <= 0x1AFFF
        or 0x1B000 <= codepoint <= 0x1B16F
        or 0xFF66 <= codepoint <= 0xFF9F
    ):
        return True
    # U+3001..U+303F contains Japanese/CJK punctuation and iteration marks;
    # U+3000 itself is excluded because it is the separator being normalized.
    if 0x3001 <= codepoint <= 0x303F:
        return True
    if character in "！？（）［］｛｝，．：；…‥—―～":
        return True
    name = unicodedata.name(character, "")
    return name.startswith("CJK UNIFIED IDEOGRAPH") or name.startswith(
        "CJK COMPATIBILITY IDEOGRAPH"
    )


def parse_source_file_to_pre_processed(path: Path) -> dict:
    data = load_source_data(path)
    return parse_to_pre_processed(data, str(path))


def load_source_data(path: Path) -> object:
    text = path.read_text(encoding="utf-8")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        objects = extract_json_objects(text)
        if not objects:
            if path.suffix.lower() == ".txt":
                return text
            raise
        if all(is_nlm_simple(item) for item in objects):
            return {"fragments": objects}
        if len(objects) == 1:
            return objects[0]
        raise ValueError(f"cannot parse multiple JSON objects in {path}")


def extract_json_objects(text: str) -> list[object]:
    decoder = json.JSONDecoder()
    objects: list[object] = []
    position = 0
    while position < len(text):
        brace_position = text.find("{", position)
        if brace_position == -1:
            break
        try:
            item, end = decoder.raw_decode(text, brace_position)
        except json.JSONDecodeError:
            raise ValueError("cannot parse JSON object in source text")
        objects.append(item)
        position = end
    return objects


def parse_to_pre_processed(data: object, source_path: str = "") -> dict:
    source_format = detect_format(data)
    if source_format == "unknown":
        raise ValueError("unknown source format")
    raw_segments = import_segments(source_format, data)
    segments = normalize_to_segments(raw_segments)
    if not segments:
        raise ValueError("no segments parsed")
    warnings = validate_source_data(source_format, data)
    warnings.extend(validate_pre_processed_segments(segments))
    return {
        "schema_version": SCHEMA_VERSION,
        "generator_version": VERSION,
        "purpose": PURPOSE,
        "source": {
            "source_format": source_format,
            "source_path": source_path,
        },
        "validation": {
            "warnings": warnings,
            "warning_count": len(warnings),
        },
        "segments": segments,
    }


def detect_format(data: object) -> str:
    if isinstance(data, str):
        return "plain_text"
    if is_nlm_fragments(data):
        return "nlm_fragments"
    if is_nlm_simple(data):
        return "nlm_simple"
    if is_hd(data):
        return "hd"
    if is_pef_legacy(data):
        return "pef_legacy"
    return "unknown"


def import_segments(source_format: str, data: object) -> list[dict]:
    if source_format == "plain_text" and isinstance(data, str):
        return import_plain_text_segments(data)
    if source_format == "nlm_fragments" and isinstance(data, dict):
        segments: list[dict] = []
        for fragment in data.get("fragments", []):
            if is_nlm_simple(fragment):
                segments.extend(import_nlm_simple_segments(fragment))
        return segments
    if source_format == "nlm_simple" and isinstance(data, dict):
        return import_nlm_simple_segments(data)
    if source_format == "hd":
        return import_hd_segments(data)
    if source_format == "pef_legacy" and isinstance(data, dict):
        return import_pef_legacy_segments(data)
    return []


def is_nlm_fragments(data: object) -> bool:
    return isinstance(data, dict) and isinstance(data.get("fragments"), list)


def is_nlm_simple(data: object) -> bool:
    return isinstance(data, dict) and (
        str(data.get("schema_version", "")).startswith(NLM_SIMPLE_SCHEMA_PREFIX)
        or (isinstance(data.get("fragment"), dict) and isinstance(data.get("items"), list))
    )


def is_hd(data: object) -> bool:
    return any("lower" in segment for segment in remastered_segments(data))


def is_pef_legacy(data: object) -> bool:
    return isinstance(data, dict) and isinstance(data.get("remastered_data"), list)


def import_nlm_simple_segments(data: dict) -> list[dict]:
    items = data.get("items", [])
    if not isinstance(items, list):
        return []
    return [normalize_nlm_item(item) for item in items if isinstance(item, dict)]


def import_plain_text_segments(text: str) -> list[dict]:
    normalized_text = text.replace("\r\n", "\n").replace("\r", "\n")
    raw_lines = normalized_text.split("\n")
    segments: list[dict] = []
    in_paragraph = False

    for line_number, raw_line in enumerate(raw_lines):
        display = raw_line.strip()
        if not display:
            in_paragraph = False
            continue

        image_file = parse_image_marker(display)
        if image_file:
            raw_segment = image_raw_segment(line_number, image_file)
            raw_segment["para_start"] = not in_paragraph
            raw_segment["line_start"] = True
            segments.append(raw_segment)
            in_paragraph = True
            continue

        title_display = legacy_plain_text_title_display(display)
        if title_display is not None:
            segments.append(
                {
                    "source_index": line_number,
                    "block_type": "title",
                    "display": title_display,
                    "audio_seed": normalize_plain_text_audio_spaces(title_display),
                    "lower_display": "",
                    "is_image": False,
                    "image_file": "",
                    "notes": [],
                    "para_start": not in_paragraph,
                    "line_start": True,
                }
            )
            in_paragraph = False
            continue

        sentence_segments = split_plain_text_line_to_segments(
            display,
            source_index=line_number,
            para_start=not in_paragraph,
        )
        if sentence_segments:
            segments.extend(sentence_segments)
            in_paragraph = True

    return segments


def legacy_plain_text_title_display(display: str) -> str | None:
    text = display.strip()
    if not text or "。" in text or len(text) >= LEGACY_TITLE_MAX_LENGTH:
        return None
    if text.startswith("# "):
        return text[2:].strip()
    if LEGACY_KANJI_TITLE_PATTERN.fullmatch(text) and len(text) < 10:
        return text
    if LEGACY_NUMBERED_TITLE_PATTERN.fullmatch(text):
        return text
    return None


def split_plain_text_line_to_segments(line: str, *, source_index: int, para_start: bool) -> list[dict]:
    sentences = [
        sentence.strip()
        for sentence in LEGACY_SENTENCE_SPLIT_PATTERN.split(line)
        if sentence.strip()
    ]
    if not sentences:
        return []

    segments: list[dict] = []
    for position, sentence in enumerate(sentences):
        segments.append(
            {
                "source_index": source_index,
                "block_type": "paragraph",
                "display": sentence,
                "audio_seed": normalize_plain_text_audio_spaces(sentence),
                "lower_display": "",
                "is_image": False,
                "image_file": "",
                "notes": [],
                "para_start": para_start and position == 0,
                "line_start": position == 0,
            }
        )
    return segments


def normalize_nlm_item(item: dict) -> dict:
    item_type = str(item.get("type") or "text").strip()
    display = str(item.get("display") or "").strip()
    audio_seed = str(item.get("audio") or "").strip()
    lower_display = str(item.get("lower_display") or "").strip()
    notes = str(item.get("notes") or "").strip()

    if item_type == "image":
        image_file = parse_image_marker(display) or parse_image_marker(audio_seed)
        if not image_file:
            image_file = parse_image_marker(notes)
        if not image_file:
            image_file = display or audio_seed
        return image_raw_segment(item.get("index"), image_file, notes)

    return {
        "source_index": item.get("index"),
        "block_type": "title" if item_type == "title" else "paragraph",
        "display": display,
        "audio_seed": audio_seed,
        "lower_display": lower_display,
        "is_image": False,
        "image_file": "",
        "notes": [notes] if notes else [],
    }


def import_hd_segments(data: object) -> list[dict]:
    return [normalize_hd_segment(segment) for segment in remastered_segments(data)]


def normalize_hd_segment(segment: dict) -> dict:
    lower = segment.get("lower") if isinstance(segment.get("lower"), dict) else {}
    is_image = bool(segment.get("is_image")) or segment.get("block_type") == "image"
    display = str(lower.get("display") or "")
    if is_image:
        return image_raw_segment(segment.get("index"), str(segment.get("image_file") or ""))
    return {
        "source_index": segment.get("index"),
        "block_type": "title" if is_hd_title_segment(segment, display) else "paragraph",
        "display": display,
        "audio_seed": str(lower.get("audio") or ""),
        "lower_display": display,
        "is_image": False,
        "image_file": "",
        "notes": [],
    }


def is_hd_title_segment(segment: dict, display: str) -> bool:
    if segment.get("is_chapter"):
        return True
    return bool(HD_TITLE_DISPLAY_PATTERN.fullmatch(display.strip()))


def import_pef_legacy_segments(data: dict) -> list[dict]:
    return [
        normalize_pef_legacy_segment(segment)
        for segment in remastered_segments(data)
    ]


def normalize_pef_legacy_segment(segment: dict) -> dict:
    is_image = bool(segment.get("is_image"))
    if is_image:
        raw_segment = image_raw_segment(segment.get("index"), str(segment.get("image_file") or ""))
        copy_structure_flags(raw_segment, segment)
        return raw_segment
    raw_segment = {
        "source_index": segment.get("index"),
        "block_type": "title" if segment.get("is_chapter") else "paragraph",
        "display": str(segment.get("display") or ""),
        "audio_seed": str(segment.get("audio") or ""),
        "lower_display": "",
        "is_image": False,
        "image_file": "",
        "is_chapter": bool(segment.get("is_chapter", False)),
        "notes": [],
    }
    copy_structure_flags(raw_segment, segment)
    return raw_segment


def copy_structure_flags(target: dict, source: dict) -> None:
    for key in ("para_start", "line_start"):
        if key in source:
            target[key] = bool(source.get(key))


def image_raw_segment(source_index: object, image_file: str, notes: str = "") -> dict:
    return {
        "source_index": source_index,
        "block_type": "image",
        "display": "",
        "audio_seed": "",
        "lower_display": "",
        "is_image": True,
        "image_file": image_file,
        "audio_policy": DEFAULT_IMAGE_AUDIO_POLICY,
        "pause_type": DEFAULT_IMAGE_PAUSE_TYPE,
        "sync": {
            "include_in_audio_timeline": DEFAULT_INCLUDE_IMAGE_IN_AUDIO_TIMELINE,
            "include_in_highlight": DEFAULT_INCLUDE_IMAGE_IN_HIGHLIGHT,
        },
        "notes": [notes] if notes else [],
    }


def remastered_segments(data: object) -> list[dict]:
    if isinstance(data, dict):
        value = data.get("remastered_data", [])
        return [item for item in value if isinstance(item, dict)] if isinstance(value, list) else []
    return []


def normalize_to_segments(raw_segments: list[dict]) -> list[dict]:
    segments: list[dict] = []
    for index, raw_segment in enumerate(raw_segments):
        is_image = bool(raw_segment.get("is_image")) or raw_segment.get("block_type") == "image"
        segment = {
            "index": index,
            "source_index": raw_segment.get("source_index"),
            "segment_id": f"seg-{index:04d}",
            "block_type": "image" if is_image else normalize_block_type(raw_segment.get("block_type")),
            "display": "" if is_image else str(raw_segment.get("display") or ""),
            "audio_seed": "" if is_image else str(raw_segment.get("audio_seed") or ""),
            "lower_display": "" if is_image else str(raw_segment.get("lower_display") or ""),
            "is_image": is_image,
            "image_file": str(raw_segment.get("image_file") or ""),
            "audio_policy": raw_segment.get("audio_policy", ""),
            "pause_type": raw_segment.get("pause_type", ""),
            "sync": raw_segment.get("sync", {}),
            "validation_warnings": [],
            "notes": raw_segment.get("notes", []),
        }
        copy_structure_flags(segment, raw_segment)
        if is_image:
            apply_image_defaults(segment)
        segments.append(segment)
    return segments


def validate_pre_processed(pre_processed: dict) -> list[str]:
    return validate_pre_processed_segments(pre_processed.get("segments", []))


def validate_source_data(source_format: str, data: object) -> list[str]:
    if source_format == "nlm_simple" and isinstance(data, dict):
        return validate_nlm_fragment(data)
    if source_format == "nlm_fragments" and isinstance(data, dict):
        warnings: list[str] = []
        for fragment_index, fragment in enumerate(data.get("fragments", [])):
            if isinstance(fragment, dict):
                for warning in validate_nlm_fragment(fragment):
                    warnings.append(f"fragment {fragment_index}: {warning}")
        return warnings
    return []


def validate_nlm_fragment(fragment: dict) -> list[str]:
    warnings: list[str] = []
    fragment_meta = fragment.get("fragment") if isinstance(fragment.get("fragment"), dict) else {}
    items = fragment.get("items", [])
    if not isinstance(items, list):
        return ["items is not a list"]
    if not items:
        return ["items is empty"]

    indexes: list[int] = []
    for position, item in enumerate(items):
        if not isinstance(item, dict):
            warnings.append(f"item at position {position} is not an object")
            continue
        index = item.get("index")
        if not isinstance(index, int):
            warnings.append(f"item at position {position} has no integer index")
            continue
        indexes.append(index)

    if indexes and isinstance(fragment_meta.get("start_index"), int):
        if min(indexes) != fragment_meta["start_index"]:
            warnings.append(
                f"fragment start_index mismatch: expected {fragment_meta['start_index']}, actual {min(indexes)}"
            )
    if indexes and isinstance(fragment_meta.get("end_index"), int):
        if max(indexes) != fragment_meta["end_index"]:
            warnings.append(
                f"fragment end_index mismatch: expected {fragment_meta['end_index']}, actual {max(indexes)}"
            )
    return warnings


def validate_pre_processed_segments(segments: list[dict]) -> list[str]:
    warnings: list[str] = []
    seen_indexes: set[int] = set()
    seen_source_indexes: set[int] = set()
    for position, segment in enumerate(segments):
        index = segment.get("index")
        if not isinstance(index, int):
            warnings.append(f"segment at position {position} has non-integer index")
        elif index in seen_indexes:
            warnings.append(f"duplicate index: {index}")
        else:
            seen_indexes.add(index)

        source_index = segment.get("source_index")
        if source_index is None:
            warnings.append(f"segment index {index} has no source_index")
        elif isinstance(source_index, int):
            if source_index in seen_source_indexes:
                warnings.append(f"duplicate source_index: {source_index}")
            seen_source_indexes.add(source_index)
        else:
            warnings.append(f"segment index {index} has non-integer source_index")

        if segment.get("block_type") == "image":
            if not segment.get("image_file"):
                warnings.append(f"image segment index {index} has no image_file")
        else:
            if not segment.get("display") and not segment.get("audio_seed"):
                warnings.append(f"text segment index {index} has no display/audio_seed")
            elif not segment.get("display"):
                warnings.append(f"text segment index {index} display is empty")
            elif not segment.get("audio_seed"):
                warnings.append(f"text segment index {index} audio_seed is empty")

        audio_seed = str(segment.get("audio_seed") or "")
        if "[M-PAUSE]" in audio_seed or "[L-PAUSE]" in audio_seed:
            warnings.append(f"segment index {index} audio_seed contains legacy pause marker")
        if "<sub " in audio_seed or "</sub>" in audio_seed:
            warnings.append(f"segment index {index} audio_seed contains legacy SSML sub tag")

    expected = list(range(len(segments)))
    actual = [segment.get("index") for segment in segments]
    if actual != expected:
        warnings.append("normalized indexes are not contiguous from 0")
    return warnings


def normalize_block_type(value: object) -> str:
    text = str(value or "paragraph").strip()
    if text in {"title", "paragraph", "image"}:
        return text
    if text == "text":
        return "paragraph"
    return "paragraph"


def apply_image_defaults(segment: dict) -> None:
    segment["display"] = ""
    segment["audio_seed"] = ""
    segment["lower_display"] = ""
    segment["audio_policy"] = segment.get("audio_policy") or DEFAULT_IMAGE_AUDIO_POLICY
    segment["pause_type"] = segment.get("pause_type") or DEFAULT_IMAGE_PAUSE_TYPE
    sync = segment.get("sync") if isinstance(segment.get("sync"), dict) else {}
    segment["sync"] = {
        "include_in_audio_timeline": bool(
            sync.get("include_in_audio_timeline", DEFAULT_INCLUDE_IMAGE_IN_AUDIO_TIMELINE)
        ),
        "include_in_highlight": bool(
            sync.get("include_in_highlight", DEFAULT_INCLUDE_IMAGE_IN_HIGHLIGHT)
        ),
    }
