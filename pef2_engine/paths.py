from __future__ import annotations

import unicodedata
from pathlib import Path

from version import VERSION


DEFAULT_INPUT_DIR = "input"
DEFAULT_OUTPUT_DIR = "output"
DEFAULT_DICTIONARIES_DIR = "dictionaries"
DEFAULT_SYSTEM_DICT_DIR = "system"
DEFAULT_USER_DICT_DIR = "user"
DEFAULT_WORK_DICT_DIR = "work"

SYSTEM_DICT_FILENAME = "★システム固定辞書.json"
STANDARD_ENGLISH_DICT_FILENAME = "★標準英単語辞書.json"
USER_DICT_FILENAME = "★ユーザ辞書.json"
BREATH_RULES_FILENAME = "★汎用息継ぎ辞書.json"

PROCESSED_JSON_FILENAME = "processed.json"
WORK_DICTIONARY_DRAFT_FILENAME = "work_dictionary_draft.json"
DICTIONARY_REVIEW_FILENAME = "dictionary_review.json"
REVIEW_ITEMS_FILENAME = "review_items.json"
AUDIO_PREVIEW_FILENAME = "audio_preview.json"
PROMOTE_CANDIDATES_FILENAME = "promote_candidates.json"
PRE_PROCESSED_JSON_FILENAME = "pre_processed.json"


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def input_dir(root: Path | None = None) -> Path:
    return (root or project_root()) / DEFAULT_INPUT_DIR


def output_dir(root: Path | None = None) -> Path:
    return (root or project_root()) / DEFAULT_OUTPUT_DIR


def dictionaries_dir(root: Path | None = None) -> Path:
    return (root or project_root()) / DEFAULT_DICTIONARIES_DIR


def system_dict_dir(root: Path | None = None) -> Path:
    return dictionaries_dir(root) / DEFAULT_SYSTEM_DICT_DIR


def user_dict_dir(root: Path | None = None) -> Path:
    return dictionaries_dir(root) / DEFAULT_USER_DICT_DIR


def work_dict_dir(root: Path | None = None) -> Path:
    return dictionaries_dir(root) / DEFAULT_WORK_DICT_DIR


def default_processed_path(root: Path | None = None) -> Path:
    return output_dir(root) / PROCESSED_JSON_FILENAME


def default_pre_processed_path(root: Path | None = None) -> Path:
    return output_dir(root) / PRE_PROCESSED_JSON_FILENAME


def default_draft_path(root: Path | None = None) -> Path:
    return output_dir(root) / WORK_DICTIONARY_DRAFT_FILENAME


def default_dictionary_review_path(root: Path | None = None) -> Path:
    return output_dir(root) / DICTIONARY_REVIEW_FILENAME


def default_review_items_path(root: Path | None = None) -> Path:
    return output_dir(root) / REVIEW_ITEMS_FILENAME


def default_audio_preview_path(root: Path | None = None) -> Path:
    return output_dir(root) / AUDIO_PREVIEW_FILENAME


def default_promote_candidates_path(root: Path | None = None) -> Path:
    return output_dir(root) / PROMOTE_CANDIDATES_FILENAME


def work_dictionary_path(input_path: Path, root: Path | None = None) -> Path:
    return work_dict_dir(root) / f"{input_path.stem}-辞書.json"


def system_dictionary_path(root: Path | None = None) -> Path:
    return resolve_named_file(system_dict_dir(root), SYSTEM_DICT_FILENAME)


def standard_english_dictionary_path(root: Path | None = None) -> Path:
    return resolve_named_file(system_dict_dir(root), STANDARD_ENGLISH_DICT_FILENAME)


def user_dictionary_path(root: Path | None = None) -> Path:
    return resolve_named_file(user_dict_dir(root), USER_DICT_FILENAME)


def breath_rules_path(root: Path | None = None) -> Path:
    return resolve_named_file(system_dict_dir(root), BREATH_RULES_FILENAME)


def resolve_named_file(directory: Path, filename: str) -> Path:
    exact = directory / filename
    if exact.exists():
        return exact

    candidates = resolve_named_file_candidates(directory, filename)
    if candidates:
        return candidates[0]
    return exact


def resolve_named_file_candidates(directory: Path, filename: str) -> list[Path]:
    target_nfc = unicodedata.normalize("NFC", filename)
    target_nfd = unicodedata.normalize("NFD", filename)
    candidates: list[Path] = []
    for path in directory.iterdir() if directory.exists() else []:
        name_nfc = unicodedata.normalize("NFC", path.name)
        name_nfd = unicodedata.normalize("NFD", path.name)
        if name_nfc == target_nfc or name_nfd == target_nfd:
            candidates.append(path)
    return candidates
