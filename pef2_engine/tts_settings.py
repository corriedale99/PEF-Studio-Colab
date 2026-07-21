from __future__ import annotations

from copy import deepcopy
import math
from pathlib import Path
from typing import Any

from pef2_engine.io_utils import read_json, write_json


WORKSPACE_SETTINGS_FILENAME = "settings.json"
WORK_TTS_SETTINGS_FILENAME = "tts_settings.json"
WORKSPACE_SETTINGS_SCHEMA_VERSION = "workspace_settings-1"
WORK_TTS_SETTINGS_SCHEMA_VERSION = "tts_settings-1"
DEFAULT_SPEED_SCALE = 1.0
MIN_SPEED_SCALE = 0.5
MAX_SPEED_SCALE = 2.0

DEFAULT_TTS_SETTINGS = {
    "voice": {
        "backend": "VOICEVOX",
        "speaker_id": 8,
        "speed_scale": DEFAULT_SPEED_SCALE,
    },
    "breath": {
        "choking_threshold": 20,
        "distance_threshold": 6,
    },
}


def workspace_settings_path(workspace_root: Path) -> Path:
    return Path(workspace_root) / WORKSPACE_SETTINGS_FILENAME


def work_tts_settings_path(work_dir: Path) -> Path:
    return Path(work_dir) / WORK_TTS_SETTINGS_FILENAME


def read_workspace_settings(workspace_root: Path) -> dict:
    return _read_optional_object(workspace_settings_path(workspace_root))


def read_work_tts_settings(work_dir: Path) -> dict:
    return _read_optional_object(work_tts_settings_path(work_dir))


def resolve_tts_settings(workspace_root: Path, work_dir: Path | None = None) -> dict:
    resolved = default_tts_settings()
    workspace_settings = read_workspace_settings(workspace_root)
    _merge_tts_settings(resolved, _workspace_tts_section(workspace_settings))
    if work_dir is not None:
        _merge_tts_settings(resolved, read_work_tts_settings(work_dir))
    return resolved


def resolve_tts_settings_with_sources(workspace_root: Path, work_dir: Path | None = None) -> dict:
    resolved = default_tts_settings()
    sources = {
        "voice": {
            "backend": "default",
            "speaker_id": "default",
            "speed_scale": "default",
        },
        "breath": {"choking_threshold": "default", "distance_threshold": "default"},
    }
    workspace_settings = read_workspace_settings(workspace_root)
    _merge_tts_settings_with_sources(
        resolved,
        sources,
        _workspace_tts_section(workspace_settings),
        "workspace",
    )
    if work_dir is not None:
        _merge_tts_settings_with_sources(
            resolved,
            sources,
            read_work_tts_settings(work_dir),
            "work",
        )
    return {
        "voice": {
            "backend": resolved["voice"]["backend"],
            "speaker_id": resolved["voice"]["speaker_id"],
            "speed_scale": resolved["voice"]["speed_scale"],
            "source": _highest_priority_source(sources["voice"].values()),
        },
        "breath": {
            "choking_threshold": resolved["breath"]["choking_threshold"],
            "distance_threshold": resolved["breath"]["distance_threshold"],
            "source": _highest_priority_source(sources["breath"].values()),
        },
    }


def build_workspace_settings(settings: dict | None = None) -> dict:
    return {
        "schema_version": WORKSPACE_SETTINGS_SCHEMA_VERSION,
        "tts": _normalize_tts_settings(settings or {}),
    }


def build_work_tts_settings(settings: dict | None = None) -> dict:
    source = deepcopy(settings) if isinstance(settings, dict) else {}
    source.pop("schema_version", None)
    normalized = _normalize_tts_settings(source, include_speed_default=False)
    return {
        "schema_version": WORK_TTS_SETTINGS_SCHEMA_VERSION,
        **normalized,
    }


def write_workspace_settings(workspace_root: Path, settings: dict | None = None) -> dict:
    data = build_workspace_settings(settings)
    write_json(workspace_settings_path(workspace_root), data)
    return data


def write_work_tts_settings(work_dir: Path, settings: dict | None = None) -> dict:
    data = build_work_tts_settings(settings)
    write_json(work_tts_settings_path(work_dir), data)
    return data


def default_tts_settings() -> dict:
    return deepcopy(DEFAULT_TTS_SETTINGS)


def normalize_speed_scale(value: Any) -> float | None:
    return _valid_speed_scale(value)


def effective_tts_settings_paths(workspace_root: Path, work_dir: Path) -> tuple[Path, ...]:
    workspace_path = workspace_settings_path(workspace_root)
    work_path = work_tts_settings_path(work_dir)
    if not work_path.is_file():
        return (workspace_path,) if workspace_path.is_file() else ()

    paths = [work_path]
    if workspace_path.is_file() and _work_settings_use_workspace_fallback(
        read_work_tts_settings(work_dir)
    ):
        paths.append(workspace_path)
    return tuple(paths)


def _read_optional_object(path: Path) -> dict:
    try:
        data = read_json(path, default={})
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def _workspace_tts_section(settings: dict) -> dict:
    tts = settings.get("tts")
    return tts if isinstance(tts, dict) else {}


def _normalize_tts_settings(
    settings: dict,
    *,
    include_speed_default: bool = True,
) -> dict:
    normalized = deepcopy(settings) if isinstance(settings, dict) else {}
    voice = normalized.get("voice") if isinstance(normalized.get("voice"), dict) else {}
    backend = _valid_backend(voice.get("backend")) or DEFAULT_TTS_SETTINGS["voice"]["backend"]
    speaker_id = _valid_non_negative_int(voice.get("speaker_id"))
    if speaker_id is None:
        speaker_id = DEFAULT_TTS_SETTINGS["voice"]["speaker_id"]
    speed_scale = _valid_speed_scale(voice.get("speed_scale"))
    voice["backend"] = backend
    voice["speaker_id"] = speaker_id
    if speed_scale is not None:
        voice["speed_scale"] = speed_scale
    elif include_speed_default:
        voice["speed_scale"] = DEFAULT_SPEED_SCALE
    else:
        voice.pop("speed_scale", None)
    normalized["voice"] = voice

    breath = normalized.get("breath") if isinstance(normalized.get("breath"), dict) else {}
    choking_threshold = _valid_positive_int(breath.get("choking_threshold"))
    if choking_threshold is None:
        choking_threshold = DEFAULT_TTS_SETTINGS["breath"]["choking_threshold"]
    distance_threshold = _valid_non_negative_int(breath.get("distance_threshold"))
    if distance_threshold is None:
        distance_threshold = DEFAULT_TTS_SETTINGS["breath"]["distance_threshold"]
    breath["choking_threshold"] = choking_threshold
    breath["distance_threshold"] = distance_threshold
    normalized["breath"] = breath
    return normalized


def _merge_tts_settings(target: dict, source: dict) -> None:
    if not isinstance(source, dict):
        return
    voice = source.get("voice")
    if isinstance(voice, dict):
        backend = _valid_backend(voice.get("backend"))
        if backend is not None:
            target["voice"]["backend"] = backend
        speaker_id = _valid_non_negative_int(voice.get("speaker_id"))
        if speaker_id is not None:
            target["voice"]["speaker_id"] = speaker_id
        speed_scale = _valid_speed_scale(voice.get("speed_scale"))
        if speed_scale is not None:
            target["voice"]["speed_scale"] = speed_scale
    breath = source.get("breath")
    if isinstance(breath, dict):
        choking_threshold = _valid_positive_int(breath.get("choking_threshold"))
        if choking_threshold is not None:
            target["breath"]["choking_threshold"] = choking_threshold
        distance_threshold = _valid_non_negative_int(breath.get("distance_threshold"))
        if distance_threshold is not None:
            target["breath"]["distance_threshold"] = distance_threshold


def _merge_tts_settings_with_sources(
    target: dict,
    sources: dict,
    source: dict,
    source_label: str,
) -> None:
    if not isinstance(source, dict):
        return
    voice = source.get("voice")
    if isinstance(voice, dict):
        backend = _valid_backend(voice.get("backend"))
        if backend is not None:
            target["voice"]["backend"] = backend
            sources["voice"]["backend"] = source_label
        speaker_id = _valid_non_negative_int(voice.get("speaker_id"))
        if speaker_id is not None:
            target["voice"]["speaker_id"] = speaker_id
            sources["voice"]["speaker_id"] = source_label
        speed_scale = _valid_speed_scale(voice.get("speed_scale"))
        if speed_scale is not None:
            target["voice"]["speed_scale"] = speed_scale
            sources["voice"]["speed_scale"] = source_label
    breath = source.get("breath")
    if isinstance(breath, dict):
        choking_threshold = _valid_positive_int(breath.get("choking_threshold"))
        if choking_threshold is not None:
            target["breath"]["choking_threshold"] = choking_threshold
            sources["breath"]["choking_threshold"] = source_label
        distance_threshold = _valid_non_negative_int(breath.get("distance_threshold"))
        if distance_threshold is not None:
            target["breath"]["distance_threshold"] = distance_threshold
            sources["breath"]["distance_threshold"] = source_label


def _highest_priority_source(values) -> str:
    priority = {"default": 0, "workspace": 1, "work": 2}
    return max((str(value) for value in values), key=lambda value: priority.get(value, -1))


def _valid_backend(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip().upper()
    return normalized if normalized in {"VOICEVOX", "GCS"} else None


def _valid_speed_scale(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return None
    if not isinstance(value, (int, float, str)):
        return None
    try:
        number = float(value)
    except (OverflowError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number if MIN_SPEED_SCALE <= number <= MAX_SPEED_SCALE else None


def _work_settings_use_workspace_fallback(settings: dict) -> bool:
    voice = settings.get("voice") if isinstance(settings.get("voice"), dict) else {}
    breath = settings.get("breath") if isinstance(settings.get("breath"), dict) else {}
    return any(
        value is None
        for value in (
            _valid_backend(voice.get("backend")),
            _valid_non_negative_int(voice.get("speaker_id")),
            _valid_speed_scale(voice.get("speed_scale")),
            _valid_positive_int(breath.get("choking_threshold")),
            _valid_non_negative_int(breath.get("distance_threshold")),
        )
    )


def _valid_positive_int(value: Any) -> int | None:
    number = _valid_int(value)
    if number is None:
        return None
    return number if number > 0 else None


def _valid_non_negative_int(value: Any) -> int | None:
    number = _valid_int(value)
    if number is None:
        return None
    return number if number >= 0 else None


def _valid_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    elif isinstance(value, str) and value.strip().isdigit():
        return int(value.strip())
    return None
