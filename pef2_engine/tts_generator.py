from __future__ import annotations

import json
import math
import os
import re
import shutil
import subprocess
import time
import urllib.parse
import urllib.request
import wave
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Protocol

from pef2_engine import workspace_paths
from pef2_engine.io_utils import read_json, write_json
from pef2_engine.tts_settings import resolve_tts_settings, resolve_tts_settings_with_sources
from pef2_engine.tts_pre_transform import SILENCE, run_tts_pre_transform


TTS_BUILD_REPORT_SCHEMA_VERSION = "tts-build-report-1"
SYNC_MAP_SCHEMA_VERSION = "sync-map-1"
TTS_BACKEND_DEFAULT = "VOICEVOX"
AUDIO_BACKUP_KEEP = 2
SYNC_DURATION_TOLERANCE_SECONDS = 0.25
FFMPEG_BIN_ENV = "FFMPEG_BIN"
FFMPEG_BIN_DEFAULT = "ffmpeg"
AUDIO_FILENAME = "audio.mp3"
SYNC_MAP_FILENAME = "sync_map.json"
TTS_BUILD_REPORT_FILENAME = "tts_build_report.json"
CONCAT_WAV_FILENAME = "concat.wav"
VOICE_PREVIEW_TEXT = "読み上げの声を確認します。"
VOICE_PREVIEW_DIRNAME = "preview"
VOICE_PREVIEW_FILENAME = "voice_preview.mp3"
WORKSPACE_TEMP_DIRNAME = "_temp_"

TTS_RETRY = {
    "VOICEVOX": {
        "max_retries": 3,
        "retry_wait_seconds": 2.0,
        "audio_query_timeout": 30,
        "synthesis_timeout": 60,
    },
    "GCS": {
        "max_retries": 3,
        "retry_wait_seconds": 2.0,
        "synthesis_timeout": 60,
    },
}

VOICEVOX_CONFIG = {
    "url": "http://localhost:50021",
    "speaker_id": 13,
    "audio_query_timeout": 30,
    "synthesis_timeout": 60,
}

GCS_CONFIG = {
    "voice_name": "ja-JP-Neural2-B",
    "language_code": "ja-JP",
    "audio_encoding": "LINEAR16",
    "sample_rate_hertz": 24000,
    "synthesis_timeout": 60,
}

JST = timezone(timedelta(hours=9))


class TTSBackend(Protocol):
    name: str

    def synthesize_to_wav(self, text: str, output_path: Path) -> None:
        ...


def generate_tts_for_work(
    work_dir: Path,
    workspace_root: Path | None = None,
    *,
    backend_name: str | None = None,
    backend: TTSBackend | None = None,
    speaker_id: int | str | None = None,
    now: datetime | None = None,
) -> dict:
    work_dir = Path(work_dir)
    workspace_root = Path(workspace_root) if workspace_root is not None else work_dir.parent
    tts_settings = resolve_tts_settings(workspace_root, work_dir)
    tts_settings_report = resolve_tts_settings_with_sources(workspace_root, work_dir)
    audio_dir = work_dir / "audio"
    timestamp = _timestamp(now)
    report = _new_report(work_dir, _resolve_backend_name(backend_name, backend))
    report["tts_settings"] = tts_settings_report
    previous_audio_exists = (audio_dir / AUDIO_FILENAME).exists()
    previous_sync_exists = (audio_dir / SYNC_MAP_FILENAME).exists()
    report["previous_audio_exists"] = previous_audio_exists
    report["previous_sync_map_exists"] = previous_sync_exists
    report["preserved_previous_outputs"] = previous_audio_exists or previous_sync_exists

    audio_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = audio_dir / f"_build_tmp_{timestamp}"
    failed_dir = audio_dir / f"_build_failed_{timestamp}"

    try:
        precheck = run_tts_pre_transform(work_dir, workspace_root)
        report["warnings"].extend(precheck.get("warnings", []))
        if precheck.get("errors"):
            report["errors"].extend(precheck.get("errors", []))
            return _fail_report(audio_dir, report, None)

        tts_units = _ordered_tts_units(precheck.get("tts_units", []))
        if not tts_units:
            report["errors"].append(_error("empty_tts_units", "tts_units is empty"))
            return _fail_report(audio_dir, report, None)

        tmp_dir.mkdir(parents=True, exist_ok=False)
        resolved_backend = backend or create_backend(
            report["backend"],
            tts_settings=tts_settings,
            speaker_id=speaker_id,
        )
        _record_backend_settings(report, resolved_backend)
        _record_effective_tts_settings(report)
        unit_records = _render_units(tts_units, tmp_dir, resolved_backend, report)
        if report["errors"]:
            return _fail_report(audio_dir, report, tmp_dir, failed_dir)

        concat_wav = tmp_dir / CONCAT_WAV_FILENAME
        final_mp3 = tmp_dir / AUDIO_FILENAME
        _concat_unit_wavs(unit_records, concat_wav, report)
        if not report["errors"]:
            _encode_mp3(concat_wav, final_mp3, report)
        if report["errors"]:
            return _fail_report(audio_dir, report, tmp_dir, failed_dir)

        sync_map = _build_sync_map(unit_records, final_mp3)
        _validate_sync_map(sync_map, unit_records, final_mp3, report)
        if report["errors"]:
            return _fail_report(audio_dir, report, tmp_dir, failed_dir)

        _commit_outputs(work_dir, tmp_dir, sync_map, report, timestamp, now)
        shutil.rmtree(tmp_dir)
        report["ok"] = True
        report["committed"] = True
        report["new_outputs_committed"] = True
        report["duration_seconds"] = sync_map["total_duration"]
        report["segments"] = len(sync_map["sync_map"])
        report["tts_units"] = len(unit_records)
        report["speak_units"] = sum(1 for item in unit_records if item["unit"].get("type") == "speak")
        report["pause_units"] = sum(1 for item in unit_records if item["unit"].get("type") == "pause")
        report["preserved_previous_outputs"] = False
        write_json(audio_dir / TTS_BUILD_REPORT_FILENAME, report)
        return report
    except Exception as error:
        report["errors"].append(_error("tts_generation_exception", f"{type(error).__name__}: {error}"))
        if tmp_dir.exists():
            return _fail_report(audio_dir, report, tmp_dir, failed_dir)
        return _fail_report(audio_dir, report, None)


def create_backend(
    backend_name: str,
    *,
    tts_settings: dict | None = None,
    speaker_id: int | str | None = None,
) -> TTSBackend:
    name = _normalize_backend_name(backend_name)
    if name == "VOICEVOX":
        return VoicevoxBackend.from_environment(tts_settings=tts_settings, speaker_id=speaker_id)
    if name == "GCS":
        return GcsBackend.from_environment()
    raise ValueError(f"unsupported backend: {backend_name}")


def generate_voice_preview_for_work(
    work_dir: Path,
    workspace_root: Path | None = None,
    *,
    speaker_id: int | str | None = None,
    now: datetime | None = None,
) -> dict:
    work_dir = Path(work_dir)
    workspace_root = Path(workspace_root) if workspace_root is not None else work_dir.parent
    tts_settings = resolve_tts_settings(workspace_root, work_dir)
    preview_dir = work_dir / "audio" / VOICE_PREVIEW_DIRNAME
    return generate_voice_preview(
        preview_dir,
        tts_settings=tts_settings,
        speaker_id=speaker_id,
        work_id=work_dir.name,
        audio_file=str(Path("audio") / VOICE_PREVIEW_DIRNAME / VOICE_PREVIEW_FILENAME),
        now=now,
    )


def generate_workspace_voice_preview(
    workspace_root: Path,
    *,
    speaker_id: int | str | None = None,
    now: datetime | None = None,
) -> dict:
    workspace_root = Path(workspace_root)
    return generate_voice_preview(
        workspace_root / WORKSPACE_TEMP_DIRNAME,
        tts_settings=resolve_tts_settings(workspace_root),
        speaker_id=speaker_id,
        work_id="workspace",
        audio_file=str(Path(WORKSPACE_TEMP_DIRNAME) / VOICE_PREVIEW_FILENAME),
        now=now,
    )


def generate_voice_preview(
    preview_dir: Path,
    *,
    tts_settings: dict,
    speaker_id: int | str | None,
    work_id: str,
    audio_file: str,
    now: datetime | None = None,
) -> dict:
    preview_dir = Path(preview_dir)
    timestamp = _timestamp(now)
    tmp_dir = preview_dir / f"_preview_tmp_{timestamp}"
    failed_dir = preview_dir / f"_preview_failed_{timestamp}"
    report = {
        "ok": False,
        "backend": "VOICEVOX",
        "speaker_id": None,
        "preview_text": VOICE_PREVIEW_TEXT,
        "audio_file": audio_file,
        "errors": [],
        "work_id": work_id,
    }

    preview_dir.mkdir(parents=True, exist_ok=True)
    try:
        tmp_dir.mkdir(parents=True, exist_ok=False)
        backend = VoicevoxBackend.from_environment(tts_settings=tts_settings, speaker_id=speaker_id)
        report["speaker_id"] = int(backend.speaker_id)
        preview_wav = tmp_dir / "voice_preview.wav"
        preview_mp3 = tmp_dir / VOICE_PREVIEW_FILENAME
        backend.synthesize_to_wav(VOICE_PREVIEW_TEXT, preview_wav)
        _encode_mp3(preview_wav, preview_mp3, report)
        if report["errors"]:
            raise RuntimeError(report["errors"][0]["message"])
        shutil.move(str(preview_mp3), preview_dir / VOICE_PREVIEW_FILENAME)
        shutil.rmtree(tmp_dir)
        report["ok"] = True
        return report
    except Exception as error:
        report["errors"].append(_error("voice_preview_failed", f"{type(error).__name__}: {error}"))
        if tmp_dir.exists():
            if failed_dir.exists():
                shutil.rmtree(failed_dir)
            shutil.move(str(tmp_dir), failed_dir)
            report["failed_build_dir"] = str(failed_dir)
        return report


class VoicevoxBackend:
    name = "VOICEVOX"

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        resolved = dict(VOICEVOX_CONFIG)
        if config:
            resolved.update(config)
        self.url = str(resolved["url"]).rstrip("/")
        self.speaker_id = int(resolved["speaker_id"])
        self.audio_query_timeout = float(resolved["audio_query_timeout"])
        self.synthesis_timeout = float(resolved["synthesis_timeout"])
        retry = TTS_RETRY["VOICEVOX"]
        self.max_retries = int(retry["max_retries"])
        self.retry_wait_seconds = float(retry["retry_wait_seconds"])

    @classmethod
    def from_environment(
        cls,
        *,
        tts_settings: dict | None = None,
        speaker_id: int | str | None = None,
    ) -> "VoicevoxBackend":
        voice_settings = {}
        if isinstance(tts_settings, dict) and isinstance(tts_settings.get("voice"), dict):
            voice_settings = tts_settings["voice"]
        config = {
            "url": os.environ.get("VOICEVOX_URL", VOICEVOX_CONFIG["url"]),
            "speaker_id": voice_settings.get("speaker_id", VOICEVOX_CONFIG["speaker_id"]),
        }
        env_speaker_id = os.environ.get("VOICEVOX_SPEAKER_ID", os.environ.get("SPEAKER_ID"))
        if env_speaker_id is not None and env_speaker_id.strip():
            config["speaker_id"] = env_speaker_id
        if speaker_id is not None and str(speaker_id).strip():
            config["speaker_id"] = speaker_id
        return cls(config)

    def synthesize_to_wav(self, text: str, output_path: Path) -> None:
        last_error: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                query = self._audio_query(text)
                wav_bytes = self._synthesis(query)
                if not wav_bytes:
                    raise RuntimeError("VOICEVOX synthesis returned empty content")
                output_path.write_bytes(wav_bytes)
                duration = wav_duration_seconds(output_path)
                if duration <= 0:
                    raise RuntimeError("VOICEVOX wav duration is zero")
                return
            except Exception as error:
                last_error = error
                if attempt < self.max_retries:
                    time.sleep(self.retry_wait_seconds)
        raise RuntimeError(f"VOICEVOX synthesis failed: {last_error}")

    def _audio_query(self, text: str) -> dict:
        params = urllib.parse.urlencode({"text": text, "speaker": self.speaker_id})
        request = urllib.request.Request(
            f"{self.url}/audio_query?{params}",
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=self.audio_query_timeout) as response:
            status = getattr(response, "status", 0)
            if status != 200:
                raise RuntimeError(f"VOICEVOX audio_query status {status}")
            body = response.read()
        try:
            data = json.loads(body.decode("utf-8"))
        except json.JSONDecodeError as error:
            raise RuntimeError(f"VOICEVOX audio_query JSON parse failed: {error}") from error
        if not isinstance(data, dict):
            raise RuntimeError("VOICEVOX audio_query response must be object")
        return data

    def _synthesis(self, query: dict) -> bytes:
        params = urllib.parse.urlencode({"speaker": self.speaker_id})
        body = json.dumps(query, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(
            f"{self.url}/synthesis?{params}",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=self.synthesis_timeout) as response:
            status = getattr(response, "status", 0)
            if status != 200:
                raise RuntimeError(f"VOICEVOX synthesis status {status}")
            return response.read()


class GcsBackend:
    name = "GCS"

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        resolved = dict(GCS_CONFIG)
        if config:
            resolved.update(config)
        self.config = resolved
        retry = TTS_RETRY["GCS"]
        self.max_retries = int(retry["max_retries"])
        self.retry_wait_seconds = float(retry["retry_wait_seconds"])

    @classmethod
    def from_environment(cls) -> "GcsBackend":
        config = {
            "voice_name": os.environ.get("GCS_VOICE", GCS_CONFIG["voice_name"]),
            "language_code": os.environ.get("GCS_LANGUAGE_CODE", GCS_CONFIG["language_code"]),
        }
        return cls(config)

    def synthesize_to_wav(self, text: str, output_path: Path) -> None:
        last_error: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                self._synthesize_once(text, output_path)
                duration = wav_duration_seconds(output_path)
                if duration <= 0:
                    raise RuntimeError("GCS wav duration is zero")
                return
            except Exception as error:
                last_error = error
                if attempt < self.max_retries:
                    time.sleep(self.retry_wait_seconds)
        raise RuntimeError(f"GCS synthesis failed: {last_error}")

    def _synthesize_once(self, text: str, output_path: Path) -> None:
        try:
            from google.cloud import texttospeech
        except Exception as error:
            raise RuntimeError("google-cloud-texttospeech is not available") from error

        client = texttospeech.TextToSpeechClient()
        synthesis_input = texttospeech.SynthesisInput(text=text)
        voice = texttospeech.VoiceSelectionParams(
            language_code=self.config["language_code"],
            name=self.config["voice_name"],
        )
        audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.LINEAR16,
            sample_rate_hertz=int(self.config["sample_rate_hertz"]),
        )
        response = client.synthesize_speech(
            input=synthesis_input,
            voice=voice,
            audio_config=audio_config,
            timeout=float(self.config["synthesis_timeout"]),
        )
        audio_content = getattr(response, "audio_content", b"")
        if not audio_content:
            raise RuntimeError("GCS synthesis returned empty content")
        output_path.write_bytes(audio_content)


def _render_units(tts_units: list[dict], tmp_dir: Path, backend: TTSBackend, report: dict) -> list[dict]:
    records: list[dict] = []
    for order, unit in enumerate(tts_units):
        output_path = tmp_dir / f"unit_{order:05d}_{unit.get('unit_id', 'unknown')}.wav"
        try:
            if unit.get("type") == "speak":
                backend.synthesize_to_wav(str(unit.get("text") or ""), output_path)
            elif unit.get("type") == "pause":
                pause_type = str(unit.get("pause_type") or "")
                seconds = SILENCE.get(pause_type)
                if seconds is None:
                    raise RuntimeError(f"unknown pause_type: {pause_type}")
                generate_silence_wav(output_path, seconds)
            else:
                raise RuntimeError(f"unknown tts unit type: {unit.get('type')}")
            duration = wav_duration_seconds(output_path)
            if duration <= 0:
                raise RuntimeError("unit wav duration is zero")
            records.append({"unit": unit, "path": output_path, "duration": duration, "order": order})
        except Exception as error:
            report["errors"].append(
                _error(
                    "tts_unit_generation_failed",
                    f"{type(error).__name__}: {error}",
                    unit_id=unit.get("unit_id"),
                    segment_index=unit.get("segment_index"),
                )
            )
            return records
    return records


def generate_silence_wav(output_path: Path, seconds: float) -> None:
    command = [
        _ffmpeg_bin(),
        "-y",
        "-f",
        "lavfi",
        "-i",
        "anullsrc=r=24000:cl=mono",
        "-t",
        f"{seconds:.3f}",
        "-acodec",
        "pcm_s16le",
        str(output_path),
    ]
    _run_command(command, "ffmpeg_silence_failed")


def wav_duration_seconds(path: Path) -> float:
    with wave.open(str(path), "rb") as handle:
        frames = handle.getnframes()
        rate = handle.getframerate()
        if rate <= 0:
            return 0.0
        return frames / float(rate)


def final_audio_duration_seconds(path: Path) -> float:
    command = [_ffmpeg_bin(), "-i", str(path), "-f", "null", "-"]
    completed = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False)
    output = completed.stderr + completed.stdout
    match = re.search(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", output)
    if not match:
        raise RuntimeError("could not read final audio duration")
    hours = int(match.group(1))
    minutes = int(match.group(2))
    seconds = float(match.group(3))
    return hours * 3600 + minutes * 60 + seconds


def _concat_unit_wavs(unit_records: list[dict], output_path: Path, report: dict) -> None:
    concat_list = output_path.parent / "concat_list.txt"
    concat_list.write_text(
        "".join(f"file '{record['path'].resolve().as_posix()}'\n" for record in unit_records),
        encoding="utf-8",
    )
    command = [
        _ffmpeg_bin(),
        "-y",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        str(concat_list),
        "-ac",
        "1",
        "-ar",
        "24000",
        "-acodec",
        "pcm_s16le",
        str(output_path),
    ]
    try:
        _run_command(command, "ffmpeg_concat_failed")
    except Exception as error:
        report["errors"].append(_error("ffmpeg_concat_failed", str(error)))


def _encode_mp3(input_wav: Path, output_mp3: Path, report: dict) -> None:
    command = [
        _ffmpeg_bin(),
        "-y",
        "-i",
        str(input_wav),
        "-codec:a",
        "libmp3lame",
        "-b:a",
        "96k",
        "-ar",
        "24000",
        "-ac",
        "1",
        str(output_mp3),
    ]
    try:
        _run_command(command, "ffmpeg_mp3_failed")
    except Exception as error:
        report["errors"].append(_error("ffmpeg_mp3_failed", str(error)))


def _build_sync_map(unit_records: list[dict], final_mp3: Path) -> dict:
    sync_items: list[dict] = []
    current_segment: int | None = None
    segment_start = 0.0
    cursor = 0.0
    for record in unit_records:
        segment_index = int(record["unit"]["segment_index"])
        if current_segment is None:
            current_segment = segment_index
            segment_start = cursor
        elif segment_index != current_segment:
            sync_items.append({"index": current_segment, "start": round(segment_start, 3), "end": round(cursor, 3)})
            current_segment = segment_index
            segment_start = cursor
        cursor += float(record["duration"])
    if current_segment is not None:
        sync_items.append({"index": current_segment, "start": round(segment_start, 3), "end": round(cursor, 3)})
    return {
        "schema_version": SYNC_MAP_SCHEMA_VERSION,
        "audio_filename": final_mp3.name,
        "total_duration": round(cursor, 3),
        "sync_map": sync_items,
    }


def _validate_sync_map(sync_map: dict, unit_records: list[dict], final_mp3: Path, report: dict) -> None:
    items = sync_map.get("sync_map", [])
    seen_indexes: set[int] = set()
    previous_start = -math.inf
    previous_end = 0.0
    for item in items:
        index = item.get("index")
        start = float(item.get("start", 0))
        end = float(item.get("end", 0))
        if index in seen_indexes:
            report["errors"].append(_error("duplicate_sync_map_index", "sync_map index is duplicated", index=index))
        seen_indexes.add(index)
        if not start < end:
            report["errors"].append(_error("invalid_sync_map_range", "sync_map start must be smaller than end", index=index))
        if start < previous_start or start < previous_end - 0.001:
            report["errors"].append(_error("sync_map_not_monotonic", "sync_map is not monotonic", index=index))
        previous_start = start
        previous_end = end

    expected_segments = len({int(record["unit"]["segment_index"]) for record in unit_records})
    if len(items) != expected_segments:
        report["errors"].append(
            _error(
                "sync_map_segment_count_mismatch",
                "sync_map count does not match target segment count",
                expected=expected_segments,
                actual=len(items),
            )
        )

    measured_duration = final_audio_duration_seconds(final_mp3)
    expected_duration = float(sync_map.get("total_duration", 0.0))
    if abs(measured_duration - expected_duration) > SYNC_DURATION_TOLERANCE_SECONDS:
        report["errors"].append(
            _error(
                "sync_duration_mismatch",
                "audio duration and sync_map total duration differ",
                audio_duration=round(measured_duration, 3),
                sync_duration=round(expected_duration, 3),
                tolerance=SYNC_DURATION_TOLERANCE_SECONDS,
            )
        )


def _commit_outputs(work_dir: Path, tmp_dir: Path, sync_map: dict, report: dict, timestamp: str, now: datetime | None) -> None:
    audio_dir = work_dir / "audio"
    _backup_existing_outputs(audio_dir, timestamp)
    write_json(tmp_dir / SYNC_MAP_FILENAME, sync_map)
    shutil.move(str(tmp_dir / AUDIO_FILENAME), audio_dir / AUDIO_FILENAME)
    shutil.move(str(tmp_dir / SYNC_MAP_FILENAME), audio_dir / SYNC_MAP_FILENAME)
    _prune_backups(audio_dir / "backups")
    _update_meta_after_audio_success(work_dir, now)
    report["audio_file"] = AUDIO_FILENAME
    report["sync_map_file"] = SYNC_MAP_FILENAME


def _backup_existing_outputs(audio_dir: Path, timestamp: str) -> None:
    backups_dir = audio_dir / "backups"
    for filename in (AUDIO_FILENAME, SYNC_MAP_FILENAME):
        source = audio_dir / filename
        if not source.exists():
            continue
        backups_dir.mkdir(parents=True, exist_ok=True)
        backup_name = f"{timestamp}_{filename}"
        shutil.copy2(source, backups_dir / backup_name)


def _prune_backups(backups_dir: Path) -> None:
    if not backups_dir.exists():
        return
    timestamps = sorted({path.name.split("_", 1)[0] for path in backups_dir.iterdir() if "_" in path.name})
    for old_timestamp in timestamps[:-AUDIO_BACKUP_KEEP]:
        for path in backups_dir.glob(f"{old_timestamp}_*"):
            if path.is_file():
                path.unlink()


def _update_meta_after_audio_success(work_dir: Path, now: datetime | None) -> None:
    meta_path = work_dir / workspace_paths.WORK_META_FILENAME
    if not meta_path.exists():
        return
    meta = read_json(meta_path)
    if not isinstance(meta, dict):
        return
    timestamp = _jst_datetime(now).isoformat()
    meta["status"] = "audio_generated"
    meta["updated_at"] = timestamp
    meta["audio_updated_at"] = timestamp
    write_json(meta_path, meta)


def _fail_report(audio_dir: Path, report: dict, tmp_dir: Path | None, failed_dir: Path | None = None) -> dict:
    if tmp_dir is not None and tmp_dir.exists():
        failed_dir = failed_dir or audio_dir / tmp_dir.name.replace("_build_tmp_", "_build_failed_", 1)
        if failed_dir.exists():
            shutil.rmtree(failed_dir)
        shutil.move(str(tmp_dir), failed_dir)
        report["failed_build_dir"] = str(failed_dir)
    report["ok"] = False
    report["committed"] = False
    report["new_outputs_committed"] = False
    report["duration_seconds"] = 0.0
    write_json(audio_dir / TTS_BUILD_REPORT_FILENAME, report)
    return report


def _record_backend_settings(report: dict, backend: TTSBackend) -> None:
    if report.get("backend") == "VOICEVOX":
        speaker_id = getattr(backend, "speaker_id", None)
        if speaker_id is not None:
            report["speaker_id"] = int(speaker_id)
    elif report.get("backend") == "GCS":
        config = getattr(backend, "config", {})
        if isinstance(config, dict):
            report["voice_name"] = config.get("voice_name")
            report["language_code"] = config.get("language_code")


def _record_effective_tts_settings(report: dict) -> None:
    settings = report.get("tts_settings")
    if not isinstance(settings, dict):
        return
    voice = settings.get("voice")
    if not isinstance(voice, dict):
        return
    voice["backend"] = report.get("backend")
    if report.get("speaker_id") is not None:
        voice["speaker_id"] = report.get("speaker_id")


def _ordered_tts_units(tts_units: list[dict]) -> list[dict]:
    return sorted(tts_units, key=lambda item: (int(item.get("segment_index", 0)), str(item.get("unit_id", ""))))


def _run_command(command: list[str], code: str) -> None:
    completed = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False)
    if completed.returncode != 0:
        message = completed.stderr.strip() or completed.stdout.strip() or f"command failed: {' '.join(command)}"
        raise RuntimeError(f"{code}: {message}")


def _ffmpeg_bin() -> str:
    candidate = os.environ.get(FFMPEG_BIN_ENV) or FFMPEG_BIN_DEFAULT
    resolved = shutil.which(candidate)
    if resolved:
        return resolved
    raise RuntimeError("ffmpeg is not available")


def _new_report(work_dir: Path, backend_name: str) -> dict:
    return {
        "schema_version": TTS_BUILD_REPORT_SCHEMA_VERSION,
        "ok": False,
        "backend": backend_name,
        "speaker_id": None,
        "voice_name": None,
        "language_code": None,
        "source_file": workspace_paths.PROCESSED_FINAL_FILENAME,
        "audio_file": AUDIO_FILENAME,
        "sync_map_file": SYNC_MAP_FILENAME,
        "segments": 0,
        "tts_units": 0,
        "speak_units": 0,
        "pause_units": 0,
        "duration_seconds": 0.0,
        "warnings": [],
        "errors": [],
        "committed": False,
        "preserved_previous_outputs": False,
        "previous_audio_exists": False,
        "previous_sync_map_exists": False,
        "new_outputs_committed": False,
        "work_id": work_dir.name,
    }


def _resolve_backend_name(backend_name: str | None, backend: TTSBackend | None) -> str:
    if backend is not None:
        return _normalize_backend_name(getattr(backend, "name", "MOCK"))
    return _normalize_backend_name(backend_name or os.environ.get("TTS_BACKEND") or TTS_BACKEND_DEFAULT)


def _normalize_backend_name(backend_name: str) -> str:
    return str(backend_name or TTS_BACKEND_DEFAULT).strip().upper()


def _timestamp(now: datetime | None = None) -> str:
    return _jst_datetime(now).strftime("%Y%m%d-%H%M%S")


def _jst_datetime(now: datetime | None = None) -> datetime:
    value = now or datetime.now(JST)
    if value.tzinfo is None:
        return value.replace(tzinfo=JST)
    return value.astimezone(JST)


def _error(code: str, message: str, **extra: Any) -> dict:
    return {"level": "error", "code": code, "message": message, **extra}
