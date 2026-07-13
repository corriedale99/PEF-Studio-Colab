from __future__ import annotations

import base64
import hashlib
import json
import logging
import os
import shutil
import tempfile
import threading
import unicodedata
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from PIL import Image, ImageOps

from pef2_engine.image_paths import normalize_image_reference


LOGGER = logging.getLogger(__name__)

THUMBNAIL_SPEC_VERSION = "thumbnail-v1"
THUMBNAIL_MAX_EDGE = 160
THUMBNAIL_JPEG_QUALITY = 75
THUMBNAIL_CACHE_DIRNAME = "pef2-thumbnail-cache"
THUMBNAIL_PLACEHOLDER_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNgYGBgAAAABQABeqhXUAAAAABJRU5ErkJggg=="
)

_SESSION_ID = uuid.uuid4().hex
_IMAGE_LOCKS: dict[str, threading.Lock] = {}
_IMAGE_LOCKS_GUARD = threading.Lock()
_GENERATION_SEMAPHORE = threading.BoundedSemaphore(2)


@dataclass(frozen=True)
class ThumbnailSourceMetadata:
    cache_key: str
    source_stored_filename: str
    source_size: int
    source_mtime_ns: int
    image_file: str
    spec_version: str
    max_edge: int
    output_format: str
    jpeg_quality: int


@dataclass(frozen=True)
class ThumbnailCacheEntry:
    path: Path
    metadata_path: Path
    metadata: dict[str, Any]
    etag: str
    cache_hit: bool


@dataclass(frozen=True)
class ThumbnailCandidate:
    path: Path
    cache_key: str
    output_format: str


class ThumbnailSourceChangedError(RuntimeError):
    pass


def thumbnail_cache_root() -> Path:
    return Path(tempfile.gettempdir()) / THUMBNAIL_CACHE_DIRNAME


def thumbnail_session_root() -> Path:
    return thumbnail_cache_root() / f"session-{_SESSION_ID}"


def build_thumbnail_cache_key(
    workspace_root: Path,
    work_id: str,
    image_file: str,
) -> str:
    normalized_workspace = _normalized_workspace_root(workspace_root)
    normalized_image_file = normalize_image_reference(image_file)
    payload = {
        "workspace_root": normalized_workspace,
        "work_id": unicodedata.normalize("NFC", str(work_id)),
        "image_file": normalized_image_file,
        "spec_version": THUMBNAIL_SPEC_VERSION,
    }
    return _json_sha256(payload)


def get_or_create_thumbnail(
    workspace_root: Path,
    work_id: str,
    image_file: str,
    source_path: Path,
) -> ThumbnailCacheEntry:
    source = Path(source_path)
    source_metadata = _source_metadata(workspace_root, work_id, image_file, source)
    thumbnail_path, metadata_path = _cache_paths(
        workspace_root,
        work_id,
        source_metadata.cache_key,
        source_metadata.output_format,
    )

    cached = _valid_cache_entry(thumbnail_path, metadata_path, source_metadata)
    if cached is not None:
        return cached

    image_lock = _image_lock(source_metadata.cache_key)
    with image_lock:
        source_metadata = _source_metadata(workspace_root, work_id, image_file, source)
        thumbnail_path, metadata_path = _cache_paths(
            workspace_root,
            work_id,
            source_metadata.cache_key,
            source_metadata.output_format,
        )
        cached = _valid_cache_entry(thumbnail_path, metadata_path, source_metadata)
        if cached is not None:
            return cached
        with _GENERATION_SEMAPHORE:
            source_metadata = _source_metadata(workspace_root, work_id, image_file, source)
            try:
                return _generate_cache_entry(source, thumbnail_path, metadata_path, source_metadata)
            except Exception:
                LOGGER.exception(
                    "Thumbnail generation failed for cache key %s",
                    source_metadata.cache_key,
                )
                raise


def build_thumbnail_candidate(
    workspace_root: Path,
    work_id: str,
    image_file: str,
    source_path: Path,
) -> ThumbnailCandidate:
    normalized_image_file = normalize_image_reference(image_file)
    cache_key = build_thumbnail_cache_key(workspace_root, work_id, normalized_image_file)
    output_format = _output_format(normalized_image_file)
    suffix = ".png" if output_format == "PNG" else ".jpg"
    candidate_path: Path | None = None
    with _image_lock(cache_key):
        with _GENERATION_SEMAPHORE:
            try:
                candidate_path = _temporary_path(
                    Path(tempfile.gettempdir()),
                    suffix,
                    prefix="pef2_thumbnail_candidate_",
                )
                _build_thumbnail_file(Path(source_path), candidate_path, output_format)
                return ThumbnailCandidate(
                    path=candidate_path,
                    cache_key=cache_key,
                    output_format=output_format,
                )
            except Exception as error:
                if candidate_path is not None:
                    _best_effort_remove(candidate_path, "thumbnail candidate")
                LOGGER.warning(
                    "Thumbnail candidate generation failed for cache key %s: %s",
                    cache_key,
                    error,
                )
                raise


def replace_image_and_activate_thumbnail(
    workspace_root: Path,
    work_id: str,
    image_file: str,
    staged_image_path: Path,
    official_image_path: Path,
    candidate: ThumbnailCandidate | None,
) -> bool:
    cache_key = build_thumbnail_cache_key(workspace_root, work_id, image_file)
    with _image_lock(cache_key):
        os.replace(staged_image_path, official_image_path)
        if candidate is None:
            return False
        try:
            source_metadata = _source_metadata(
                workspace_root,
                work_id,
                image_file,
                official_image_path,
            )
            if (
                candidate.cache_key != source_metadata.cache_key
                or candidate.output_format != source_metadata.output_format
            ):
                raise ValueError("thumbnail candidate does not match the official image")
            thumbnail_path, metadata_path = _cache_paths(
                workspace_root,
                work_id,
                source_metadata.cache_key,
                source_metadata.output_format,
            )
            _activate_thumbnail_candidate(
                candidate,
                thumbnail_path,
                metadata_path,
                source_metadata,
            )
            return True
        except Exception:
            LOGGER.exception(
                "Thumbnail candidate activation failed for cache key %s",
                cache_key,
            )
            return False


def discard_thumbnail_candidate(candidate: ThumbnailCandidate | None) -> bool:
    if candidate is None:
        return True
    return _best_effort_remove(candidate.path, "thumbnail candidate")


def cleanup_work_thumbnail_cache(workspace_root: Path, work_id: str) -> bool:
    work_cache_dir = _work_cache_dir(workspace_root, work_id)
    return _best_effort_remove(work_cache_dir, "work thumbnail cache")


def cleanup_current_thumbnail_cache() -> bool:
    return _best_effort_remove(thumbnail_session_root(), "current thumbnail cache")


def cleanup_stale_thumbnail_caches() -> bool:
    root = thumbnail_cache_root()
    if not root.exists():
        return True
    success = True
    try:
        entries = list(root.iterdir())
    except OSError as error:
        LOGGER.warning("Could not inspect stale thumbnail caches: %s", error)
        return False
    current = thumbnail_session_root()
    for entry in entries:
        if entry == current:
            continue
        success = _best_effort_remove(entry, "stale thumbnail cache") and success
    return success


def cleanup_all_thumbnail_caches() -> bool:
    return _best_effort_remove(thumbnail_cache_root(), "all thumbnail caches")


def _source_metadata(
    workspace_root: Path,
    work_id: str,
    image_file: str,
    source_path: Path,
) -> ThumbnailSourceMetadata:
    normalized_image_file = normalize_image_reference(image_file)
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    stat = source_path.stat()
    output_format = _output_format(normalized_image_file)
    return ThumbnailSourceMetadata(
        cache_key=build_thumbnail_cache_key(workspace_root, work_id, normalized_image_file),
        source_stored_filename=source_path.name,
        source_size=stat.st_size,
        source_mtime_ns=stat.st_mtime_ns,
        image_file=normalized_image_file,
        spec_version=THUMBNAIL_SPEC_VERSION,
        max_edge=THUMBNAIL_MAX_EDGE,
        output_format=output_format,
        jpeg_quality=THUMBNAIL_JPEG_QUALITY,
    )


def _cache_paths(
    workspace_root: Path,
    work_id: str,
    cache_key: str,
    output_format: str,
) -> tuple[Path, Path]:
    cache_dir = _work_cache_dir(workspace_root, work_id)
    suffix = ".png" if output_format == "PNG" else ".jpg"
    return cache_dir / f"{cache_key}{suffix}", cache_dir / f"{cache_key}.json"


def _work_cache_dir(workspace_root: Path, work_id: str) -> Path:
    workspace_key = hashlib.sha256(_normalized_workspace_root(workspace_root).encode("utf-8")).hexdigest()
    work_key = hashlib.sha256(unicodedata.normalize("NFC", str(work_id)).encode("utf-8")).hexdigest()
    return thumbnail_session_root() / workspace_key / work_key


def _valid_cache_entry(
    thumbnail_path: Path,
    metadata_path: Path,
    source_metadata: ThumbnailSourceMetadata,
) -> ThumbnailCacheEntry | None:
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        expected_source = asdict(source_metadata)
        if metadata.get("source") != expected_source:
            return None
        if not thumbnail_path.is_file():
            return None
        thumbnail_bytes = thumbnail_path.read_bytes()
        if not thumbnail_bytes:
            return None
        if metadata.get("thumbnail_size") != len(thumbnail_bytes):
            return None
        if metadata.get("thumbnail_sha256") != hashlib.sha256(thumbnail_bytes).hexdigest():
            return None
        etag = str(metadata.get("etag") or "")
        if not etag or etag != _source_etag(expected_source):
            return None
    except (FileNotFoundError, OSError, TypeError, ValueError, json.JSONDecodeError):
        return None
    return ThumbnailCacheEntry(
        path=thumbnail_path,
        metadata_path=metadata_path,
        metadata=metadata,
        etag=etag,
        cache_hit=True,
    )


def _generate_cache_entry(
    source_path: Path,
    thumbnail_path: Path,
    metadata_path: Path,
    source_metadata: ThumbnailSourceMetadata,
) -> ThumbnailCacheEntry:
    thumbnail_path.parent.mkdir(parents=True, exist_ok=True)
    image_tmp: Path | None = None
    metadata_tmp: Path | None = None
    try:
        image_tmp = _temporary_path(thumbnail_path.parent, thumbnail_path.suffix)
        _build_thumbnail_file(source_path, image_tmp, source_metadata.output_format)
        if not _source_still_matches(source_path, source_metadata):
            raise ThumbnailSourceChangedError("source image changed during thumbnail generation")
        thumbnail_bytes = image_tmp.read_bytes()
        source_data = asdict(source_metadata)
        metadata = {
            "source": source_data,
            "thumbnail_size": len(thumbnail_bytes),
            "thumbnail_sha256": hashlib.sha256(thumbnail_bytes).hexdigest(),
            "etag": _source_etag(source_data),
        }
        metadata_tmp = _temporary_path(metadata_path.parent, ".json")
        metadata_tmp.write_text(
            json.dumps(metadata, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(image_tmp, thumbnail_path)
        image_tmp = None
        os.replace(metadata_tmp, metadata_path)
        metadata_tmp = None
        return ThumbnailCacheEntry(
            path=thumbnail_path,
            metadata_path=metadata_path,
            metadata=metadata,
            etag=metadata["etag"],
            cache_hit=False,
        )
    finally:
        for temporary in (image_tmp, metadata_tmp):
            if temporary is None:
                continue
            try:
                temporary.unlink()
            except OSError:
                pass


def _activate_thumbnail_candidate(
    candidate: ThumbnailCandidate,
    thumbnail_path: Path,
    metadata_path: Path,
    source_metadata: ThumbnailSourceMetadata,
) -> None:
    thumbnail_path.parent.mkdir(parents=True, exist_ok=True)
    image_tmp: Path | None = None
    metadata_tmp: Path | None = None
    try:
        image_tmp = _temporary_path(thumbnail_path.parent, thumbnail_path.suffix)
        shutil.copyfile(candidate.path, image_tmp)
        thumbnail_bytes = image_tmp.read_bytes()
        source_data = asdict(source_metadata)
        metadata = {
            "source": source_data,
            "thumbnail_size": len(thumbnail_bytes),
            "thumbnail_sha256": hashlib.sha256(thumbnail_bytes).hexdigest(),
            "etag": _source_etag(source_data),
        }
        metadata_tmp = _temporary_path(metadata_path.parent, ".json")
        metadata_tmp.write_text(
            json.dumps(metadata, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(image_tmp, thumbnail_path)
        image_tmp = None
        os.replace(metadata_tmp, metadata_path)
        metadata_tmp = None
    finally:
        for temporary in (image_tmp, metadata_tmp):
            if temporary is not None:
                _best_effort_remove(temporary, "thumbnail activation temporary file")


def _build_thumbnail_file(source_path: Path, output_path: Path, output_format: str) -> None:
    with Image.open(source_path) as opened:
        image = ImageOps.exif_transpose(opened)
        image.thumbnail(
            (THUMBNAIL_MAX_EDGE, THUMBNAIL_MAX_EDGE),
            Image.Resampling.LANCZOS,
        )
        if output_format == "JPEG":
            if image.mode != "RGB":
                image = image.convert("RGB")
            image.save(
                output_path,
                format="JPEG",
                quality=THUMBNAIL_JPEG_QUALITY,
                optimize=True,
            )
        else:
            image.save(output_path, format="PNG")


def _temporary_path(
    directory: Path,
    suffix: str,
    *,
    prefix: str = ".thumbnail_",
) -> Path:
    with tempfile.NamedTemporaryFile(
        delete=False,
        dir=directory,
        prefix=prefix,
        suffix=suffix,
    ) as temporary:
        return Path(temporary.name)


def _image_lock(cache_key: str) -> threading.Lock:
    with _IMAGE_LOCKS_GUARD:
        lock = _IMAGE_LOCKS.get(cache_key)
        if lock is None:
            lock = threading.Lock()
            _IMAGE_LOCKS[cache_key] = lock
        return lock


def _output_format(image_file: str) -> str:
    return "PNG" if Path(image_file).suffix.lower() == ".png" else "JPEG"


def _source_still_matches(source_path: Path, source_metadata: ThumbnailSourceMetadata) -> bool:
    try:
        stat = source_path.stat()
    except OSError:
        return False
    return (
        source_path.name == source_metadata.source_stored_filename
        and stat.st_size == source_metadata.source_size
        and stat.st_mtime_ns == source_metadata.source_mtime_ns
    )


def _normalized_workspace_root(workspace_root: Path) -> str:
    return unicodedata.normalize("NFC", str(Path(workspace_root).expanduser().resolve(strict=False)))


def _json_sha256(value: Any) -> str:
    serialized = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _source_etag(source_data: dict[str, Any]) -> str:
    return _json_sha256(source_data)


def _best_effort_remove(path: Path, label: str) -> bool:
    try:
        if path.is_dir() and not path.is_symlink():
            shutil.rmtree(path)
        elif path.exists() or path.is_symlink():
            path.unlink()
        return True
    except OSError as error:
        LOGGER.warning("Could not remove %s %s: %s", label, path, error)
        return False
