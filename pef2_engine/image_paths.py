from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path, PureWindowsPath
from typing import Iterable


ALLOWED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}
WINDOWS_DRIVE_PATTERN = re.compile(r"^[A-Za-z]:")


class ImagePathError(ValueError):
    pass


class InvalidImagePathError(ImagePathError):
    pass


class UnsupportedImageExtensionError(ImagePathError):
    pass


class AmbiguousImagePathError(ImagePathError):
    pass


@dataclass(frozen=True)
class ResolvedImagePath:
    image_file: str
    filename: str
    path: Path
    stored_filename: str


def normalize_nfc(value: str) -> str:
    return unicodedata.normalize("NFC", value)


def validate_image_reference(image_file: str) -> str:
    raw = str(image_file or "").strip()
    if not raw or any(ord(char) < 0x20 for char in raw):
        raise InvalidImagePathError("image reference is empty or contains control characters")
    if raw.startswith(("/", "\\")) or "\\" in raw:
        raise InvalidImagePathError("image reference must not be absolute or contain backslashes")
    if WINDOWS_DRIVE_PATTERN.match(raw) or PureWindowsPath(raw).drive:
        raise InvalidImagePathError("image reference must not contain a Windows drive or UNC path")

    parts = raw.split("/")
    if len(parts) == 1:
        filename = parts[0]
    elif len(parts) == 2 and parts[0] == "images":
        filename = parts[1]
    else:
        raise InvalidImagePathError("image reference must be a filename or images/filename")

    if not filename or filename in {".", ".."} or ".." in filename:
        raise InvalidImagePathError("image filename is invalid")
    if "/" in filename or "\\" in filename or Path(filename).name != filename:
        raise InvalidImagePathError("image filename must not contain a directory")
    return f"images/{filename}"


def normalize_image_reference(image_file: str) -> str:
    normalized = normalize_nfc(validate_image_reference(image_file))
    validate_image_extension(normalized.removeprefix("images/"))
    return normalized


def image_filename(image_file: str) -> str:
    return validate_image_reference(image_file).removeprefix("images/")


def validate_image_extension(filename: str) -> None:
    if Path(filename).suffix.lower() not in ALLOWED_IMAGE_EXTENSIONS:
        raise UnsupportedImageExtensionError("image filename extension is unsupported")


def image_compare_key(value: object) -> str:
    return normalize_nfc(str(value or "").strip())


def normalized_name_matches(entries: Iterable[Path], filename: str) -> list[Path]:
    target_key = normalize_nfc(filename)
    return [entry for entry in entries if normalize_nfc(entry.name) == target_key]


def resolve_existing_image(images_dir: Path, image_file: str) -> ResolvedImagePath | None:
    validated = validate_image_reference(image_file)
    filename = validated.removeprefix("images/")
    base = _safe_images_directory(images_dir, create=False)
    if base is None:
        return None

    matches = normalized_name_matches(base.iterdir(), filename)
    if len(matches) > 1:
        raise AmbiguousImagePathError(f"multiple image files normalize to {normalize_nfc(filename)!r}")
    if not matches:
        validate_image_extension(filename)
        return None

    entry = matches[0]
    resolved = _resolve_contained_file(base, entry)
    validate_image_extension(filename)
    return ResolvedImagePath(
        image_file=validated,
        filename=filename,
        path=resolved,
        stored_filename=entry.name,
    )


def resolve_image_upload_target(images_dir: Path, image_file: str) -> Path:
    normalized_reference = normalize_nfc(validate_image_reference(image_file))
    normalized_filename = normalized_reference.removeprefix("images/")
    base = _safe_images_directory(images_dir, create=True)
    if base is None:
        raise InvalidImagePathError("images directory could not be created")

    matches = normalized_name_matches(base.iterdir(), normalized_filename)
    if len(matches) > 1:
        raise AmbiguousImagePathError(
            f"multiple image files normalize to {normalize_nfc(normalized_filename)!r}"
        )
    if matches:
        resolved = _resolve_contained_file(base, matches[0])
        validate_image_extension(normalized_filename)
        return resolved

    validate_image_extension(normalized_filename)
    target = base / normalized_filename
    if target.exists() or target.is_symlink():
        return _resolve_contained_file(base, target)
    return target


def _safe_images_directory(images_dir: Path, *, create: bool) -> Path | None:
    directory = Path(images_dir)
    if directory.name != "images":
        raise InvalidImagePathError("image directory must be named images")
    if directory.is_symlink():
        raise InvalidImagePathError("images directory must not be a symbolic link")
    if not directory.exists():
        if not create:
            return None
        directory.mkdir(parents=True, exist_ok=True)
    if not directory.is_dir():
        raise InvalidImagePathError("images path is not a directory")

    resolved_parent = directory.parent.resolve(strict=True)
    resolved = directory.resolve(strict=True)
    if resolved.parent != resolved_parent or resolved.name != "images":
        raise InvalidImagePathError("images directory resolves outside the work directory")
    return resolved


def _resolve_contained_file(base: Path, path: Path) -> Path:
    try:
        resolved = path.resolve(strict=True)
        resolved.relative_to(base)
    except (FileNotFoundError, OSError, ValueError) as error:
        raise InvalidImagePathError("image path resolves outside the images directory") from error
    if not resolved.is_file():
        raise InvalidImagePathError("image path is not a file")
    return resolved
