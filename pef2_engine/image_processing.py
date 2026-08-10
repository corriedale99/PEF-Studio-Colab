from __future__ import annotations

import shutil
import struct
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageOps


PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
PNG_RAW_COPY_SAFE_CHUNKS = {
    b"IHDR",
    b"PLTE",
    b"IDAT",
    b"IEND",
    b"tRNS",
    b"iCCP",
    b"sRGB",
    b"gAMA",
    b"cHRM",
    b"sBIT",
}


@dataclass(frozen=True)
class ProcessedImage:
    source_size: int
    output_size: int
    optimized: bool
    original_used: bool
    resized: bool
    metadata_removed: bool


def process_image_for_output(
    source_path: Path,
    output_path: Path,
    *,
    max_edge: int | None,
    jpeg_quality: int,
    resample: Image.Resampling,
    reducing_gap: float | None,
    png_compress_level: int,
    jpeg_optimize: bool,
    allow_original_if_not_smaller: bool,
    copy_small_metadata_free_png: bool,
) -> ProcessedImage:
    source = Path(source_path)
    output = Path(output_path)
    source_size = source.stat().st_size
    output_format = _output_format(source)
    unsafe_metadata = source_has_unsafe_metadata(source, output_format)

    with Image.open(source, formats=(output_format,)) as opened:
        opened.load()
        original_orientation = opened.getexif().get(274, 1)
        icc_profile = opened.info.get("icc_profile")
        if not isinstance(icc_profile, bytes):
            icc_profile = None
        transparency = opened.info.get("transparency")
        oriented = ImageOps.exif_transpose(opened)
        resized = max_edge is not None and max(oriented.size) > max_edge

        if (
            output_format == "PNG"
            and copy_small_metadata_free_png
            and not resized
            and not unsafe_metadata
            and original_orientation in {None, 1}
        ):
            output.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, output)
            return ProcessedImage(
                source_size=source_size,
                output_size=output.stat().st_size,
                optimized=False,
                original_used=True,
                resized=False,
                metadata_removed=False,
            )

        image = oriented
        if resized:
            image = _prepare_resampling_mode(image, transparency)
            image.thumbnail(
                (max_edge, max_edge),
                resample,
                reducing_gap=reducing_gap,
            )

        clean = image.copy()
        clean.info.clear()
        output.parent.mkdir(parents=True, exist_ok=True)
        _save_clean_image(
            clean,
            output,
            output_format=output_format,
            jpeg_quality=jpeg_quality,
            png_compress_level=png_compress_level,
            jpeg_optimize=jpeg_optimize,
            icc_profile=icc_profile,
            transparency=transparency,
        )

    output_size = output.stat().st_size
    may_use_original = (
        allow_original_if_not_smaller
        and not resized
        and not unsafe_metadata
        and original_orientation in {None, 1}
        and output_size >= source_size
    )
    if may_use_original:
        shutil.copy2(source, output)
        output_size = output.stat().st_size
        return ProcessedImage(
            source_size=source_size,
            output_size=output_size,
            optimized=False,
            original_used=True,
            resized=False,
            metadata_removed=False,
        )

    return ProcessedImage(
        source_size=source_size,
        output_size=output_size,
        optimized=True,
        original_used=False,
        resized=resized,
        metadata_removed=unsafe_metadata,
    )


def source_is_safe_for_raw_copy(source_path: Path) -> bool:
    source = Path(source_path)
    try:
        output_format = _output_format(source)
        if source_has_unsafe_metadata(source, output_format):
            return False
        with Image.open(source, formats=(output_format,)) as opened:
            opened.load()
            return opened.getexif().get(274, 1) in {None, 1}
    except Exception:
        return False


def source_has_unsafe_metadata(source_path: Path, output_format: str) -> bool:
    if output_format == "JPEG":
        return _jpeg_has_unsafe_metadata(Path(source_path))
    if output_format == "PNG":
        return _png_has_unsafe_metadata(Path(source_path))
    return True


def _output_format(path: Path) -> str:
    extension = path.suffix.lower()
    if extension in {".jpg", ".jpeg"}:
        return "JPEG"
    if extension == ".png":
        return "PNG"
    raise ValueError(f"unsupported image extension: {extension}")


def _prepare_resampling_mode(image: Image.Image, transparency: object) -> Image.Image:
    if image.mode == "P":
        return image.convert("RGBA" if transparency is not None else "RGB")
    if image.mode == "1":
        return image.convert("L")
    return image


def _save_clean_image(
    image: Image.Image,
    output_path: Path,
    *,
    output_format: str,
    jpeg_quality: int,
    png_compress_level: int,
    jpeg_optimize: bool,
    icc_profile: bytes | None,
    transparency: object,
) -> None:
    save_options: dict[str, object] = {"exif": b""}
    if icc_profile is not None:
        save_options["icc_profile"] = icc_profile

    if output_format == "JPEG":
        if image.mode not in {"L", "RGB", "CMYK"}:
            image = image.convert("RGB")
        image.save(
            output_path,
            format="JPEG",
            quality=jpeg_quality,
            optimize=jpeg_optimize,
            **save_options,
        )
        return

    if image.mode == "I":
        image = image.convert("I;16")
    if transparency is not None and image.mode in {"1", "L", "P", "RGB"}:
        save_options["transparency"] = transparency
    image.save(
        output_path,
        format="PNG",
        compress_level=png_compress_level,
        **save_options,
    )


def _jpeg_has_unsafe_metadata(path: Path) -> bool:
    with path.open("rb") as source:
        if source.read(2) != b"\xff\xd8":
            return True
        while True:
            prefix = source.read(1)
            if not prefix:
                return True
            if prefix != b"\xff":
                continue
            marker_bytes = source.read(1)
            while marker_bytes == b"\xff":
                marker_bytes = source.read(1)
            if not marker_bytes:
                return True
            marker = marker_bytes[0]
            if marker in {0xD9, 0xDA}:
                return False
            if marker == 0x00 or 0xD0 <= marker <= 0xD8:
                continue
            length_bytes = source.read(2)
            if len(length_bytes) != 2:
                return True
            segment_length = struct.unpack(">H", length_bytes)[0]
            if segment_length < 2:
                return True
            payload = source.read(segment_length - 2)
            if len(payload) != segment_length - 2:
                return True
            if marker == 0xE0 and payload.startswith((b"JFIF\x00", b"JFXX\x00")):
                continue
            if marker == 0xE2 and payload.startswith(b"ICC_PROFILE\x00"):
                continue
            if marker == 0xEE and payload.startswith(b"Adobe"):
                continue
            if 0xE0 <= marker <= 0xEF or marker == 0xFE:
                return True


def _png_has_unsafe_metadata(path: Path) -> bool:
    with path.open("rb") as source:
        if source.read(len(PNG_SIGNATURE)) != PNG_SIGNATURE:
            return True
        saw_iend = False
        while not saw_iend:
            length_bytes = source.read(4)
            chunk_type = source.read(4)
            if len(length_bytes) != 4 or len(chunk_type) != 4:
                return True
            chunk_length = struct.unpack(">I", length_bytes)[0]
            if chunk_type not in PNG_RAW_COPY_SAFE_CHUNKS:
                return True
            source.seek(chunk_length + 4, 1)
            saw_iend = chunk_type == b"IEND"
        return False
