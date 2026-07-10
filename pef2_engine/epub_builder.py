from __future__ import annotations

import html
import json
import math
import re
import shutil
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from pef2_engine import workspace_paths
from pef2_engine.image_alt_review import image_dimensions
from pef2_engine.image_paths import (
    AmbiguousImagePathError,
    ImagePathError,
    image_filename,
    resolve_existing_image,
    validate_image_reference,
)
from pef2_engine.io_utils import read_json, write_json


EPUB_BUILD_REPORT_SCHEMA_VERSION = "epub-build-report-1"
EPUB_BACKUP_KEEP = 2
PROCESSED_FINAL_FILENAME = workspace_paths.PROCESSED_FINAL_FILENAME
AUDIO_FILENAME = "audio.mp3"
SYNC_MAP_FILENAME = "sync_map.json"
EPUB_BUILD_REPORT_FILENAME = "epub_build_report.json"
MIMETYPE_CONTENT = "application/epub+zip"
JST = timezone(timedelta(hours=9))
IMAGE_MEDIA_TYPES = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
}
CONTROL_ALLOWED = {0x09, 0x0A, 0x0D}


def generate_epub_for_work(
    work_dir: Path,
    workspace_root: Path | None = None,
    *,
    now: datetime | None = None,
    allow_missing_images: bool = True,
) -> dict:
    work_dir = Path(work_dir)
    workspace_root = Path(workspace_root) if workspace_root is not None else work_dir.parent
    epub_dir = work_dir / "epub"
    timestamp = _timestamp(now)
    report = _new_report(work_dir)
    report["allow_missing_images"] = bool(allow_missing_images)
    epub_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = epub_dir / f"_build_tmp_{timestamp}"
    failed_dir = epub_dir / f"_build_failed_{timestamp}"

    try:
        tmp_dir.mkdir(parents=True, exist_ok=False)
        context = _preflight(work_dir, workspace_root, report, allow_missing_images=allow_missing_images)
        if report["errors"]:
            return _fail_report(report, tmp_dir, failed_dir)

        output_name = f"{context['safe_title']}.epub"
        report["output_epub"] = f"epub/{output_name}"
        _build_epub_tmp(tmp_dir, context, output_name, report)
        if report["errors"]:
            return _fail_report(report, tmp_dir, failed_dir)

        epub_path = tmp_dir / output_name
        _postbuild(epub_path, context, report)
        if report["errors"]:
            return _fail_report(report, tmp_dir, failed_dir)

        report["ok"] = True
        report["committed"] = True
        report["checks"]["preflight"] = True
        report["checks"]["postbuild"] = True
        report["segments"] = len(context["epub_segments"])
        report["sync_map_count"] = len(context["sync_items"])
        report["images"] = [item["href"] for item in context["images"]]
        report["duration_seconds"] = context["duration_seconds"]
        _commit_outputs(work_dir, epub_dir, tmp_dir, output_name, report, timestamp, now)
        shutil.rmtree(tmp_dir)
        return report
    except Exception as error:
        report["errors"].append(_error("epub_generation_exception", f"{type(error).__name__}: {error}"))
        if tmp_dir.exists():
            return _fail_report(report, tmp_dir, failed_dir)
        return _fail_report(report, None, None)


def _preflight(work_dir: Path, workspace_root: Path, report: dict, *, allow_missing_images: bool) -> dict:
    processed_path = work_dir / PROCESSED_FINAL_FILENAME
    audio_path = work_dir / "audio" / AUDIO_FILENAME
    sync_path = work_dir / "audio" / SYNC_MAP_FILENAME
    images_dir = work_dir / "images"
    alt_review = _load_image_alt_review(work_dir, report)
    alt_items_by_segment = _image_alt_items_by_segment(alt_review, report)

    processed = _read_required_json(
        processed_path,
        "missing_processed_final",
        "invalid_processed_final_json",
        report,
    )
    sync_data = _read_required_json(
        sync_path,
        "missing_sync_map",
        "invalid_sync_map_json",
        report,
    )
    if not audio_path.exists():
        report["errors"].append(_error("missing_audio", f"missing file: {audio_path}"))
    elif audio_path.stat().st_size == 0:
        report["errors"].append(_error("empty_audio", "audio.mp3 size is zero"))

    segments = _extract_segments(processed, report)
    sync_items = _extract_sync_items(sync_data, report)
    if report["errors"]:
        return {}

    segment_by_index = _segments_by_index(segments, report)
    sync_by_index = _sync_by_index(sync_items, report)
    _validate_sync_ranges(sync_items, report)
    _validate_index_sets(segment_by_index, sync_by_index, report)
    if report["errors"]:
        return {}

    epub_segments: list[dict] = []
    images: dict[str, dict] = {}
    for segment in sorted(segments, key=lambda item: int(item["index"])):
        epub_segment = _make_epub_segment(
            segment,
            sync_by_index[int(segment["index"])],
            images_dir,
            report,
            alt_items_by_segment=alt_items_by_segment,
            allow_missing_images=allow_missing_images,
        )
        if epub_segment:
            epub_segments.append(epub_segment)
            image_item = epub_segment.get("image")
            if image_item and not image_item.get("missing"):
                images[image_item["href"]] = image_item

    if report["errors"]:
        return {}

    title = _resolve_title(processed, work_dir)
    safe_title = workspace_paths.sanitize_work_title(title)
    duration_seconds = max(float(item["end"]) for item in sync_items) if sync_items else 0.0
    return {
        "work_dir": work_dir,
        "workspace_root": workspace_root,
        "processed": processed,
        "title": title,
        "safe_title": safe_title,
        "audio_path": audio_path,
        "sync_items": sorted(sync_items, key=lambda item: int(item["index"])),
        "epub_segments": epub_segments,
        "images": list(images.values()),
        "duration_seconds": round(duration_seconds, 3),
    }


def _build_epub_tmp(tmp_dir: Path, context: dict, output_name: str, report: dict) -> None:
    oebps_dir = tmp_dir / "OEBPS"
    meta_inf_dir = tmp_dir / "META-INF"
    audio_dir = oebps_dir / "audio"
    images_dir = oebps_dir / "images"
    oebps_dir.mkdir(parents=True, exist_ok=True)
    meta_inf_dir.mkdir(parents=True, exist_ok=True)
    audio_dir.mkdir(parents=True, exist_ok=True)
    if context["images"]:
        images_dir.mkdir(parents=True, exist_ok=True)

    (tmp_dir / "mimetype").write_text(MIMETYPE_CONTENT, encoding="ascii")
    (meta_inf_dir / "container.xml").write_text(_container_xml(), encoding="utf-8")
    (oebps_dir / "text.xhtml").write_text(_text_xhtml(context), encoding="utf-8")
    (oebps_dir / "nav.xhtml").write_text(_nav_xhtml(context), encoding="utf-8")
    (oebps_dir / "smil.smil").write_text(_smil_xml(context), encoding="utf-8")
    (oebps_dir / "content.opf").write_text(_content_opf(context), encoding="utf-8")
    shutil.copy2(context["audio_path"], audio_dir / AUDIO_FILENAME)
    for image in context["images"]:
        shutil.copy2(image["source"], images_dir / image["filename"])

    _write_epub_zip(tmp_dir, tmp_dir / output_name)


def _postbuild(epub_path: Path, context: dict, report: dict) -> None:
    if not epub_path.exists():
        report["errors"].append(_error("missing_built_epub", ".epub file was not generated"))
        return
    if epub_path.stat().st_size == 0:
        report["errors"].append(_error("empty_built_epub", ".epub size is zero"))
        return

    try:
        with zipfile.ZipFile(epub_path, "r") as archive:
            infos = archive.infolist()
            names = archive.namelist()
            if not infos or infos[0].filename != "mimetype":
                report["errors"].append(_error("mimetype_not_first", "mimetype is not first zip entry"))
            elif infos[0].compress_type != zipfile.ZIP_STORED:
                report["errors"].append(_error("mimetype_compressed", "mimetype must be stored without compression"))

            required = {
                "META-INF/container.xml",
                "OEBPS/content.opf",
                "OEBPS/nav.xhtml",
                "OEBPS/text.xhtml",
                "OEBPS/smil.smil",
                "OEBPS/audio/audio.mp3",
            }
            for name in sorted(required):
                if name not in names:
                    report["errors"].append(_error("missing_zip_entry", "required EPUB entry is missing", entry=name))

            if report["errors"]:
                return

            content_opf = archive.read("OEBPS/content.opf").decode("utf-8")
            text_xhtml = archive.read("OEBPS/text.xhtml").decode("utf-8")
            smil_xml = archive.read("OEBPS/smil.smil").decode("utf-8")
            _validate_opf(content_opf, report)
            _validate_smil_text_refs(text_xhtml, smil_xml, len(context["sync_items"]), report)
            for image in context["images"]:
                if image["href"] not in content_opf:
                    report["errors"].append(_error("missing_image_manifest", "image is missing in manifest", image=image["href"]))
                if f"OEBPS/{image['href']}" not in names:
                    report["errors"].append(_error("missing_image_zip_entry", "image is missing in EPUB zip", image=image["href"]))
    except zipfile.BadZipFile as error:
        report["errors"].append(_error("invalid_epub_zip", f"{type(error).__name__}: {error}"))


def _commit_outputs(
    work_dir: Path,
    epub_dir: Path,
    tmp_dir: Path,
    output_name: str,
    report: dict,
    timestamp: str,
    now: datetime | None,
) -> None:
    target_epub = epub_dir / output_name
    _backup_existing_epub(target_epub, epub_dir / "backups", timestamp)
    write_json(tmp_dir / EPUB_BUILD_REPORT_FILENAME, report)
    shutil.move(str(tmp_dir / output_name), target_epub)
    shutil.move(str(tmp_dir / EPUB_BUILD_REPORT_FILENAME), epub_dir / EPUB_BUILD_REPORT_FILENAME)
    _prune_backups(epub_dir / "backups")
    _update_meta_after_epub_success(work_dir, now)


def _fail_report(report: dict, tmp_dir: Path | None, failed_dir: Path | None) -> dict:
    report["ok"] = False
    report["committed"] = False
    report["duration_seconds"] = 0.0
    if tmp_dir is not None and tmp_dir.exists():
        failed_dir = failed_dir or tmp_dir.parent / tmp_dir.name.replace("_build_tmp_", "_build_failed_", 1)
        if failed_dir.exists():
            shutil.rmtree(failed_dir)
        write_json(tmp_dir / EPUB_BUILD_REPORT_FILENAME, report)
        shutil.move(str(tmp_dir), failed_dir)
        report["failed_build_dir"] = str(failed_dir)
    return report


def _extract_segments(processed: object, report: dict) -> list[dict]:
    if not isinstance(processed, dict):
        report["errors"].append(_error("invalid_processed_final", "04_processed_final.json top-level must be object"))
        return []
    segments = processed.get("segments")
    if segments is None and isinstance(processed.get("remastered_data"), list):
        report["warnings"].append(_warning("legacy_remastered_data_used", "remastered_data was used as segment list"))
        segments = processed.get("remastered_data")
    if not isinstance(segments, list):
        report["errors"].append(_error("missing_segments", "segments must be a list"))
        return []
    if not segments:
        report["errors"].append(_error("empty_segments", "segments is empty"))
        return []

    valid_segments: list[dict] = []
    seen: set[int] = set()
    for position, segment in enumerate(segments):
        if not isinstance(segment, dict):
            report["errors"].append(_error("invalid_segment", "segment must be object", position=position))
            continue
        index = segment.get("index")
        if not _is_nonnegative_int(index):
            report["errors"].append(_error("invalid_segment_index", "segment.index must be non-negative integer", position=position))
            continue
        if int(index) in seen:
            report["errors"].append(_error("duplicate_segment_index", "segment.index is duplicated", index=index))
            continue
        seen.add(int(index))
        display = segment.get("display")
        if not isinstance(display, str):
            report["errors"].append(_error("invalid_display", "segment.display must be string", index=index))
            continue
        valid_segments.append(segment)
    return valid_segments


def _extract_sync_items(sync_data: object, report: dict) -> list[dict]:
    if not isinstance(sync_data, dict):
        report["errors"].append(_error("invalid_sync_map", "sync_map.json top-level must be object"))
        return []
    sync_items = sync_data.get("sync_map")
    if not isinstance(sync_items, list):
        report["errors"].append(_error("missing_sync_map_items", "sync_map must be a list"))
        return []
    if not sync_items:
        report["errors"].append(_error("empty_sync_map", "sync_map is empty"))
        return []
    valid_items: list[dict] = []
    seen: set[int] = set()
    for position, item in enumerate(sync_items):
        if not isinstance(item, dict):
            report["errors"].append(_error("invalid_sync_item", "sync_map item must be object", position=position))
            continue
        index = item.get("index")
        if not _is_nonnegative_int(index):
            report["errors"].append(_error("invalid_sync_index", "sync_map index must be non-negative integer", position=position))
            continue
        if int(index) in seen:
            report["errors"].append(_error("duplicate_sync_map_index", "sync_map index is duplicated", index=index))
            continue
        seen.add(int(index))
        start = item.get("start")
        end = item.get("end")
        if not _is_number(start) or not _is_number(end):
            report["errors"].append(_error("invalid_sync_time", "sync_map start/end must be number", index=index))
            continue
        valid_items.append({"index": int(index), "start": float(start), "end": float(end)})
    return valid_items


def _segments_by_index(segments: list[dict], report: dict) -> dict[int, dict]:
    by_index: dict[int, dict] = {}
    for segment in segments:
        index = int(segment["index"])
        if index in by_index:
            report["errors"].append(_error("duplicate_segment_index", "segment.index is duplicated", index=index))
        by_index[index] = segment
    return by_index


def _sync_by_index(sync_items: list[dict], report: dict) -> dict[int, dict]:
    by_index: dict[int, dict] = {}
    for item in sync_items:
        index = int(item["index"])
        if index in by_index:
            report["errors"].append(_error("duplicate_sync_map_index", "sync_map index is duplicated", index=index))
        by_index[index] = item
    return by_index


def _validate_sync_ranges(sync_items: list[dict], report: dict) -> None:
    previous_start = -math.inf
    previous_end = 0.0
    for item in sorted(sync_items, key=lambda value: int(value["index"])):
        index = item["index"]
        start = float(item["start"])
        end = float(item["end"])
        if not start < end:
            report["errors"].append(_error("invalid_sync_map_range", "sync_map start must be smaller than end", index=index))
        if start < previous_start or start < previous_end - 0.001:
            report["errors"].append(_error("sync_map_not_monotonic", "sync_map time is not monotonic", index=index))
        previous_start = start
        previous_end = end


def _validate_index_sets(segment_by_index: dict[int, dict], sync_by_index: dict[int, dict], report: dict) -> None:
    segment_indexes = set(segment_by_index)
    sync_indexes = set(sync_by_index)
    missing_in_segments = sorted(sync_indexes - segment_indexes)
    missing_in_sync = sorted(segment_indexes - sync_indexes)
    for index in missing_in_segments:
        report["errors"].append(_error("sync_index_missing_in_segments", "sync_map index is missing in 04_processed_final.json", index=index))
    for index in missing_in_sync:
        report["errors"].append(_error("segment_index_missing_in_sync", "04_processed_final.json segment index is missing in sync_map", index=index))


def _make_epub_segment(
    segment: dict,
    sync_item: dict,
    images_dir: Path,
    report: dict,
    *,
    alt_items_by_segment: dict[int, dict],
    allow_missing_images: bool,
) -> dict | None:
    index = int(segment["index"])
    sid = _sid(index)
    par_id = _par_id(index)
    display = _safe_display(segment["display"])
    kind = _segment_kind(segment)
    epub_segment = {
        "index": index,
        "sid": sid,
        "par_id": par_id,
        "kind": kind,
        "display": display,
        "clip_begin": _format_time(float(sync_item["start"])),
        "clip_end": _format_time(float(sync_item["end"])),
        "image": None,
        "para_start": bool(segment.get("para_start")),
        "line_start": bool(segment.get("line_start")),
    }
    if kind == "image":
        alt_text, alt_empty = _resolve_image_alt(index, alt_items_by_segment)
        if alt_empty:
            report["alt_empty_count"] = int(report.get("alt_empty_count") or 0) + 1
        image_file = segment.get("image_file")
        if not isinstance(image_file, str) or not image_file.strip():
            if allow_missing_images:
                image_item = _missing_image_item("", images_dir, index)
                image_item["alt"] = alt_text
                report["warnings"].append(
                    _warning(
                        "missing_images",
                        "image_file is missing; fallback text was inserted",
                        index=index,
                        image_file="",
                        path=str(image_item["searched_path"]),
                    )
                )
                epub_segment["image"] = image_item
                return epub_segment
            report["errors"].append(_error("missing_image_file", "image segment image_file is empty", index=index))
            return None
        image_item = _resolve_image(image_file, images_dir, index, report, allow_missing_images=allow_missing_images)
        if image_item is None:
            return None
        image_item["alt"] = alt_text
        epub_segment["image"] = image_item
    return epub_segment


def _segment_kind(segment: dict) -> str:
    if segment.get("block_type") == "title" or segment.get("is_chapter") is True:
        return "title"
    if segment.get("block_type") == "image" or segment.get("is_image") is True:
        return "image"
    return "text"


def _resolve_image(
    image_file: str,
    images_dir: Path,
    index: int,
    report: dict,
    *,
    allow_missing_images: bool,
) -> dict | None:
    try:
        validated_image_file = validate_image_reference(image_file)
        filename = image_filename(validated_image_file)
    except ImagePathError:
        if allow_missing_images:
            return _unavailable_image_item(
                image_file,
                images_dir,
                index,
                report,
                reason="invalid_image_file",
                message="image_file must be relative; fallback text was inserted",
            )
        report["errors"].append(_error("invalid_image_file", "image_file must be relative", index=index, image_file=image_file))
        return None
    extension = Path(filename).suffix.lower()
    media_type = IMAGE_MEDIA_TYPES.get(extension)
    if media_type is None:
        if allow_missing_images:
            return _unavailable_image_item(
                image_file,
                images_dir,
                index,
                report,
                filename=filename,
                reason="unsupported_image_type",
                message="image type is unsupported; fallback text was inserted",
            )
        report["errors"].append(_error("unsupported_image_type", "image type must be png, jpg, or jpeg", index=index, image_file=image_file))
        return None
    try:
        resolved_image = resolve_existing_image(images_dir, validated_image_file)
    except AmbiguousImagePathError as error:
        report["errors"].append(
            _error("ambiguous_image_file", str(error), index=index, image_file=image_file)
        )
        return None
    except ImagePathError:
        if allow_missing_images:
            return _unavailable_image_item(
                image_file,
                images_dir,
                index,
                report,
                filename=filename,
                reason="invalid_image_path",
                message="image_file resolves outside images/; fallback text was inserted",
            )
        report["errors"].append(
            _error("invalid_image_path", "image_file resolves outside images/", index=index, image_file=image_file)
        )
        return None
    if resolved_image is None:
        if allow_missing_images:
            image_item = _missing_image_item(image_file, images_dir, index, filename=filename)
            report["warnings"].append(
                _warning(
                    "missing_images",
                    "image_file is missing; fallback text was inserted",
                    index=index,
                    image_file=image_file,
                    path=str(image_item["searched_path"]),
                )
            )
            return image_item
        report["errors"].append(
            _error(
                "missing_image",
                "image_file is missing",
                index=index,
                image_file=image_file,
                path=str(images_dir / filename),
            )
        )
        return None
    source = resolved_image.path
    if not _image_file_readable(source):
        if allow_missing_images:
            return _unavailable_image_item(
                image_file,
                images_dir,
                index,
                report,
                filename=filename,
                reason="unreadable_image",
                message="image_file is unreadable or broken; fallback text was inserted",
            )
        report["errors"].append(_error("unreadable_image", "image_file is unreadable or broken", index=index, image_file=image_file, path=str(source)))
        return None
    return {
        "filename": filename,
        "href": f"images/{filename}",
        "source": source,
        "media_type": media_type,
    }


def _missing_image_item(image_file: str, images_dir: Path, index: int, *, filename: str | None = None) -> dict:
    fallback_filename = filename or Path(str(image_file).replace("\\", "/")).name
    searched_path = images_dir / fallback_filename if fallback_filename else images_dir
    display_image_file = f"images/{fallback_filename}" if fallback_filename else str(image_file or "")
    return {
        "filename": fallback_filename,
        "href": "",
        "source": None,
        "media_type": "",
        "missing": True,
        "image_file": display_image_file,
        "searched_path": searched_path,
        "index": index,
    }


def _unavailable_image_item(
    image_file: str,
    images_dir: Path,
    index: int,
    report: dict,
    *,
    reason: str,
    message: str,
    filename: str | None = None,
) -> dict:
    image_item = _missing_image_item(image_file, images_dir, index, filename=filename)
    report["warnings"].append(
        _warning(
            "missing_images",
            message,
            index=index,
            image_file=image_item["image_file"],
            path=str(image_item["searched_path"]),
            reason=reason,
        )
    )
    return image_item


def _image_file_readable(path: Path) -> bool:
    width, height = image_dimensions(path)
    return width is not None and height is not None


def _container_xml() -> str:
    return """<?xml version="1.0" encoding="UTF-8"?>
<container version="1.0" xmlns="urn:oasis:names:tc:opendocument:xmlns:container">
  <rootfiles>
    <rootfile full-path="OEBPS/content.opf" media-type="application/oebps-package+xml"/>
  </rootfiles>
</container>
"""


def _text_xhtml(context: dict) -> str:
    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<!DOCTYPE html>',
        '<html xmlns="http://www.w3.org/1999/xhtml" xml:lang="ja" lang="ja">',
        "<head>",
        f"  <title>{_xml_text(context['title'])}</title>",
        "  <meta charset=\"UTF-8\" />",
        "  <style>",
        "    html, body { writing-mode: horizontal-tb; -epub-writing-mode: horizontal-tb; text-orientation: mixed; direction: ltr; }",
        "    body { font-family: sans-serif; line-height: 1.95; margin: 0; padding: 1.5em; color: #222; }",
        "    section { max-width: 42em; margin: 0 auto; }",
        "    p { margin: 0.35em 0 1.45em 0; margin-block: 0.35em 1.45em; text-align: start; }",
        "    br.line-start { display: block; margin-top: 0.35em; margin-block-start: 0.35em; content: \"\"; }",
        "    h1 { font-size: 1.6em; line-height: 1.5; margin: 0 0 1.8em 0; text-align: center; }",
        "    h2 { font-size: 1.25em; line-height: 1.6; margin: 2.2em 0 1.1em 0; margin-block: 2.2em 1.1em; padding: 0.4em 0.6em; }",
        "    .img-box { margin: 1.5em 0; text-align: center; }",
        "    .img-box img { max-width: 90%; height: auto; }",
        "    .missing-image { display: inline-block; padding: 0.8em 1em; border: 1px solid #bbb; color: #666; }",
        "    .mo-active { background-color: #fff59d; color: inherit; font-weight: inherit; border-radius: 0.2em; }",
        "  </style>",
        "</head>",
        "<body>",
        "  <section>",
        f"  <h1>{_xml_text(context['title'])}</h1>",
    ]
    paragraph_open = False
    for segment in context["epub_segments"]:
        if segment["kind"] == "title":
            if paragraph_open:
                lines.append("  </p>")
                paragraph_open = False
            lines.append(f"  <h2 id=\"{segment['sid']}\">{_xml_text(segment['display'])}</h2>")
        elif segment["kind"] == "image":
            if paragraph_open:
                lines.append("  </p>")
                paragraph_open = False
            image = segment["image"]
            if image and image.get("missing"):
                image_label = image.get("image_file") or "不明"
                lines.append(
                    f"  <div class=\"img-box\"><span id=\"{segment['sid']}\"></span>"
                    f"<div class=\"missing-image\">画像ファイルがみつかりません: {_xml_text(image_label)}</div></div>"
                )
            else:
                lines.append(
                    f"  <div class=\"img-box\"><span id=\"{segment['sid']}\"></span>"
                    f"&#x3000;&#x3000;&#x3000;<img src=\"{_xml_attr(image['href'])}\" alt=\"{_xml_attr(image.get('alt', ''))}\" /></div>"
                )
        else:
            if not paragraph_open or segment["para_start"]:
                if paragraph_open:
                    lines.append("  </p>")
                lines.append("  <p>")
                paragraph_open = True
                if segment["line_start"]:
                    lines.append("    <br class=\"line-start\" />")
            elif segment["line_start"]:
                lines.append("    <br class=\"line-start\" />")
            lines.append(f"    <span id=\"{segment['sid']}\">{_xml_text(segment['display'])}</span>")
    if paragraph_open:
        lines.append("  </p>")
    lines.extend(["  </section>", "</body>", "</html>", ""])
    return "\n".join(lines)


def _nav_xhtml(context: dict) -> str:
    title_items = [segment for segment in context["epub_segments"] if segment["kind"] == "title"]
    if not title_items:
        title_items = context["epub_segments"][:1]
    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<!DOCTYPE html>',
        '<html xmlns="http://www.w3.org/1999/xhtml" xmlns:epub="http://www.idpf.org/2007/ops" xml:lang="ja" lang="ja">',
        "<head>",
        f"  <title>{_xml_text(context['title'])}</title>",
        "  <meta charset=\"UTF-8\" />",
        "</head>",
        "<body>",
        "  <nav epub:type=\"toc\" id=\"toc\">",
        "    <h1>目次</h1>",
        "    <ol>",
    ]
    for segment in title_items:
        label = segment["display"] or context["title"]
        lines.append(f"      <li><a href=\"text.xhtml#{segment['sid']}\">{_xml_text(label)}</a></li>")
    lines.extend(["    </ol>", "  </nav>", "</body>", "</html>", ""])
    return "\n".join(lines)


def _smil_xml(context: dict) -> str:
    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<smil xmlns="http://www.w3.org/ns/SMIL" xmlns:epub="http://www.idpf.org/2007/ops" version="3.0">',
        "  <body>",
        "    <seq id=\"seq1\" epub:textref=\"text.xhtml\">",
    ]
    for segment in context["epub_segments"]:
        lines.extend(
            [
                f"      <par id=\"{segment['par_id']}\">",
                f"        <text src=\"text.xhtml#{segment['sid']}\"/>",
                f"        <audio src=\"audio/audio.mp3\" clipBegin=\"{segment['clip_begin']}\" clipEnd=\"{segment['clip_end']}\"/>",
                "      </par>",
            ]
        )
    lines.extend(["    </seq>", "  </body>", "</smil>", ""])
    return "\n".join(lines)


def _content_opf(context: dict) -> str:
    image_items = []
    for image_index, image in enumerate(context["images"], start=1):
        image_items.append(
            f"    <item id=\"image{image_index}\" href=\"{_xml_attr(image['href'])}\" media-type=\"{image['media_type']}\"/>"
        )
    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<package xmlns="http://www.idpf.org/2007/opf" version="3.0" unique-identifier="bookid" prefix="media: http://www.idpf.org/epub/vocab/overlays/#">',
        "  <metadata xmlns:dc=\"http://purl.org/dc/elements/1.1/\">",
        f"    <dc:identifier id=\"bookid\">urn:pef2:{_xml_text(context['work_dir'].name)}</dc:identifier>",
        f"    <dc:title>{_xml_text(context['title'])}</dc:title>",
        "    <dc:language>ja</dc:language>",
        f"    <meta property=\"media:duration\">{_format_time(context['duration_seconds'])}</meta>",
        "    <meta property=\"media:active-class\">mo-active</meta>",
        "  </metadata>",
        "  <manifest>",
        "    <item id=\"nav\" href=\"nav.xhtml\" media-type=\"application/xhtml+xml\" properties=\"nav\"/>",
        "    <item id=\"text1\" href=\"text.xhtml\" media-type=\"application/xhtml+xml\" media-overlay=\"smil1\"/>",
        "    <item id=\"smil1\" href=\"smil.smil\" media-type=\"application/smil+xml\"/>",
        "    <item id=\"audio1\" href=\"audio/audio.mp3\" media-type=\"audio/mpeg\"/>",
    ]
    lines.extend(image_items)
    lines.extend(
        [
            "  </manifest>",
            "  <spine page-progression-direction=\"ltr\">",
            "    <itemref idref=\"text1\"/>",
            "  </spine>",
            "</package>",
            "",
        ]
    )
    return "\n".join(lines)


def _write_epub_zip(tmp_dir: Path, epub_path: Path) -> None:
    with zipfile.ZipFile(epub_path, "w") as archive:
        archive.write(tmp_dir / "mimetype", "mimetype", compress_type=zipfile.ZIP_STORED)
        for path in sorted(tmp_dir.rglob("*")):
            if path == epub_path or path.name == "mimetype" or path.is_dir():
                continue
            archive.write(path, path.relative_to(tmp_dir).as_posix(), compress_type=zipfile.ZIP_DEFLATED)


def _validate_opf(content_opf: str, report: dict) -> None:
    required_fragments = [
        'href="text.xhtml"',
        'href="smil.smil"',
        'href="audio/audio.mp3"',
        'media-overlay="smil1"',
        'property="media:duration"',
        'property="media:active-class">mo-active',
    ]
    for fragment in required_fragments:
        if fragment not in content_opf:
            report["errors"].append(_error("invalid_content_opf", "content.opf required fragment is missing", fragment=fragment))


def _validate_smil_text_refs(text_xhtml: str, smil_xml: str, sync_count: int, report: dict) -> None:
    text_ids = set(re.findall(r'\bid="(s\d{4,})"', text_xhtml))
    smil_ids = re.findall(r'text\.xhtml#(s\d{4,})', smil_xml)
    for sid in smil_ids:
        if sid not in text_ids:
            report["errors"].append(_error("smil_text_id_missing", "smil text src id is missing in text.xhtml", sid=sid))
    par_count = len(re.findall(r"<par\b", smil_xml))
    if par_count != sync_count:
        report["errors"].append(
            _error("smil_par_count_mismatch", "smil par count does not match sync_map count", expected=sync_count, actual=par_count)
        )


def _backup_existing_epub(target_epub: Path, backups_dir: Path, timestamp: str) -> None:
    if not target_epub.exists():
        return
    backups_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(target_epub, backups_dir / f"{timestamp}_{target_epub.name}")


def _prune_backups(backups_dir: Path) -> None:
    if not backups_dir.exists():
        return
    timestamps = sorted({path.name.split("_", 1)[0] for path in backups_dir.iterdir() if "_" in path.name})
    for old_timestamp in timestamps[:-EPUB_BACKUP_KEEP]:
        for path in backups_dir.glob(f"{old_timestamp}_*"):
            if path.is_file():
                path.unlink()


def _update_meta_after_epub_success(work_dir: Path, now: datetime | None) -> None:
    meta_path = work_dir / workspace_paths.WORK_META_FILENAME
    if not meta_path.exists():
        return
    try:
        meta = read_json(meta_path)
    except Exception:
        return
    if not isinstance(meta, dict):
        return
    timestamp = _jst_datetime(now).isoformat()
    meta["status"] = "exported"
    meta["updated_at"] = timestamp
    meta["epub_updated_at"] = timestamp
    write_json(meta_path, meta)


def _read_required_json(path: Path, missing_code: str, invalid_code: str, report: dict) -> object | None:
    if not path.exists():
        report["errors"].append(_error(missing_code, f"missing file: {path}"))
        return None
    try:
        return read_json(path)
    except json.JSONDecodeError as error:
        report["errors"].append(_error(invalid_code, f"{type(error).__name__}: {error}"))
    except Exception as error:
        report["errors"].append(_error("read_error", f"{type(error).__name__}: {error}", path=str(path)))
    return None


def _load_image_alt_review(work_dir: Path, report: dict) -> dict:
    path = workspace_paths.image_alt_review_path(work_dir)
    report["image_alt_review"] = {
        "loaded": False,
        "path": path.name,
        "items": 0,
    }
    if not path.exists():
        return {}
    try:
        review = read_json(path)
    except Exception as error:
        report["warnings"].append(
            _warning("image_alt_review_unreadable", f"{type(error).__name__}: {error}", path=str(path))
        )
        return {}
    if not isinstance(review, dict):
        report["warnings"].append(_warning("image_alt_review_invalid", "image_alt_review.json top-level must be object"))
        return {}
    items = review.get("items")
    report["image_alt_review"] = {
        "loaded": True,
        "path": path.name,
        "items": len(items) if isinstance(items, list) else 0,
    }
    return review


def _image_alt_items_by_segment(review: dict, report: dict) -> dict[int, dict]:
    items = review.get("items")
    if not isinstance(items, list):
        return {}
    by_segment: dict[int, dict] = {}
    duplicates: set[int] = set()
    for item in items:
        if not isinstance(item, dict):
            continue
        index = _coerce_nonnegative_index(item.get("segment_index"))
        if index is None:
            continue
        if index in by_segment:
            duplicates.add(index)
            continue
        by_segment[index] = item
    for index in sorted(duplicates):
        by_segment.pop(index, None)
        report["warnings"].append(
            _warning("image_alt_review_duplicate_segment", "duplicate segment_index was ignored", index=index)
        )
    return by_segment


def _resolve_image_alt(index: int, alt_items_by_segment: dict[int, dict]) -> tuple[str, bool]:
    item = alt_items_by_segment.get(index)
    if not isinstance(item, dict):
        return "", True
    if item.get("is_decorative") is True:
        return "", False
    user_alt = _clean_alt(item.get("user_alt_ja"))
    if user_alt:
        return user_alt, False
    gemini_alt = _clean_alt(item.get("gemini_alt_ja"))
    if gemini_alt:
        return gemini_alt, False
    return "", True


def _clean_alt(value: object) -> str:
    if value is None:
        return ""
    return _remove_invalid_xml_chars(str(value)).strip()


def _resolve_title(processed: object, work_dir: Path) -> str:
    if isinstance(processed, dict):
        title = str(processed.get("title") or "").strip()
        if title:
            return title
    meta_path = work_dir / workspace_paths.WORK_META_FILENAME
    if meta_path.exists():
        try:
            meta = read_json(meta_path)
        except Exception:
            meta = None
        if isinstance(meta, dict):
            title = str(meta.get("title") or "").strip()
            if title:
                return title
    return work_dir.name


def _safe_display(value: str) -> str:
    text = _remove_invalid_xml_chars(value)
    return text.replace("&", "\uff06").replace("<", "\uff1c").replace(">", "\uff1e")


def _remove_invalid_xml_chars(value: str) -> str:
    chars: list[str] = []
    for char in value:
        codepoint = ord(char)
        if codepoint in CONTROL_ALLOWED:
            chars.append(char)
        elif 0x20 <= codepoint <= 0xD7FF:
            chars.append(char)
        elif 0xE000 <= codepoint <= 0xFFFD:
            chars.append(char)
        elif 0x10000 <= codepoint <= 0x10FFFF:
            chars.append(char)
    return "".join(chars)


def _xml_text(value: object) -> str:
    return html.escape(str(value), quote=False)


def _xml_attr(value: object) -> str:
    return html.escape(str(value), quote=True)


def _format_time(seconds: float) -> str:
    millis = int(round(float(seconds) * 1000))
    hours, remainder = divmod(millis, 3600 * 1000)
    minutes, remainder = divmod(remainder, 60 * 1000)
    secs, millis = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"


def _sid(index: int) -> str:
    return f"s{index:04d}"


def _par_id(index: int) -> str:
    return f"p{index:04d}"


def _is_nonnegative_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _coerce_nonnegative_index(value: object) -> int | None:
    if _is_nonnegative_int(value):
        return int(value)
    if isinstance(value, str) and value.strip().isdigit():
        return int(value.strip())
    return None


def _is_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _new_report(work_dir: Path) -> dict:
    return {
        "schema_version": EPUB_BUILD_REPORT_SCHEMA_VERSION,
        "ok": False,
        "committed": False,
        "work_id": work_dir.name,
        "input_processed": PROCESSED_FINAL_FILENAME,
        "input_audio": "audio/audio.mp3",
        "input_sync_map": "audio/sync_map.json",
        "output_epub": "",
        "allow_missing_images": False,
        "alt_empty_count": 0,
        "image_alt_review": {
            "loaded": False,
            "path": "image_alt_review.json",
            "items": 0,
        },
        "segments": 0,
        "sync_map_count": 0,
        "images": [],
        "duration_seconds": 0.0,
        "checks": {
            "preflight": False,
            "postbuild": False,
        },
        "warnings": [],
        "errors": [],
    }


def _timestamp(now: datetime | None = None) -> str:
    return _jst_datetime(now).strftime("%Y%m%d-%H%M%S")


def _jst_datetime(now: datetime | None = None) -> datetime:
    value = now or datetime.now(JST)
    if value.tzinfo is None:
        return value.replace(tzinfo=JST)
    return value.astimezone(JST)


def _error(code: str, message: str, **extra: Any) -> dict:
    return {"level": "error", "code": code, "message": message, **extra}


def _warning(code: str, message: str, **extra: Any) -> dict:
    return {"level": "warning", "code": code, "message": message, **extra}
