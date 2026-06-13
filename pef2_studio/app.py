from __future__ import annotations

import os
from pathlib import Path

try:
    from flask import (
        Flask,
        abort,
        redirect,
        render_template,
        request,
        send_file,
        url_for,
    )
except ModuleNotFoundError as error:
    Flask = None
    abort = None
    redirect = None
    render_template = None
    request = None
    send_file = None
    url_for = None
    FLASK_IMPORT_ERROR = error
else:
    FLASK_IMPORT_ERROR = None

from pef2_studio.workspace_view import (
    WORKS_SORT_OPTIONS,
    add_dictionary_review_item_submission,
    build_workspace_tts_settings_view,
    create_ai_dictionary_review_submission,
    create_empty_dictionary_processed_submission,
    create_manual_dictionary_review_submission,
    finalize_dictionary_review_direct_submission,
    finalize_dictionary_review_submission,
    get_workspace_root,
    import_legacy_dictionary_upload,
    import_legacy_pef_upload,
    import_text_upload,
    load_ai_dictionary_review_confirmation,
    load_dictionary_review_page,
    load_work_images_page,
    load_work_delete_confirmation,
    list_works,
    load_generation_placeholder,
    load_work_detail,
    move_work_to_trash,
    normalize_works_sort,
    save_work_draft,
    save_work_final,
    save_work_image_upload,
    save_work_tts_settings_submission,
    save_workspace_tts_settings_submission,
    resolve_work_image_file,
    resolve_work_dir,
    start_reedit_from_final,
)
from pef2_engine.tts_generator import (
    VOICE_PREVIEW_DIRNAME,
    VOICE_PREVIEW_FILENAME,
    WORKSPACE_TEMP_DIRNAME,
)
from pef2_studio.generation import (
    build_generation_notice_result,
    run_epub_generation,
    run_tts_generation,
    run_voice_preview_generation,
    run_workspace_voice_preview_generation,
)
from pef2_studio.ui_constants import SYMBOL_CATEGORY_UI


def create_app(workspace_root: Path | None = None):
    if FLASK_IMPORT_ERROR is not None:
        raise RuntimeError(
            "Flask is required to run PEF Studio"
        ) from FLASK_IMPORT_ERROR

    app = Flask(__name__)
    resolved_workspace_root = (
        Path(workspace_root) if workspace_root is not None else get_workspace_root()
    )

    @app.get("/")
    def works_index():
        selected_sort = normalize_works_sort(request.args.get("sort"))
        return _render_works_index(selected_sort=selected_sort)

    @app.post("/workspace/voice-settings")
    def save_workspace_tts_settings():
        selected_sort = normalize_works_sort(request.form.get("sort"))
        result = save_workspace_tts_settings_submission(
            resolved_workspace_root, request.form
        )
        return _render_works_index(
            selected_sort=selected_sort,
            workspace_settings_result=result,
            status_code=200 if result.get("status") == "success" else 400,
        )

    @app.post("/workspace/voice-preview")
    def generate_workspace_voice_preview():
        selected_sort = normalize_works_sort(request.form.get("sort"))
        speaker_id = _form_speaker_id(request.form.get("speaker_id", ""))
        if speaker_id is None:
            result = {
                "status": "failed",
                "ok": False,
                "message": "話者IDを確認してください。",
            }
        else:
            result = run_workspace_voice_preview_generation(
                resolved_workspace_root, speaker_id=speaker_id
            )
        return _render_works_index(
            selected_sort=selected_sort,
            workspace_voice_preview_result=result,
            status_code=200 if result.get("status") == "success" else 400,
        )

    @app.get("/workspace/voice-preview/audio")
    def workspace_voice_preview_audio():
        preview_path = (
            resolved_workspace_root / WORKSPACE_TEMP_DIRNAME / VOICE_PREVIEW_FILENAME
        )
        if not preview_path.is_file():
            return "試聴音声が見つかりません。", 404
        return send_file(preview_path, mimetype="audio/mpeg")

    @app.get("/works/<work_id>/delete")
    def confirm_delete_work(work_id: str):
        work = load_work_delete_confirmation(resolved_workspace_root, work_id)
        if work is None:
            abort(404)
        return render_template(
            "work_delete_confirm.html",
            workspace_root=resolved_workspace_root,
            work=work,
            selected_sort=normalize_works_sort(request.args.get("sort")),
            result=None,
        )

    @app.post("/works/<work_id>/delete")
    def delete_work(work_id: str):
        selected_sort = normalize_works_sort(request.form.get("sort"))
        result = move_work_to_trash(resolved_workspace_root, work_id)
        if result.get("status") == "success":
            return redirect(
                url_for("works_index", sort=selected_sort, delete_notice="deleted")
            )
        work = load_work_delete_confirmation(resolved_workspace_root, work_id)
        if work is None:
            abort(404)
        return (
            render_template(
                "work_delete_confirm.html",
                workspace_root=resolved_workspace_root,
                work=work,
                selected_sort=selected_sort,
                result=result,
            ),
            400,
        )

    @app.get("/imports/legacy-pef")
    def import_legacy_pef_form():
        return render_template(
            "import_legacy_pef.html",
            workspace_root=resolved_workspace_root,
            errors=[],
        )

    @app.get("/imports/text")
    def import_text_form():
        return render_template(
            "import_text.html",
            workspace_root=resolved_workspace_root,
            errors=[],
            form_data={"title": ""},
        )

    @app.post("/imports/legacy-pef")
    def import_legacy_pef():
        result = import_legacy_pef_upload(
            resolved_workspace_root,
            json_upload=request.files.get("source_json"),
            txt_upload=request.files.get("source_txt"),
        )
        if result.get("status") == "success":
            return redirect(url_for("work_detail", work_id=result["work_id"]))
        return (
            render_template(
                "import_legacy_pef.html",
                workspace_root=resolved_workspace_root,
                errors=result.get("errors", []),
            ),
            400,
        )

    @app.post("/imports/text")
    def import_text():
        result = import_text_upload(
            resolved_workspace_root,
            title=request.form.get("title", ""),
            txt_upload=request.files.get("source_txt"),
        )
        if result.get("status") == "success":
            return redirect(
                url_for(
                    "work_detail",
                    work_id=result["work_id"],
                    import_notice="text_imported",
                )
            )
        return (
            render_template(
                "import_text.html",
                workspace_root=resolved_workspace_root,
                errors=result.get("errors", []),
                form_data={"title": request.form.get("title", "")},
            ),
            400,
        )

    @app.get("/works/<work_id>")
    def work_detail(work_id: str):
        return _render_work_detail(
            work_id,
            view=request.args.get("view"),
            page=request.args.get("page"),
            per_page=request.args.get("per_page"),
            generation_result=build_generation_notice_result(
                work_id, request.args.get("generation_notice")
            ),
            dictionary_import_result=_import_notice_result(
                request.args.get("import_notice")
            ),
        )

    @app.get("/works/<work_id>/reading-edit")
    def reading_edit(work_id: str):
        return _render_reading_edit(
            work_id,
            view=request.args.get("view"),
            page=request.args.get("page"),
            per_page=request.args.get("per_page"),
        )

    @app.get("/works/<work_id>/dictionary-review")
    def dictionary_review(work_id: str):
        return _render_dictionary_review(work_id)

    @app.get("/works/<work_id>/images")
    def work_images(work_id: str):
        page = load_work_images_page(resolved_workspace_root, work_id)
        if page is None:
            abort(404)
        return render_template(
            "work_images.html",
            workspace_root=resolved_workspace_root,
            page=page,
        )

    @app.post("/works/<work_id>/images/<segment_index>/upload")
    def upload_work_image(work_id: str, segment_index: str):
        result = save_work_image_upload(
            resolved_workspace_root,
            work_id,
            segment_index,
            request.files.get("image_upload"),
        )
        if result is None:
            abort(404)
        page = load_work_images_page(resolved_workspace_root, work_id, result=result)
        if page is None:
            abort(404)
        return (
            render_template(
                "work_images.html",
                workspace_root=resolved_workspace_root,
                page=page,
            ),
            200 if result.get("status") == "success" else 400,
        )

    @app.get("/works/<work_id>/images/<segment_index>/file")
    def work_image_file(work_id: str, segment_index: str):
        image_path = resolve_work_image_file(
            resolved_workspace_root, work_id, segment_index
        )
        if image_path is None:
            abort(404)
        suffix = image_path.suffix.lower()
        mimetype = "image/png" if suffix == ".png" else "image/jpeg"
        return send_file(image_path, mimetype=mimetype)

    @app.post("/works/<work_id>/dictionary-review/add-item")
    def add_dictionary_review_item(work_id: str):
        result = add_dictionary_review_item_submission(
            resolved_workspace_root, work_id, request.form
        )
        if result is None:
            abort(404)
        return _render_dictionary_review(
            work_id,
            result=result,
            form_items=result.get("form_items"),
            status_code=200 if result.get("status") == "success" else 400,
        )

    @app.post("/works/<work_id>/dictionary-review/finalize")
    def finalize_dictionary_review(work_id: str):
        result = finalize_dictionary_review_submission(
            resolved_workspace_root, work_id, request.form
        )
        if result is None:
            abort(404)
        if result.get("status") == "success":
            return redirect(url_for("work_detail", work_id=work_id))
        return _render_dictionary_review(
            work_id,
            result=result,
            form_items=result.get("form_items"),
            status_code=400,
        )

    @app.post("/works/<work_id>/dictionary-review/direct-finalize")
    def direct_finalize_dictionary_review(work_id: str):
        result = finalize_dictionary_review_direct_submission(
            resolved_workspace_root, work_id
        )
        if result is None:
            abort(404)
        if result.get("status") == "success":
            return redirect(url_for("work_detail", work_id=work_id))
        result.setdefault("title", "辞書から編集用データを作成")
        return _render_work_detail(
            work_id,
            dictionary_import_result=result,
            status_code=400,
        )

    @app.post("/works/<work_id>/create-empty-dictionary-processed")
    def create_empty_dictionary_processed(work_id: str):
        result = create_empty_dictionary_processed_submission(
            resolved_workspace_root, work_id
        )
        if result is None:
            abort(404)
        if result.get("status") == "success":
            return redirect(url_for("work_detail", work_id=work_id))
        result.setdefault("title", "辞書なし編集用データ作成結果")
        return _render_work_detail(
            work_id,
            dictionary_import_result=result,
            status_code=400,
        )

    @app.post("/works/<work_id>/create-manual-dictionary-review")
    def create_manual_dictionary_review(work_id: str):
        result = create_manual_dictionary_review_submission(
            resolved_workspace_root, work_id
        )
        if result is None:
            abort(404)
        if result.get("status") == "success":
            return redirect(url_for("dictionary_review", work_id=work_id))
        result.setdefault("title", "手動辞書作成結果")
        return _render_work_detail(
            work_id,
            dictionary_import_result=result,
            status_code=400,
        )

    @app.post("/works/<work_id>/create-ai-dictionary-review")
    def create_ai_dictionary_review(work_id: str):
        result = create_ai_dictionary_review_submission(
            resolved_workspace_root, work_id
        )
        if result is None:
            abort(404)
        if result.get("status") == "success":
            return _render_work_detail(work_id, dictionary_import_result=result)
        result.setdefault("title", "AI辞書候補")
        return _render_work_detail(
            work_id,
            dictionary_import_result=result,
            status_code=400,
        )

    @app.get("/works/<work_id>/ai-dictionary-review/confirm")
    def confirm_ai_dictionary_review(work_id: str):
        page_data = load_ai_dictionary_review_confirmation(
            resolved_workspace_root, work_id
        )
        if page_data is None:
            abort(404)
        if page_data.get("status") == "blocked":
            result = page_data.get("result") or {}
            result.setdefault("title", "AI辞書候補")
            return _render_work_detail(
                work_id,
                dictionary_import_result=result,
                status_code=400,
            )
        return render_template(
            "ai_dictionary_confirm.html",
            workspace_root=resolved_workspace_root,
            confirm=page_data,
        )

    @app.post("/works/<work_id>/import-legacy-dictionary")
    def import_legacy_dictionary(work_id: str):
        result = import_legacy_dictionary_upload(
            resolved_workspace_root,
            work_id,
            dictionary_upload=request.files.get("source_dictionary"),
        )
        if result is None:
            abort(404)
        if result.get("status") == "success":
            return _render_work_detail(work_id, dictionary_import_result=result)
        return _render_work_detail(
            work_id,
            dictionary_import_result=result,
            status_code=400,
        )

    @app.post("/works/<work_id>/voice-settings")
    def save_tts_settings(work_id: str):
        result = save_work_tts_settings_submission(
            resolved_workspace_root, work_id, request.form
        )
        if result is None:
            abort(404)
        return _render_work_detail(
            work_id,
            tts_settings_result=result,
            status_code=200 if result.get("status") == "success" else 400,
        )

    @app.post("/works/<work_id>/voice-preview")
    def generate_voice_preview(work_id: str):
        speaker_id = _form_speaker_id(request.form.get("speaker_id", ""))
        if speaker_id is None:
            result = {
                "status": "failed",
                "ok": False,
                "message": "話者IDを確認してください。",
            }
        else:
            result = run_voice_preview_generation(
                resolved_workspace_root, work_id, speaker_id=speaker_id
            )
        if result is None:
            abort(404)
        return _render_work_detail(
            work_id,
            voice_preview_result=result,
            status_code=200 if result.get("status") == "success" else 400,
        )

    @app.get("/works/<work_id>/voice-preview/audio")
    def voice_preview_audio(work_id: str):
        work_dir = resolve_work_dir(resolved_workspace_root, work_id)
        if work_dir is None:
            abort(404)
        preview_path = (
            work_dir / "audio" / VOICE_PREVIEW_DIRNAME / VOICE_PREVIEW_FILENAME
        )
        if not preview_path.is_file():
            return "試聴音声が見つかりません。", 404
        return send_file(preview_path, mimetype="audio/mpeg")

    @app.get("/works/<work_id>/tts")
    def tts_placeholder(work_id: str):
        return generation_placeholder(work_id, "tts")

    @app.get("/works/<work_id>/epub")
    def epub_placeholder(work_id: str):
        return generation_placeholder(work_id, "epub")

    @app.post("/works/<work_id>/tts")
    def generate_tts(work_id: str):
        result = run_tts_generation(resolved_workspace_root, work_id)
        if result is None:
            abort(404)
        detail = load_work_detail(resolved_workspace_root, work_id, view="final")
        if detail is None:
            abort(404)
        return render_template(
            "work_detail.html",
            workspace_root=resolved_workspace_root,
            work=detail,
            symbol_category_ui=SYMBOL_CATEGORY_UI,
            generation_result=result,
        )

    @app.post("/works/<work_id>/epub")
    def generate_epub(work_id: str):
        result = run_epub_generation(
            resolved_workspace_root,
            work_id,
            allow_missing_images=request.form.get("allow_missing_images") == "1",
        )
        if result is None:
            abort(404)
        detail = load_work_detail(resolved_workspace_root, work_id, view="final")
        if detail is None:
            abort(404)
        return render_template(
            "work_detail.html",
            workspace_root=resolved_workspace_root,
            work=detail,
            symbol_category_ui=SYMBOL_CATEGORY_UI,
            generation_result=result,
        )

    @app.get("/works/<work_id>/epub/download")
    def download_epub(work_id: str):
        work_dir = resolve_work_dir(resolved_workspace_root, work_id)
        if work_dir is None:
            abort(404)
        epub_path = _latest_epub_path(work_dir)
        if epub_path is None:
            return "生成済みEPUBが見つかりません。先にEPUB生成を実行してください。", 404
        return send_file(epub_path, as_attachment=True, download_name=epub_path.name)

    def generation_placeholder(work_id: str, generation_kind: str):
        generation = load_generation_placeholder(
            resolved_workspace_root, work_id, generation_kind
        )
        if generation is None:
            abort(404)
        status_code = 200 if generation["can_generate"] else 400
        return (
            render_template(
                "generation_placeholder.html",
                workspace_root=resolved_workspace_root,
                generation=generation,
            ),
            status_code,
        )

    @app.post("/works/<work_id>/draft")
    def save_draft(work_id: str):
        result = save_work_draft(resolved_workspace_root, work_id, request.form)
        if result is None:
            abort(404)
        if result.get("status") != "success":
            segment_index = str(result.get("segment_index") or "")
            return _render_reading_edit(
                work_id,
                page=request.form.get("page"),
                per_page=request.form.get("per_page"),
                save_errors=[_format_save_error_message(segment_index)],
                posted_audio_edits=result.get("posted_edits"),
                segment_errors=_segment_error_map(segment_index),
                status_code=400,
            )
        return redirect(_reading_edit_url(work_id, request.form))

    @app.post("/works/<work_id>/final")
    def save_final(work_id: str):
        result = save_work_final(resolved_workspace_root, work_id, request.form)
        if result is None:
            abort(404)
        if result.get("status") != "success":
            segment_index = str(result.get("segment_index") or "")
            return _render_reading_edit(
                work_id,
                page=request.form.get("page"),
                per_page=request.form.get("per_page"),
                save_errors=[_format_save_error_message(segment_index)],
                posted_audio_edits=result.get("posted_edits"),
                segment_errors=_segment_error_map(segment_index),
                status_code=400,
            )
        return redirect(url_for("work_detail", work_id=work_id))

    @app.post("/works/<work_id>/start-reedit")
    def start_reedit(work_id: str):
        result = start_reedit_from_final(resolved_workspace_root, work_id, request.form)
        if result is None:
            abort(404)
        if result.get("status") == "conflict":
            abort(409)
        if result.get("status") != "success":
            abort(400)
        return redirect(url_for("reading_edit", work_id=work_id))

    def _render_work_detail(
        work_id: str,
        *,
        view: str | None = None,
        page: object | None = None,
        per_page: object | None = None,
        save_errors: list[str] | None = None,
        posted_audio_edits: dict[str, str] | None = None,
        segment_errors: dict[str, str] | None = None,
        generation_result: dict | None = None,
        dictionary_import_result: dict | None = None,
        tts_settings_result: dict | None = None,
        voice_preview_result: dict | None = None,
        status_code: int = 200,
    ):
        detail = load_work_detail(
            resolved_workspace_root,
            work_id,
            view=view,
            page=page,
            per_page=per_page,
            save_errors=save_errors,
            posted_audio_edits=posted_audio_edits,
            segment_errors=segment_errors,
        )
        if detail is None:
            abort(404)
        return (
            render_template(
                "work_detail.html",
                workspace_root=resolved_workspace_root,
                work=detail,
                symbol_category_ui=SYMBOL_CATEGORY_UI,
                generation_result=generation_result,
                dictionary_import_result=dictionary_import_result,
                tts_settings_result=tts_settings_result,
                voice_preview_result=voice_preview_result,
            ),
            status_code,
        )

    def _render_reading_edit(
        work_id: str,
        *,
        view: str | None = None,
        page: object | None = None,
        per_page: object | None = None,
        save_errors: list[str] | None = None,
        posted_audio_edits: dict[str, str] | None = None,
        segment_errors: dict[str, str] | None = None,
        status_code: int = 200,
    ):
        detail = load_work_detail(
            resolved_workspace_root,
            work_id,
            view=view,
            page=page,
            per_page=per_page,
            save_errors=save_errors,
            posted_audio_edits=posted_audio_edits,
            segment_errors=segment_errors,
        )
        if detail is None:
            abort(404)
        return (
            render_template(
                "reading_edit.html",
                workspace_root=resolved_workspace_root,
                work=detail,
                symbol_category_ui=SYMBOL_CATEGORY_UI,
            ),
            status_code,
        )

    def _render_dictionary_review(
        work_id: str,
        *,
        result: dict | None = None,
        form_items: list[dict] | None = None,
        status_code: int = 200,
    ):
        page_data = load_dictionary_review_page(
            resolved_workspace_root,
            work_id,
            form_items=form_items,
        )
        if page_data is None:
            abort(404)
        return (
            render_template(
                "dictionary_review.html",
                workspace_root=resolved_workspace_root,
                review=page_data,
                result=result,
            ),
            status_code,
        )

    def _render_works_index(
        *,
        selected_sort: str,
        workspace_settings_result: dict | None = None,
        workspace_voice_preview_result: dict | None = None,
        status_code: int = 200,
    ):
        return (
            render_template(
                "works_index.html",
                workspace_root=resolved_workspace_root,
                works=list_works(resolved_workspace_root, sort=selected_sort),
                sort_options=WORKS_SORT_OPTIONS,
                selected_sort=selected_sort,
                delete_notice=_delete_notice_result(request.args.get("delete_notice")),
                workspace_tts_settings=build_workspace_tts_settings_view(
                    resolved_workspace_root
                ),
                workspace_settings_result=workspace_settings_result,
                workspace_voice_preview_result=workspace_voice_preview_result,
            ),
            status_code,
        )

    def _form_speaker_id(value: object) -> int | None:
        text = str(value or "").strip()
        if not text.isdigit():
            return None
        speaker_id = int(text)
        return speaker_id if speaker_id >= 0 else None

    return app


def _format_save_error_message(segment_index: str) -> str:
    if segment_index:
        return f"保存エラー。index {segment_index} の読み指定を確認してください。"
    return "保存エラー。読み指定を確認してください。"


def _segment_error_map(segment_index: str) -> dict[str, str]:
    if not segment_index:
        return {}
    return {
        segment_index: "読み指定の書き方を確認してください。例: ｜原文《読み》 または |原文《読み》",
    }


def _reading_edit_url(work_id: str, form_data) -> str:
    values = {"work_id": work_id}
    target_page = form_data.get("target_page") or form_data.get("page")
    if target_page:
        values["page"] = target_page
    per_page = form_data.get("per_page")
    if per_page:
        values["per_page"] = per_page
    return url_for("reading_edit", **values)


def _latest_epub_path(work_dir: Path) -> Path | None:
    epub_dir = work_dir / "epub"
    if not epub_dir.is_dir():
        return None
    candidates = [
        path
        for path in epub_dir.glob("*.epub")
        if path.is_file() and not any(part.startswith("_build_") for part in path.parts)
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


def _import_notice_result(import_notice: str | None) -> dict | None:
    if import_notice == "text_imported":
        return {
            "status": "success",
            "title": "テキスト原稿の取り込み結果",
            "message": "テキスト原稿を取り込みました。",
            "errors": [],
            "warnings": [],
        }
    return None


def _delete_notice_result(delete_notice: str | None) -> dict | None:
    if delete_notice == "deleted":
        return {
            "status": "success",
            "message": "作品を一覧から削除しました。",
        }
    return None


def main() -> None:
    if FLASK_IMPORT_ERROR is not None:
        raise SystemExit(
            "Flask is not installed. Install Flask before running PEF Studio."
        )
    port = int(os.environ.get("PEF_STUDIO_PORT", "5000"))
    create_app().run(host="0.0.0.0", port=port, debug=False)


if __name__ == "__main__":
    main()
