from __future__ import annotations

import os
from pathlib import Path

try:
    from flask import (
        Flask,
        abort,
        jsonify,
        redirect,
        render_template,
        request,
        send_file,
        url_for,
    )
except ModuleNotFoundError as error:
    Flask = None
    abort = None
    jsonify = None
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
    save_work_image_alt_settings_submission,
    save_work_image_alt_target_submission,
    save_work_image_alt_targets_submission,
    save_work_image_alt_review_submission,
    save_work_bulk_image_uploads,
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
from pef2_engine.generation_lock import (
    STALE_LOCK_CLEARED_MESSAGE,
    active_generation_lock_message,
    clear_stale_generation_lock,
    is_generation_lock_stale,
    read_generation_lock,
)
from pef2_studio.generation import (
    build_generation_notice_result,
    cancel_ai_dictionary_review_task,
    cancel_image_alt_generation_task,
    cancel_tts_generation_task,
    load_image_alt_generation_progress,
    load_ai_dictionary_review_progress,
    load_latest_blocked_generation_result,
    load_tts_generation_progress,
    run_epub_generation,
    run_tts_generation,
    run_voice_preview_generation,
    run_workspace_voice_preview_generation,
    start_ai_dictionary_review_task,
    start_image_alt_generation_task,
    start_tts_generation_task,
)
from pef2_studio.generation_progress import is_valid_task_id
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
            return redirect(
                url_for(
                    "work_detail",
                    work_id=result["work_id"],
                    import_notice="legacy_imported",
                    _anchor="manuscript-card",
                )
            )
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
                    _anchor="manuscript-card",
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
            status_message=_status_notice_result(request.args.get("status_notice")),
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
        page = load_work_images_page(
            resolved_workspace_root,
            work_id,
            selected_segment_index=request.args.get("selected"),
            show_decorative=request.args.get("show_decorative") == "1",
            show_thumbnails=request.args.get("show_thumbnails", "1") == "1",
        )
        if page is None:
            abort(404)
        return render_template(
            "work_images.html",
            workspace_root=resolved_workspace_root,
            page=page,
        )

    @app.post("/works/<work_id>/images/alt-targets")
    def save_work_image_alt_targets(work_id: str):
        show_decorative = request.form.get("show_decorative") == "1"
        show_thumbnails = request.form.get("show_thumbnails", "1") == "1"
        selected_segment_index = request.form.get("selected")
        lock_result = _guard_generation_lock_for_post(work_id)
        if lock_result is not None:
            page = load_work_images_page(
                resolved_workspace_root,
                work_id,
                result=lock_result,
                selected_segment_index=selected_segment_index,
                show_decorative=show_decorative,
                show_thumbnails=show_thumbnails,
            )
            if page is None:
                abort(404)
            return (
                render_template(
                    "work_images.html",
                    workspace_root=resolved_workspace_root,
                    page=page,
                ),
                _lock_status_code(lock_result),
            )
        result = save_work_image_alt_targets_submission(
            resolved_workspace_root,
            work_id,
            show_decorative=show_decorative,
            send_to_ai=request.form.get("send_to_ai") == "1",
        )
        if result is None:
            abort(404)
        result_for_page = None if result.get("status") == "success" else result
        page = load_work_images_page(
            resolved_workspace_root,
            work_id,
            result=result_for_page,
            selected_segment_index=selected_segment_index,
            show_decorative=show_decorative,
            show_thumbnails=show_thumbnails,
        )
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

    @app.post("/works/<work_id>/images/<segment_index>/alt-target")
    def save_work_image_alt_target(work_id: str, segment_index: str):
        lock_result = _guard_generation_lock_for_post(work_id)
        if lock_result is not None:
            return _json_response(
                {
                    "ok": False,
                    "status": "locked",
                    "message": str(lock_result.get("message") or "生成中のため保存できません。"),
                },
                _lock_status_code(lock_result),
            )
        result = save_work_image_alt_target_submission(
            resolved_workspace_root,
            work_id,
            segment_index,
            send_to_ai=request.form.get("send_to_ai") == "1",
        )
        if result is None:
            abort(404)
        return _json_response(
            {
                "ok": result.get("status") == "success",
                "status": result.get("status"),
                "message": result.get("message"),
                "send_to_ai": request.form.get("send_to_ai") == "1" and result.get("status") == "success",
            },
            200 if result.get("status") == "success" else 400,
        )

    @app.post("/works/<work_id>/images/settings")
    def save_work_image_alt_settings(work_id: str):
        lock_result = _guard_generation_lock_for_post(work_id)
        if lock_result is not None:
            return _json_response(
                {
                    "ok": False,
                    "status": "locked",
                    "message": str(lock_result.get("message") or "生成中のため保存できません。"),
                },
                _lock_status_code(lock_result),
            )
        result = save_work_image_alt_settings_submission(
            resolved_workspace_root,
            work_id,
            request.form,
        )
        if result is None:
            abort(404)
        return _json_response(
            {
                "ok": result.get("status") == "success",
                "status": result.get("status"),
                "message": result.get("message"),
                "alt_length_target": result.get("alt_length_target"),
            },
            200 if result.get("status") == "success" else 400,
        )

    @app.post("/works/<work_id>/images/<segment_index>/upload")
    def upload_work_image(work_id: str, segment_index: str):
        show_decorative = request.form.get("show_decorative") == "1"
        show_thumbnails = request.form.get("show_thumbnails", "1") == "1"
        lock_result = _guard_generation_lock_for_post(work_id)
        if lock_result is not None:
            page = load_work_images_page(
                resolved_workspace_root,
                work_id,
                result=lock_result,
                selected_segment_index=segment_index,
                show_decorative=show_decorative,
                show_thumbnails=show_thumbnails,
            )
            if page is None:
                abort(404)
            return (
                render_template(
                    "work_images.html",
                    workspace_root=resolved_workspace_root,
                    page=page,
                ),
                _lock_status_code(lock_result),
            )
        result = save_work_image_upload(
            resolved_workspace_root,
            work_id,
            segment_index,
            request.files.get("image_upload"),
        )
        if result is None:
            abort(404)
        page = load_work_images_page(
            resolved_workspace_root,
            work_id,
            result=result,
            selected_segment_index=segment_index,
            show_decorative=show_decorative,
            show_thumbnails=show_thumbnails,
        )
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

    @app.post("/works/<work_id>/images/bulk-upload")
    def upload_work_images_bulk(work_id: str):
        show_decorative = request.form.get("show_decorative") == "1"
        show_thumbnails = request.form.get("show_thumbnails", "1") == "1"
        selected_segment_index = request.form.get("selected")
        lock_result = _guard_generation_lock_for_post(work_id)
        if lock_result is not None:
            page = load_work_images_page(
                resolved_workspace_root,
                work_id,
                result=lock_result,
                selected_segment_index=selected_segment_index,
                show_decorative=show_decorative,
                show_thumbnails=show_thumbnails,
            )
            if page is None:
                abort(404)
            return (
                render_template(
                    "work_images.html",
                    workspace_root=resolved_workspace_root,
                    page=page,
                ),
                _lock_status_code(lock_result),
            )

        result = save_work_bulk_image_uploads(
            resolved_workspace_root,
            work_id,
            request.files.getlist("image_uploads"),
            overwrite_existing=request.form.get("overwrite_existing") == "1",
        )
        if result is None:
            abort(404)
        page = load_work_images_page(
            resolved_workspace_root,
            work_id,
            result=result,
            selected_segment_index=selected_segment_index,
            show_decorative=show_decorative,
            show_thumbnails=show_thumbnails,
        )
        if page is None:
            abort(404)
        return (
            render_template(
                "work_images.html",
                workspace_root=resolved_workspace_root,
                page=page,
            ),
            400 if result.get("status") == "failed" else 200,
        )

    @app.post("/works/<work_id>/images/<segment_index>/alt")
    def save_work_image_alt_review(work_id: str, segment_index: str):
        show_decorative = request.form.get("show_decorative") == "1"
        show_thumbnails = request.form.get("show_thumbnails", "1") == "1"
        lock_result = _guard_generation_lock_for_post(work_id)
        if lock_result is not None:
            page = load_work_images_page(
                resolved_workspace_root,
                work_id,
                result=lock_result,
                selected_segment_index=segment_index,
                show_decorative=show_decorative,
                show_thumbnails=show_thumbnails,
            )
            if page is None:
                abort(404)
            return (
                render_template(
                    "work_images.html",
                    workspace_root=resolved_workspace_root,
                    page=page,
                ),
                _lock_status_code(lock_result),
            )
        result = save_work_image_alt_review_submission(
            resolved_workspace_root,
            work_id,
            segment_index,
            request.form,
        )
        if result is None:
            abort(404)
        page = load_work_images_page(
            resolved_workspace_root,
            work_id,
            result=result,
            selected_segment_index=segment_index,
            show_decorative=show_decorative,
            show_thumbnails=show_thumbnails,
        )
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

    @app.post("/works/<work_id>/images/alt-generation/start")
    def start_image_alt_generation(work_id: str):
        result = start_image_alt_generation_task(
            resolved_workspace_root,
            work_id,
            segment_index=str(request.form.get("segment_index") or ""),
        )
        if result is None:
            abort(404)
        return _json_response(result, _task_start_status_code(result))

    @app.get("/works/<work_id>/images/alt-generation/progress/<task_id>")
    def image_alt_generation_progress(work_id: str, task_id: str):
        if not is_valid_task_id(task_id, operation="image_alt"):
            return _json_response(
                {
                    "ok": False,
                    "status": "invalid_task_id",
                    "message": "進捗情報を確認できませんでした。",
                },
                400,
            )
        progress = load_image_alt_generation_progress(
            resolved_workspace_root,
            work_id,
            task_id,
        )
        if progress is None:
            return _json_response(
                {
                    "ok": False,
                    "status": "not_found",
                    "message": "進捗情報が見つかりません。画面を再読み込みしてください。",
                },
                404,
            )
        return _json_response(
            {
                "ok": True,
                "status": progress.get("status"),
                "message": progress.get("message"),
                "progress": progress,
            },
            200,
        )

    @app.post("/works/<work_id>/images/alt-generation/cancel/<task_id>")
    def cancel_image_alt_generation(work_id: str, task_id: str):
        if not is_valid_task_id(task_id, operation="image_alt"):
            return _json_response(
                {
                    "ok": False,
                    "status": "invalid_task_id",
                    "message": "進捗情報を確認できませんでした。",
                },
                400,
            )
        result = cancel_image_alt_generation_task(
            resolved_workspace_root,
            work_id,
            task_id,
        )
        if result is None:
            return _json_response(
                {
                    "ok": False,
                    "status": "not_found",
                    "message": "進捗情報が見つかりません。画面を再読み込みしてください。",
                },
                404,
            )
        return _json_response(result, _generation_status_code(result))

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
        lock_result = _guard_generation_lock_for_post(work_id)
        if lock_result is not None:
            return _render_dictionary_review(
                work_id,
                result=lock_result,
                status_code=_lock_status_code(lock_result),
            )
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
        lock_result = _guard_generation_lock_for_post(work_id)
        if lock_result is not None:
            return _render_dictionary_review(
                work_id,
                result=lock_result,
                status_code=_lock_status_code(lock_result),
            )
        result = finalize_dictionary_review_submission(
            resolved_workspace_root, work_id, request.form
        )
        if result is None:
            abort(404)
        if result.get("status") == "success":
            return redirect(
                url_for(
                    "work_detail",
                    work_id=work_id,
                    status_notice="dictionary_finalized",
                    _anchor="dictionary-card",
                )
            )
        return _render_dictionary_review(
            work_id,
            result=result,
            form_items=result.get("form_items"),
            status_code=400,
        )

    @app.post("/works/<work_id>/dictionary-review/direct-finalize")
    def direct_finalize_dictionary_review(work_id: str):
        lock_result = _guard_generation_lock_for_post(work_id)
        if lock_result is not None:
            lock_result.setdefault("title", "辞書から編集用データを作成")
            return _render_work_detail(
                work_id,
                dictionary_import_result=lock_result,
                status_code=_lock_status_code(lock_result),
            )
        result = finalize_dictionary_review_direct_submission(
            resolved_workspace_root, work_id
        )
        if result is None:
            abort(404)
        if result.get("status") == "success":
            return redirect(
                url_for(
                    "work_detail",
                    work_id=work_id,
                    status_notice="dictionary_finalized",
                    _anchor="dictionary-card",
                )
            )
        result.setdefault("title", "辞書から編集用データを作成")
        return _render_work_detail(
            work_id,
            dictionary_import_result=result,
            status_code=400,
        )

    @app.post("/works/<work_id>/create-empty-dictionary-processed")
    def create_empty_dictionary_processed(work_id: str):
        lock_result = _guard_generation_lock_for_post(work_id)
        if lock_result is not None:
            lock_result.setdefault("title", "辞書なし編集用データ作成結果")
            return _render_work_detail(
                work_id,
                dictionary_import_result=lock_result,
                status_code=_lock_status_code(lock_result),
            )
        result = create_empty_dictionary_processed_submission(
            resolved_workspace_root, work_id
        )
        if result is None:
            abort(404)
        if result.get("status") == "success":
            return redirect(
                url_for(
                    "work_detail",
                    work_id=work_id,
                    status_notice="empty_dictionary_processed",
                    _anchor="dictionary-card",
                )
            )
        result.setdefault("title", "辞書なし編集用データ作成結果")
        return _render_work_detail(
            work_id,
            dictionary_import_result=result,
            status_code=400,
        )

    @app.post("/works/<work_id>/create-manual-dictionary-review")
    def create_manual_dictionary_review(work_id: str):
        lock_result = _guard_generation_lock_for_post(work_id)
        if lock_result is not None:
            lock_result.setdefault("title", "手動辞書作成結果")
            return _render_work_detail(
                work_id,
                dictionary_import_result=lock_result,
                status_code=_lock_status_code(lock_result),
            )
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
        lock_result = _guard_generation_lock_for_post(work_id)
        if lock_result is not None:
            lock_result.setdefault("title", "AI辞書候補")
            return _render_work_detail(
                work_id,
                dictionary_import_result=lock_result,
                status_code=_lock_status_code(lock_result),
            )
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

    @app.post("/works/<work_id>/ai-dictionary-review/start")
    def start_ai_dictionary_review(work_id: str):
        result = start_ai_dictionary_review_task(resolved_workspace_root, work_id)
        if result is None:
            abort(404)
        return _json_response(result, _task_start_status_code(result))

    @app.get("/works/<work_id>/ai-dictionary-review/progress/<task_id>")
    def ai_dictionary_review_progress(work_id: str, task_id: str):
        if not is_valid_task_id(task_id, operation="ai_dictionary"):
            return _json_response(
                {
                    "ok": False,
                    "status": "invalid_task_id",
                    "message": "進捗情報を確認できませんでした。",
                },
                400,
            )
        progress = load_ai_dictionary_review_progress(
            resolved_workspace_root,
            work_id,
            task_id,
        )
        if progress is None:
            return _json_response(
                {
                    "ok": False,
                    "status": "not_found",
                    "message": "進捗情報が見つかりません。画面を再読み込みしてください。",
                },
                404,
            )
        return _json_response(
            {
                "ok": True,
                "status": progress.get("status"),
                "message": progress.get("message"),
                "progress": progress,
            },
            200,
        )

    @app.post("/works/<work_id>/ai-dictionary-review/cancel/<task_id>")
    def cancel_ai_dictionary_review(work_id: str, task_id: str):
        if not is_valid_task_id(task_id, operation="ai_dictionary"):
            return _json_response(
                {
                    "ok": False,
                    "status": "invalid_task_id",
                    "message": "進捗情報を確認できませんでした。",
                },
                400,
            )
        result = cancel_ai_dictionary_review_task(
            resolved_workspace_root,
            work_id,
            task_id,
        )
        if result is None:
            return _json_response(
                {
                    "ok": False,
                    "status": "not_found",
                    "message": "進捗情報が見つかりません。画面を再読み込みしてください。",
                },
                404,
            )
        return _json_response(result, 202 if result.get("ok") else 409)

    @app.post("/works/<work_id>/import-legacy-dictionary")
    def import_legacy_dictionary(work_id: str):
        lock_result = _guard_generation_lock_for_post(work_id)
        if lock_result is not None:
            return _render_work_detail(
                work_id,
                dictionary_import_result=lock_result,
                status_code=_lock_status_code(lock_result),
            )
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
        lock_result = _guard_generation_lock_for_post(work_id)
        if lock_result is not None:
            return _render_work_detail(
                work_id,
                tts_settings_result=lock_result,
                status_code=_lock_status_code(lock_result),
            )
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
        return (
            render_template(
                "work_detail.html",
                workspace_root=resolved_workspace_root,
                work=detail,
                symbol_category_ui=SYMBOL_CATEGORY_UI,
                generation_result=result,
            ),
            _generation_status_code(result),
        )

    @app.post("/works/<work_id>/tts/start")
    def start_tts_generation(work_id: str):
        result = start_tts_generation_task(
            resolved_workspace_root,
            work_id,
            next_action="epub" if request.form.get("next_action") == "epub" else "",
        )
        if result is None:
            abort(404)
        return _json_response(result, _task_start_status_code(result))

    @app.get("/works/<work_id>/tts/progress/<task_id>")
    def tts_generation_progress(work_id: str, task_id: str):
        if not is_valid_task_id(task_id, operation="tts"):
            return _json_response(
                {
                    "ok": False,
                    "status": "invalid_task_id",
                    "message": "進捗情報を確認できませんでした。",
                },
                400,
            )
        progress = load_tts_generation_progress(
            resolved_workspace_root,
            work_id,
            task_id,
        )
        if progress is None:
            return _json_response(
                {
                    "ok": False,
                    "status": "not_found",
                    "message": "進捗情報が見つかりません。画面を再読み込みしてください。",
                },
                404,
            )
        return _json_response(
            {
                "ok": True,
                "status": progress.get("status"),
                "message": progress.get("message"),
                "progress": progress,
            },
            200,
        )

    @app.post("/works/<work_id>/tts/cancel/<task_id>")
    def cancel_tts_generation(work_id: str, task_id: str):
        if not is_valid_task_id(task_id, operation="tts"):
            return _json_response(
                {
                    "ok": False,
                    "status": "invalid_task_id",
                    "message": "進捗情報を確認できませんでした。",
                },
                400,
            )
        result = cancel_tts_generation_task(
            resolved_workspace_root,
            work_id,
            task_id,
        )
        if result is None:
            return _json_response(
                {
                    "ok": False,
                    "status": "not_found",
                    "message": "進捗情報が見つかりません。画面を再読み込みしてください。",
                },
                404,
            )
        return _json_response(result, 202 if result.get("ok") else 409)

    @app.post("/works/<work_id>/epub")
    def generate_epub(work_id: str):
        result = run_epub_generation(
            resolved_workspace_root,
            work_id,
            allow_missing_images=True,
        )
        if result is None:
            abort(404)
        detail = load_work_detail(resolved_workspace_root, work_id, view="final")
        if detail is None:
            abort(404)
        return (
            render_template(
                "work_detail.html",
                workspace_root=resolved_workspace_root,
                work=detail,
                symbol_category_ui=SYMBOL_CATEGORY_UI,
                generation_result=result,
            ),
            _generation_status_code(result),
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
        lock_result = _guard_generation_lock_for_post(work_id)
        if lock_result is not None:
            return _render_reading_edit(
                work_id,
                page=request.form.get("page"),
                per_page=request.form.get("per_page"),
                save_errors=[lock_result["message"]],
                status_code=_lock_status_code(lock_result),
            )
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
        lock_result = _guard_generation_lock_for_post(work_id)
        if lock_result is not None:
            return _render_reading_edit(
                work_id,
                page=request.form.get("page"),
                per_page=request.form.get("per_page"),
                save_errors=[lock_result["message"]],
                status_code=_lock_status_code(lock_result),
            )
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
        return redirect(url_for("work_detail", work_id=work_id, status_notice="final_saved", _anchor="reading-edit-card"))

    @app.post("/works/<work_id>/start-reedit")
    def start_reedit(work_id: str):
        lock_result = _guard_generation_lock_for_post(work_id)
        if lock_result is not None:
            return _render_work_detail(
                work_id,
                reading_edit_result=lock_result,
                status_code=_lock_status_code(lock_result),
            )
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
        status_message: dict | None = None,
        reading_edit_result: dict | None = None,
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
        if generation_result is None and not detail["active_generation_lock"]["active"]:
            generation_result = load_latest_blocked_generation_result(
                resolved_workspace_root,
                work_id,
            )
        if dictionary_import_result is None and not detail["active_generation_lock"]["active"]:
            dictionary_import_result = _ai_dictionary_completion_notice(work_id)
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
                status_message=status_message,
                reading_edit_result=reading_edit_result,
            ),
            status_code,
        )

    def _ai_dictionary_completion_notice(work_id: str) -> dict | None:
        task_id = request.args.get("ai_dictionary_task", "")
        if not is_valid_task_id(task_id, operation="ai_dictionary"):
            return None
        progress = load_ai_dictionary_review_progress(
            resolved_workspace_root,
            work_id,
            task_id,
        )
        if not progress or progress.get("status") != "completed":
            return None
        result = progress.get("result") if isinstance(progress.get("result"), dict) else {}
        try:
            draft_count = int(result.get("draft_count") or 0)
        except (TypeError, ValueError):
            draft_count = 0
        return {
            "status": "success",
            "ok": True,
            "title": "AI辞書候補",
            "message": "辞書採用語は0件でした。" if draft_count == 0 else "辞書生成が終わりました。",
            "card_anchor": "dictionary-card",
            "display_scope": "dictionary_card",
        }

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

    def _guard_generation_lock_for_post(work_id: str) -> dict | None:
        work_dir = resolve_work_dir(resolved_workspace_root, work_id)
        if work_dir is None:
            return None
        lock_data = read_generation_lock(work_dir)
        if lock_data is None:
            return None
        if is_generation_lock_stale(lock_data):
            backup_path = clear_stale_generation_lock(work_dir, expected_lock=lock_data)
            return _post_lock_result(
                "stale_cleared",
                STALE_LOCK_CLEARED_MESSAGE,
                backup_path=backup_path,
            )
        return _post_lock_result(
            "locked",
            active_generation_lock_message(lock_data),
            lock_data=lock_data,
        )

    return app


def _post_lock_result(
    status: str,
    message: str,
    *,
    lock_data: dict | None = None,
    backup_path: Path | None = None,
) -> dict:
    errors = [message]
    if lock_data:
        operation = lock_data.get("operation")
        started_at = lock_data.get("started_at")
        if operation:
            errors.append(f"operation: {operation}")
        if started_at:
            errors.append(f"started_at: {started_at}")
    if backup_path is not None:
        errors.append(f"backup_path: {backup_path}")
    return {
        "status": status,
        "ok": False,
        "message": message,
        "errors": errors,
        "committed": False,
        "meta_status_changed": False,
    }


def _lock_status_code(result: dict) -> int:
    return 400 if result.get("status") == "stale_cleared" else 409


def _generation_status_code(result: dict) -> int:
    status = result.get("status")
    if status == "stale_cleared":
        return 400
    if status == "locked":
        return 409
    return 200


def _task_start_status_code(result: dict) -> int:
    status = result.get("status")
    if status == "started":
        return 202
    if status == "locked":
        return 409
    if status == "stale_cleared":
        return 400
    if status == "preflight_failed":
        return 400
    return 500 if status == "failed" else 400


def _json_response(payload: dict, status_code: int):
    response = jsonify(payload)
    response.headers["Cache-Control"] = "no-store"
    return response, status_code


def _lock_generation_result(work_id: str, result: dict) -> dict:
    return {
        "generation_kind": "lock",
        "status": result.get("status"),
        "ok": False,
        "message": result.get("message"),
        "output_paths": [],
        "report_path": "",
        "failed_build_dir": "",
        "missing_files": [],
        "missing_images": [],
        "needs_confirmation": False,
        "download_ready": False,
        "dev_log": result.get("errors", []),
        "work_id": work_id,
    }


def _format_save_error_message(segment_index: str) -> str:
    if segment_index:
        return f"保存エラー。本文番号 {segment_index} の読み指定を確認してください。"
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
            "display_scope": "header",
            "card_anchor": "manuscript-card",
        }
    if import_notice == "legacy_imported":
        return {
            "status": "success",
            "title": "旧PEF原稿の取り込み結果",
            "message": "旧PEF原稿を取り込みました。",
            "errors": [],
            "warnings": [],
            "display_scope": "header",
            "card_anchor": "manuscript-card",
        }
    return None


def _status_notice_result(status_notice: str | None) -> dict | None:
    notices = {
        "dictionary_finalized": {
            "message": "辞書を確定し、編集用データを作成しました。",
            "card_anchor": "dictionary-card",
        },
        "empty_dictionary_processed": {
            "message": "辞書を使わずに編集用データを作成しました。",
            "card_anchor": "dictionary-card",
        },
        "final_saved": {
            "message": "編集完了として確定しました。",
            "card_anchor": "reading-edit-card",
        },
    }
    notice = notices.get(str(status_notice or ""))
    if not notice:
        return None
    return {
        "status": "success",
        "message": notice["message"],
        "card_anchor": notice["card_anchor"],
    }


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
