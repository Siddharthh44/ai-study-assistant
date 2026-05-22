from __future__ import annotations

import hashlib
import json
import logging
import os
import secrets
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import markdown
from passlib.context import CryptContext
from fastapi import (
    BackgroundTasks,
    Depends,
    FastAPI,
    File,
    Form,
    HTTPException,
    Request,
    UploadFile,
)
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import delete, desc, func, or_, select
from sqlalchemy.orm import Session
from starlette.middleware.sessions import SessionMiddleware

from backend.database import configure_database, get_db, init_database, session_scope
from backend.models import AnalyticsEvent, Bookmark, ExportRecord, Flashcard, HistoryEvent, Note, Quiz, QuizAttempt, QuizQuestion, StudySession, UploadedFile, User, UserSetting
from backend.services.export_service import build_json_export, build_pdf_export, build_text_export
from backend.services.summary_service import ContentGenerationError, generate_full_content, generate_summary
from backend.services.text_extraction import extract_upload_text, normalize_tags


BASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_SESSION_SECRET = os.getenv("SESSION_SECRET", "study-assistant-dev-secret")
PASSWORD_CONTEXT = CryptContext(schemes=["bcrypt"], deprecated="auto")
logger = logging.getLogger(__name__)

# Compatibility exports retained for older tests and callers.
processed_results: dict[str, dict[str, Any]] = {}


class SummaryRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str = Field(..., min_length=1)
    mode: str = "short"


class SummaryResponse(BaseModel):
    summary: str
    key_points: list[str]


class SignupPayload(BaseModel):
    full_name: str = Field(..., min_length=1)
    email: str = Field(..., min_length=3)
    password: str = Field(..., min_length=8)


class LoginPayload(BaseModel):
    email: str = Field(..., min_length=3)
    password: str = Field(..., min_length=1)


class BookmarkPayload(BaseModel):
    content_type: str
    content_id: int


class NoteUpdatePayload(BaseModel):
    title: str | None = None
    bookmarked: bool | None = None


class FlashcardReviewPayload(BaseModel):
    rating: str


class QuizAttemptPayload(BaseModel):
    answers: dict[str, int]
    completion_seconds: int = 0


class SettingsPayload(BaseModel):
    full_name: str | None = None
    institution: str | None = None
    exam: str | None = None
    exam_date: str | None = None
    theme: str | None = None
    spaced_repetition: bool | None = None
    show_hints: bool | None = None
    progress_tracking: bool | None = None
    note_length: str | None = None
    ai_tone: str | None = None
    auto_flashcards: bool | None = None
    auto_quiz: bool | None = None
    questions_per_quiz: int | None = None
    difficulty: int | None = None
    reminders: bool | None = None
    reminder_time: str | None = None
    export_format: str | None = None
    export_font_size: str | None = None
    export_include_cover: bool | None = None
    export_include_toc: bool | None = None
    export_include_concepts: bool | None = None
    export_include_pages: bool | None = None
    export_header: str | None = None


def create_app(config: dict[str, Any] | None = None) -> FastAPI:
    config = config or {}
    database_url = config.get("database_url") or os.getenv("DATABASE_URL")
    configure_database(database_url)
    init_database()

    session_secret = config.get("session_secret") or os.getenv("SESSION_SECRET")
    if not session_secret:
        if not config.get("testing") and os.getenv("VERCEL"):
            raise RuntimeError("SESSION_SECRET must be set for production deployments.")
        session_secret = DEFAULT_SESSION_SECRET

    session_same_site = config.get("session_same_site") or os.getenv("SESSION_SAME_SITE", "lax")
    session_max_age = int(config.get("session_max_age") or os.getenv("SESSION_MAX_AGE_SECONDS", 60 * 60 * 24 * 14))
    session_https_only = config.get("session_https_only")
    if session_https_only is None:
        session_https_only = os.getenv("SESSION_HTTPS_ONLY", "").strip().lower() in {"1", "true", "yes", "on"} or bool(os.getenv("VERCEL"))

    app = FastAPI()
    app.state.config = {
        "process_inline": config.get("process_inline", False),
        "testing": config.get("testing", False),
        "database_url": database_url,
        "session_same_site": session_same_site,
        "session_max_age": session_max_age,
        "session_https_only": session_https_only,
    }
    app.state.templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))
    app.state.templates.env.globals["now"] = datetime.now

    app.add_middleware(
        SessionMiddleware,
        secret_key=session_secret,
        same_site=session_same_site,
        https_only=session_https_only,
        max_age=session_max_age,
        session_cookie="study_assistant_session",
    )
    app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")

    register_routes(app)
    return app


def register_routes(app: FastAPI) -> None:
    templates: Jinja2Templates = app.state.templates

    def render_template(request, name, context=None):
        context = context or {}

        if "page_data" not in context:
            context["page_data"] = {}
        if "shell_data" not in context:
            context["shell_data"] = {}

        return templates.TemplateResponse(
            request=request,
            name=name,
            context=context
        )

    def render_app_template(request: Request, db: Session, user: User, name: str, page_data: dict[str, Any], page_title: str):
        return render_template(
            request,
            name,
            {
                "page_data": page_data,
                "shell_data": build_shell_payload(db, user, page_title),
            },
        )

    def require_html_user(request: Request, db: Session = Depends(get_db)) -> User:
        user = get_current_user(request, db)
        if user is None:
            raise RedirectException("/login")
        return user

    def require_api_user(request: Request, db: Session = Depends(get_db)) -> User:
        user = get_current_user(request, db)
        if user is None:
            raise HTTPException(status_code=401, detail="Authentication required")
        return user

    @app.exception_handler(RedirectException)
    async def redirect_exception_handler(_: Request, exc: "RedirectException") -> RedirectResponse:
        return RedirectResponse(url=exc.location, status_code=303)

    @app.get("/health")
    def health_check() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/summarize", response_model=SummaryResponse)
    def summarize(request: SummaryRequest) -> SummaryResponse:
        result = generate_summary(request.text, request.mode)
        if result.get("error"):
            raise HTTPException(status_code=500, detail=result["error"])
        return SummaryResponse(summary=result["summary"], key_points=result.get("key_points", []))

    @app.get("/", response_class=HTMLResponse)
    def home(request: Request, db: Session = Depends(get_db)):
        user = get_current_user(request, db)
        if user:
            return RedirectResponse(url="/dashboard", status_code=303)
        return RedirectResponse(url="/login", status_code=303)

    @app.get("/login", response_class=HTMLResponse)
    def login_page(request: Request):
        return render_template(request, "auth.html", {"page_data": {"initial_mode": "login"}})

    @app.get("/signup", response_class=HTMLResponse)
    def signup_page(request: Request):
        return render_template(request, "auth.html", {"page_data": {"initial_mode": "signup"}})

    @app.post("/api/auth/signup")
    def signup(payload: SignupPayload, request: Request, db: Session = Depends(get_db)):
        existing = db.scalar(select(User).where(User.email == payload.email.lower()))
        if existing:
            raise HTTPException(status_code=409, detail="An account with that email already exists.")

        user = User(
            full_name=payload.full_name.strip(),
            email=payload.email.lower().strip(),
            password_hash=hash_password(payload.password),
        )
        db.add(user)
        db.flush()
        db.add(UserSetting(user_id=user.id))
        db.commit()
        db.refresh(user)
        request.session["user_id"] = user.id
        return JSONResponse(
            status_code=201,
            content={
                "user": {"id": user.id, "full_name": user.full_name, "email": user.email},
                "redirect_url": "/dashboard",
            },
        )

    @app.post("/api/auth/login")
    def login(payload: LoginPayload, request: Request, db: Session = Depends(get_db)):
        user = db.scalar(select(User).where(User.email == payload.email.lower().strip()))
        if user is None or not verify_and_upgrade_password(db, user, payload.password):
            raise HTTPException(status_code=401, detail="Incorrect email or password.")
        request.session["user_id"] = user.id
        return {"redirect_url": "/dashboard", "user": {"id": user.id, "full_name": user.full_name, "email": user.email}}

    @app.post("/logout")
    def logout(request: Request) -> RedirectResponse:
        request.session.clear()
        return RedirectResponse(url="/login", status_code=303)

    @app.get("/dashboard", response_class=HTMLResponse)
    def dashboard_page(request: Request, user: User = Depends(require_html_user), db: Session = Depends(get_db)):
        return render_app_template(request, db, user, "dashboard.html", build_dashboard_payload(db, user.id), "Dashboard")

    @app.get("/upload", response_class=HTMLResponse)
    def upload_page(request: Request, user: User = Depends(require_html_user), db: Session = Depends(get_db)):
        return render_app_template(request, db, user, "upload.html", build_upload_payload(db, user.id), "Add New Study Material")

    @app.get("/processing", response_class=HTMLResponse)
    def processing_page(request: Request, note_id: int | None = None, result_id: str | None = None, user: User = Depends(require_html_user), db: Session = Depends(get_db)):
        resolved_note_id = note_id or parse_legacy_result_id(result_id)
        page_data = build_processing_payload(db, user.id, resolved_note_id)
        return render_template(request, "processing.html", {"page_data": page_data})

    @app.get("/notes", response_class=HTMLResponse)
    def notes_page(request: Request, user: User = Depends(require_html_user), db: Session = Depends(get_db)):
        return render_app_template(request, db, user, "notespage.html", build_notes_page_payload(db, user.id), "My Notes")

    @app.get("/notes/{note_id}", response_class=HTMLResponse)
    def note_page(request: Request, note_id: int, user: User = Depends(require_html_user), db: Session = Depends(get_db)):
        note = require_note(db, user.id, note_id)
        record_history(db, user.id, "viewed", "note", note.id, note.title, note.subject)
        record_analytics(db, user.id, "note_viewed", "note", note.id, note.subject, note_id=note.id)
        note.last_viewed_at = datetime.now(UTC).replace(tzinfo=None)
        db.commit()
        return render_app_template(request, db, user, "noteview.html", build_note_detail_payload(db, user.id, note), "View Note")

    @app.get("/flashcards", response_class=HTMLResponse)
    def flashcards_page(request: Request, note_id: int | None = None, result_id: str | None = None, user: User = Depends(require_html_user), db: Session = Depends(get_db)):
        resolved_note_id = note_id or parse_legacy_result_id(result_id)
        return render_app_template(request, db, user, "flashcards.html", build_flashcards_payload(db, user.id, resolved_note_id), "Flashcards")

    @app.get("/quizzes", response_class=HTMLResponse)
    def quizzes_page(request: Request, user: User = Depends(require_html_user), db: Session = Depends(get_db)):
        return render_app_template(request, db, user, "quizzes.html", build_quizzes_page_payload(db, user.id), "Quizzes")

    @app.get("/quiz/{quiz_id}", response_class=HTMLResponse)
    def quiz_page(request: Request, quiz_id: int, user: User = Depends(require_html_user), db: Session = Depends(get_db)):
        quiz = require_quiz(db, user.id, quiz_id)
        return render_app_template(request, db, user, "quiz.html", build_quiz_payload(db, user.id, quiz), "Quiz")

    @app.get("/quiz-results", response_class=HTMLResponse)
    def quiz_results_page(request: Request, attempt_id: int | None = None, quiz_id: int | None = None, result_id: str | None = None, note_id: int | None = None, user: User = Depends(require_html_user), db: Session = Depends(get_db)):
        payload = build_quiz_results_payload(db, user.id, attempt_id=attempt_id, quiz_id=quiz_id, note_id=note_id or parse_legacy_result_id(result_id))
        return render_app_template(request, db, user, "quizresults.html", payload, "Quiz Results")

    @app.get("/history", response_class=HTMLResponse)
    def history_page(request: Request, user: User = Depends(require_html_user), db: Session = Depends(get_db)):
        return render_app_template(request, db, user, "history.html", build_history_payload(db, user.id), "History")

    @app.get("/bookmarks", response_class=HTMLResponse)
    def bookmarks_page(request: Request, user: User = Depends(require_html_user), db: Session = Depends(get_db)):
        return render_app_template(request, db, user, "bookmarks.html", build_bookmarks_payload(db, user.id), "Bookmarks")

    @app.get("/progress", response_class=HTMLResponse)
    def progress_page(request: Request, user: User = Depends(require_html_user), db: Session = Depends(get_db)):
        return render_app_template(request, db, user, "progress.html", build_progress_payload(db, user.id), "Your Progress")

    @app.get("/export", response_class=HTMLResponse)
    def export_page(request: Request, note_id: int | None = None, result_id: str | None = None, user: User = Depends(require_html_user), db: Session = Depends(get_db)):
        resolved_note_id = note_id or parse_legacy_result_id(result_id)
        return render_app_template(request, db, user, "export.html", build_export_page_payload(db, user.id, resolved_note_id), "Export")

    @app.get("/settings", response_class=HTMLResponse)
    def settings_page(request: Request, user: User = Depends(require_html_user), db: Session = Depends(get_db)):
        return render_app_template(request, db, user, "settings.html", build_settings_payload(db, user), "Settings")

    # Compatibility routes preserved.
    @app.get("/noteview", response_class=HTMLResponse)
    def legacy_note_page(request: Request, result_id: str | None = None, user: User = Depends(require_html_user), db: Session = Depends(get_db)):
        note_id = parse_legacy_result_id(result_id)
        if note_id is None:
            raise RedirectException("/notes")
        return note_page(request, note_id, user, db)

    @app.get("/quiz", response_class=HTMLResponse)
    def legacy_quiz_page(request: Request, result_id: str | None = None, user: User = Depends(require_html_user), db: Session = Depends(get_db)):
        note_id = parse_legacy_result_id(result_id)
        if note_id is None:
            raise RedirectException("/quizzes")
        note = require_note(db, user.id, note_id)
        if note.quiz is None:
            raise RedirectException("/quizzes")
        return quiz_page(request, note.quiz.id, user, db)

    @app.get("/quizresults", response_class=HTMLResponse)
    @app.get("/quiz-result", response_class=HTMLResponse)
    def legacy_quiz_results_page(request: Request, result_id: str | None = None, attempt_id: int | None = None, user: User = Depends(require_html_user), db: Session = Depends(get_db)):
        return quiz_results_page(request, attempt_id=attempt_id, note_id=parse_legacy_result_id(result_id), user=user, db=db)

    @app.get("/result", response_class=HTMLResponse)
    def legacy_result_page(request: Request, result_id: str | None = None, user: User = Depends(require_html_user), db: Session = Depends(get_db)):
        note_id = parse_legacy_result_id(result_id)
        if note_id is None:
            raise RedirectException("/notes")
        return note_page(request, note_id, user, db)

    @app.get("/mynotes", response_class=HTMLResponse)
    def legacy_notes_page(request: Request, user: User = Depends(require_html_user), db: Session = Depends(get_db)):
        return notes_page(request, user, db)

    @app.get("/pyqanalysis", response_class=HTMLResponse)
    def pyqanalysis_page(request: Request, user: User = Depends(require_html_user)):
        return RedirectResponse(url="/upload", status_code=303)

    @app.get("/api/dashboard")
    def dashboard_api(user: User = Depends(require_api_user), db: Session = Depends(get_db)):
        return build_dashboard_payload(db, user.id)

    @app.get("/api/processing/{note_id}")
    def processing_status_api(note_id: int, user: User = Depends(require_api_user), db: Session = Depends(get_db)):
        note = require_note(db, user.id, note_id)
        return {"note_id": note.id, "status": note.status, "title": note.title, "error_message": note.error_message}

    @app.post("/process")
    async def process_material(
        request: Request,
        background_tasks: BackgroundTasks,
        source_type: str = Form("text"),
        text: str | None = Form(default=None),
        subject: str | None = Form(default=None),
        tags: str | None = Form(default=None),
        file: UploadFile | None = File(default=None),
        user: User = Depends(require_api_user),
        db: Session = Depends(get_db),
    ):
        content_text, source_name, normalized_source_type, upload_metadata = await resolve_source_payload(source_type, text, file)
        note = Note(
            user_id=user.id,
            title=(subject or source_name or "Untitled Note").strip(),
            subject=(subject or "General").strip(),
            source_type=normalized_source_type,
            source_name=source_name,
            source_text=content_text,
            tags_json=json.dumps(normalize_tags(tags)),
            status="processing",
        )
        db.add(note)
        db.flush()
        if upload_metadata is not None and source_name is not None:
            db.add(
                UploadedFile(
                    user_id=user.id,
                    note_id=note.id,
                    source_type=str(upload_metadata.get("source_type") or normalized_source_type),
                    original_filename=source_name,
                    media_type=upload_metadata.get("media_type"),
                    size_bytes=int(upload_metadata.get("size_bytes") or 0),
                    extracted_char_count=int(upload_metadata.get("extracted_char_count") or 0),
                    checksum=str(upload_metadata.get("checksum")) if upload_metadata.get("checksum") else None,
                    status="processed",
                )
            )
        db.commit()
        db.refresh(note)

        if app.state.config["process_inline"]:
            run_processing_pipeline(note.id)
        else:
            background_tasks.add_task(run_processing_pipeline, note.id)

        return JSONResponse(
            status_code=202,
            content={
                "note_id": note.id,
                "result_id": str(note.id),
                "status": "processing",
                "redirect_url": f"/processing?note_id={note.id}",
            },
        )

    @app.get("/api/notes")
    def notes_api(
        search: str | None = None,
        subject: str | None = None,
        sort: str = "newest",
        user: User = Depends(require_api_user),
        db: Session = Depends(get_db),
    ):
        return {"items": list_notes_payload(db, user.id, search=search, subject=subject, sort=sort)}

    @app.get("/api/notes/{note_id}")
    def note_detail_api(note_id: int, user: User = Depends(require_api_user), db: Session = Depends(get_db)):
        note = require_note(db, user.id, note_id)
        payload = build_note_detail_payload(db, user.id, note)
        return {**payload["note"], "flashcards": payload["flashcards"], "quiz": payload["quiz"]}

    @app.patch("/api/notes/{note_id}")
    def update_note_api(note_id: int, payload: NoteUpdatePayload, user: User = Depends(require_api_user), db: Session = Depends(get_db)):
        note = require_note(db, user.id, note_id)
        if payload.title is not None:
            note.title = payload.title.strip() or note.title
        db.commit()
        db.refresh(note)
        return build_note_detail_payload(db, user.id, note)

    @app.delete("/api/notes/{note_id}")
    def delete_note_api(note_id: int, user: User = Depends(require_api_user), db: Session = Depends(get_db)):
        note = require_note(db, user.id, note_id)
        db.delete(note)
        db.commit()
        return Response(status_code=204)

    @app.post("/api/bookmarks/toggle")
    def toggle_bookmark_api(payload: BookmarkPayload, user: User = Depends(require_api_user), db: Session = Depends(get_db)):
        existing = db.scalar(
            select(Bookmark).where(
                Bookmark.user_id == user.id,
                Bookmark.content_type == payload.content_type,
                Bookmark.content_id == payload.content_id,
            )
        )
        if existing:
            db.delete(existing)
            bookmarked = False
        else:
            db.add(Bookmark(user_id=user.id, content_type=payload.content_type, content_id=payload.content_id))
            bookmarked = True

        db.commit()
        title = lookup_content_title(db, payload.content_type, payload.content_id)
        record_history(db, user.id, "bookmarked", payload.content_type, payload.content_id, title, f"{'Saved' if bookmarked else 'Removed'} bookmark")
        record_analytics(
            db,
            user.id,
            "bookmark_toggled",
            payload.content_type,
            payload.content_id,
            lookup_content_subject(db, payload.content_type, payload.content_id),
            note_id=payload.content_id if payload.content_type == "note" else None,
            value=1 if bookmarked else -1,
            metadata={"bookmarked": bookmarked},
        )
        db.commit()
        return {"bookmarked": bookmarked}

    @app.get("/api/bookmarks")
    def bookmarks_api(user: User = Depends(require_api_user), db: Session = Depends(get_db)):
        return build_bookmarks_payload(db, user.id)

    @app.get("/api/flashcards")
    def flashcards_api(note_id: int | None = None, user: User = Depends(require_api_user), db: Session = Depends(get_db)):
        return build_flashcards_payload(db, user.id, note_id)

    @app.post("/api/flashcards/{flashcard_id}/review")
    def review_flashcard_api(flashcard_id: int, payload: FlashcardReviewPayload, user: User = Depends(require_api_user), db: Session = Depends(get_db)):
        flashcard = db.scalar(
            select(Flashcard)
            .join(Note, Flashcard.note_id == Note.id)
            .where(Flashcard.id == flashcard_id, Note.user_id == user.id)
        )
        if flashcard is None:
            raise HTTPException(status_code=404, detail="Flashcard not found.")

        rating = payload.rating.lower()
        if rating not in {"still", "almost", "got"}:
            raise HTTPException(status_code=400, detail="Invalid rating.")

        flashcard.review_count += 1
        flashcard.last_rating = rating
        flashcard.last_reviewed_at = datetime.now(UTC).replace(tzinfo=None)
        if rating == "got":
            flashcard.got_it_count += 1
            flashcard.mastery_level += 2
            flashcard.next_review_at = datetime.now(UTC).replace(tzinfo=None) + timedelta(days=3)
        elif rating == "almost":
            flashcard.almost_count += 1
            flashcard.mastery_level += 1
            flashcard.next_review_at = datetime.now(UTC).replace(tzinfo=None) + timedelta(days=1)
        else:
            flashcard.still_learning_count += 1
            flashcard.mastery_level = max(flashcard.mastery_level - 1, 0)
            flashcard.next_review_at = datetime.now(UTC).replace(tzinfo=None) + timedelta(hours=12)

        record_history(db, user.id, "flashcard_reviewed", "flashcard", flashcard.id, flashcard.question, rating)
        record_study_session(
            db,
            user_id=user.id,
            session_type="flashcard_review",
            note_id=flashcard.note_id,
            flashcard_id=flashcard.id,
            duration_seconds=0,
            metrics={"rating": rating, "mastery_level": flashcard.mastery_level},
        )
        record_analytics(db, user.id, "flashcard_reviewed", "flashcard", flashcard.id, flashcard.note.subject, note_id=flashcard.note_id, value=flashcard.mastery_level)
        db.commit()
        return serialize_flashcard(flashcard, bookmarked=is_bookmarked(db, user.id, "flashcard", flashcard.id))

    @app.get("/api/quizzes")
    def quizzes_api(user: User = Depends(require_api_user), db: Session = Depends(get_db)):
        return build_quizzes_page_payload(db, user.id)

    @app.get("/api/quizzes/{quiz_id}")
    def quiz_api(quiz_id: int, user: User = Depends(require_api_user), db: Session = Depends(get_db)):
        quiz = require_quiz(db, user.id, quiz_id)
        return build_quiz_payload(db, user.id, quiz)

    @app.post("/api/quizzes/{quiz_id}/attempts")
    def submit_quiz_attempt_api(quiz_id: int, payload: QuizAttemptPayload, user: User = Depends(require_api_user), db: Session = Depends(get_db)):
        quiz = require_quiz(db, user.id, quiz_id)
        result = evaluate_quiz_attempt(quiz, payload.answers, payload.completion_seconds)
        attempt = QuizAttempt(
            quiz_id=quiz.id,
            user_id=user.id,
            answers_json=json.dumps(payload.answers),
            score=result["score"],
            total_questions=result["total_questions"],
            percent=result["percent"],
            completion_seconds=result["completion_seconds"],
            wrong_answers_json=json.dumps(result["wrong_answers"]),
            weak_topics_json=json.dumps(result["weak_topics"]),
            recommendations_json=json.dumps(result["recommendations"]),
            strengths_json=json.dumps(result["strengths"]),
            weaknesses_json=json.dumps(result["weaknesses"]),
        )
        db.add(attempt)
        quiz.weak_topics_json = json.dumps(result["weak_topics"])
        record_history(db, user.id, "quiz_completed", "quiz", quiz.id, quiz.title, f"{result['percent']}%")
        record_study_session(
            db,
            user_id=user.id,
            session_type="quiz_attempt",
            note_id=quiz.note_id,
            quiz_id=quiz.id,
            duration_seconds=result["completion_seconds"],
            metrics={"score": result["score"], "percent": result["percent"], "total_questions": result["total_questions"]},
        )
        record_analytics(db, user.id, "quiz_completed", "quiz", quiz.id, quiz.note.subject, note_id=quiz.note_id, quiz_id=quiz.id, value=result["percent"])
        db.commit()
        db.refresh(attempt)
        return JSONResponse(status_code=201, content={"attempt_id": attempt.id, "result": serialize_attempt(attempt)})

    @app.get("/api/quiz-attempts/{attempt_id}")
    def quiz_attempt_api(attempt_id: int, user: User = Depends(require_api_user), db: Session = Depends(get_db)):
        attempt = db.scalar(select(QuizAttempt).join(Quiz).join(Note).where(QuizAttempt.id == attempt_id, Note.user_id == user.id))
        if attempt is None:
            raise HTTPException(status_code=404, detail="Quiz attempt not found.")
        return serialize_attempt(attempt)

    @app.get("/api/history")
    def history_api(user: User = Depends(require_api_user), db: Session = Depends(get_db)):
        return build_history_payload(db, user.id)

    @app.get("/api/progress")
    def progress_api(user: User = Depends(require_api_user), db: Session = Depends(get_db)):
        return build_progress_payload(db, user.id)

    @app.get("/api/settings")
    def settings_api(user: User = Depends(require_api_user), db: Session = Depends(get_db)):
        return build_settings_payload(db, user)

    @app.post("/api/settings")
    def update_settings_api(payload: SettingsPayload, user: User = Depends(require_api_user), db: Session = Depends(get_db)):
        settings = get_or_create_settings(db, user)
        apply_settings_payload(user, settings, payload)
        db.commit()
        return build_settings_payload(db, user)

    @app.post("/api/settings/reset-progress")
    def reset_progress_api(user: User = Depends(require_api_user), db: Session = Depends(get_db)):
        note_ids = db.scalars(select(Note.id).where(Note.user_id == user.id)).all()
        quiz_ids = db.scalars(select(Quiz.id).join(Note).where(Note.user_id == user.id)).all()
        if quiz_ids:
            db.execute(delete(QuizAttempt).where(QuizAttempt.user_id == user.id))
        db.execute(delete(StudySession).where(StudySession.user_id == user.id))
        db.execute(delete(AnalyticsEvent).where(AnalyticsEvent.user_id == user.id))
        if note_ids:
            db.execute(delete(Bookmark).where(Bookmark.user_id == user.id))
            db.execute(delete(HistoryEvent).where(HistoryEvent.user_id == user.id))
            flashcards = db.scalars(select(Flashcard).where(Flashcard.note_id.in_(note_ids))).all()
            for flashcard in flashcards:
                flashcard.review_count = 0
                flashcard.got_it_count = 0
                flashcard.almost_count = 0
                flashcard.still_learning_count = 0
                flashcard.mastery_level = 0
                flashcard.last_rating = None
                flashcard.last_reviewed_at = None
                flashcard.next_review_at = None
        db.commit()
        return {"status": "reset"}

    @app.get("/export/json/{item_id}")
    def export_json(item_id: int, content_type: str = "note", user: User = Depends(require_api_user), db: Session = Depends(get_db)):
        document, filename = build_export_document(db, user.id, content_type, item_id)
        store_export(db, user.id, content_type, item_id, "json")
        return Response(
            content=build_json_export(document),
            media_type="application/json",
            headers={"Content-Disposition": f'attachment; filename="{filename}.json"'},
        )

    @app.get("/export/txt/{item_id}")
    def export_txt(item_id: int, content_type: str = "note", user: User = Depends(require_api_user), db: Session = Depends(get_db)):
        document, filename = build_export_document(db, user.id, content_type, item_id)
        store_export(db, user.id, content_type, item_id, "txt")
        return Response(
            content=build_text_export(document),
            media_type="text/plain; charset=utf-8",
            headers={"Content-Disposition": f'attachment; filename="{filename}.txt"'},
        )

    @app.get("/export/pdf/{item_id}")
    def export_pdf(item_id: int, content_type: str = "note", user: User = Depends(require_api_user), db: Session = Depends(get_db)):
        document, filename = build_export_document(db, user.id, content_type, item_id)
        settings = get_or_create_settings(db, user)
        store_export(db, user.id, content_type, item_id, "pdf")
        pdf_bytes = build_pdf_export(document, header=settings.export_header or document["note"]["title"])
        return StreamingResponse(
            iter([pdf_bytes]),
            media_type="application/pdf",
            headers={"Content-Disposition": f'attachment; filename="{filename}.pdf"'},
        )


class RedirectException(Exception):
    def __init__(self, location: str):
        self.location = location


def hash_password(password: str) -> str:
    return PASSWORD_CONTEXT.hash(password)


def verify_password(password: str, password_hash: str) -> bool:
    if is_legacy_password_hash(password_hash):
        return verify_legacy_password(password, password_hash)

    try:
        return PASSWORD_CONTEXT.verify(password, password_hash)
    except ValueError:
        return False


def hash_legacy_password(password: str) -> str:
    salt = secrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt.encode("utf-8"), 390000)
    return f"{salt}${digest.hex()}"


def verify_legacy_password(password: str, password_hash: str) -> bool:
    try:
        salt, stored_hash = password_hash.split("$", 1)
    except ValueError:
        return False
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt.encode("utf-8"), 390000)
    return secrets.compare_digest(digest.hex(), stored_hash)


def is_legacy_password_hash(password_hash: str) -> bool:
    return password_hash.count("$") == 1 and not password_hash.startswith("$2")


def verify_and_upgrade_password(db: Session, user: User, password: str) -> bool:
    if verify_password(password, user.password_hash):
        if is_legacy_password_hash(user.password_hash):
            user.password_hash = hash_password(password)
            db.commit()
            db.refresh(user)
        return True
    return False


def get_current_user(request: Request, db: Session) -> User | None:
    user_id = request.session.get("user_id")
    if not user_id:
        return None
    return db.get(User, int(user_id))


def parse_legacy_result_id(result_id: str | None) -> int | None:
    if result_id is None:
        return None
    try:
        return int(result_id)
    except ValueError:
        return None


async def resolve_source_payload(source_type: str, text: str | None, file: UploadFile | None) -> tuple[str, str | None, str, dict[str, Any] | None]:
    normalized = source_type.lower().strip()
    if normalized == "text":
        if not text or not text.strip():
            raise HTTPException(status_code=400, detail="Paste some study material before submitting.")
        return text.strip(), None, "text", None

    if normalized not in {"file", "image", "audio", "pyq"}:
        raise HTTPException(status_code=400, detail="Unsupported source type.")

    if file is None:
        raise HTTPException(status_code=400, detail="Please attach a file to continue.")

    try:
        extracted, filename, metadata = await extract_upload_text(file)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    metadata["source_type"] = normalized
    return extracted, filename, "file", metadata


def run_processing_pipeline(note_id: int) -> None:
    with session_scope() as db:
        note = db.get(Note, note_id)
        if note is None:
            return

        try:
            result = generate_full_content(note.source_text)
            persist_generated_content(db, note, result)
            record_history(db, note.user_id, "generated", "note", note.id, note.title, note.subject)
            record_analytics(db, note.user_id, "note_generated", "note", note.id, note.subject, note_id=note.id, value=len(result.get("flashcards") or []))
        except ContentGenerationError as exc:
            logger.warning("Content generation failed for note_id=%s: %s", note_id, exc)
            note.status = "failed"
            note.error_message = str(exc)
        except Exception as exc:  # pragma: no cover - defensive
            logger.exception("Unexpected processing failure for note_id=%s", note_id)
            note.status = "failed"
            note.error_message = "Something went wrong while generating your study pack."


def persist_generated_content(db: Session, note: Note, result: dict[str, Any]) -> None:
    title = result.get("title") or note.title
    note.title = title
    note.summary = str(result.get("summary") or "")
    note.notes_markdown = str(result.get("notes") or result.get("summary") or "")
    note.key_concepts_json = json.dumps(result.get("key_concepts") or [])
    note.difficulty_analysis_json = json.dumps(
        result.get("difficulty_analysis") or derive_difficulty_analysis(result)
    )
    note.study_recommendations_json = json.dumps(
        result.get("study_recommendations") or derive_study_recommendations(result)
    )
    note.status = "ready"
    note.error_message = None

    db.query(Flashcard).filter(Flashcard.note_id == note.id).delete()
    if note.quiz is not None:
        db.delete(note.quiz)
        db.flush()

    flashcards = result.get("flashcards") or []
    for index, card in enumerate(flashcards):
        db.add(
            Flashcard(
                note_id=note.id,
                position=index,
                question=str(card.get("question") or f"Flashcard {index + 1}"),
                answer=str(card.get("answer") or ""),
                topic=card.get("topic"),
            )
        )

    quiz_questions = result.get("quiz") or []
    if quiz_questions:
        quiz = Quiz(note_id=note.id, title=f"{note.title} Quiz", duration_seconds=max(len(quiz_questions), 1) * 60)
        db.add(quiz)
        db.flush()
        for index, question in enumerate(quiz_questions):
            options = question.get("options") or []
            db.add(
                QuizQuestion(
                    quiz_id=quiz.id,
                    position=index,
                    prompt=str(question.get("question") or f"Question {index + 1}"),
                    options_json=json.dumps(options),
                    correct_index=get_correct_index(options, question.get("answer")),
                    explanation=str(question.get("explanation") or ""),
                    topic=question.get("topic") or note.subject,
                    difficulty=question.get("difficulty"),
                )
            )

    processed_results[str(note.id)] = result


def derive_difficulty_analysis(result: dict[str, Any]) -> dict[str, Any]:
    flashcards = result.get("flashcards") or []
    quiz = result.get("quiz") or []
    overall = "easy" if len(quiz) <= 5 else "medium" if len(quiz) <= 10 else "hard"
    return {
        "overall": overall,
        "challenging_topics": sorted({question.get("topic", "General") for question in quiz if question.get("topic")}),
        "flashcards_generated": len(flashcards),
    }


def derive_study_recommendations(result: dict[str, Any]) -> list[str]:
    recommendations = ["Review the generated notes once before attempting the quiz."]
    if result.get("flashcards"):
        recommendations.append("Use the flashcards for a spaced-repetition pass later today.")
    if result.get("quiz"):
        recommendations.append("Take the quiz after reviewing the summary to identify weak topics.")
    return recommendations


def get_correct_index(options: list[str], answer: Any) -> int:
    if isinstance(answer, int):
        return answer if 0 <= answer < len(options) else 0
    if isinstance(answer, str):
        for index, option in enumerate(options):
            if option == answer or option.startswith(answer):
                return index
    return 0


def require_note(db: Session, user_id: int, note_id: int) -> Note:
    note = db.scalar(select(Note).where(Note.id == note_id, Note.user_id == user_id))
    if note is None:
        raise HTTPException(status_code=404, detail="Note not found.")
    return note


def require_quiz(db: Session, user_id: int, quiz_id: int) -> Quiz:
    quiz = db.scalar(select(Quiz).join(Note).where(Quiz.id == quiz_id, Note.user_id == user_id))
    if quiz is None:
        raise HTTPException(status_code=404, detail="Quiz not found.")
    return quiz


def is_bookmarked(db: Session, user_id: int, content_type: str, content_id: int) -> bool:
    return db.scalar(
        select(Bookmark.id).where(
            Bookmark.user_id == user_id,
            Bookmark.content_type == content_type,
            Bookmark.content_id == content_id,
        )
    ) is not None


def humanize_delta(timestamp: datetime | None) -> str:
    if timestamp is None:
        return "Just now"
    now = datetime.now(UTC).replace(tzinfo=None)
    delta = now - timestamp
    if delta.days >= 7:
        return timestamp.strftime("%d %b %Y")
    if delta.days >= 1:
        return "Yesterday" if delta.days == 1 else f"{delta.days} days ago"
    hours = delta.seconds // 3600
    if hours:
        return f"{hours}h ago"
    minutes = max(delta.seconds // 60, 1)
    return f"{minutes}m ago"


def note_preview(note: Note) -> str:
    if note.summary:
        return note.summary
    cleaned = note.notes_markdown.replace("#", "").strip()
    return cleaned[:160] + ("..." if len(cleaned) > 160 else "")


def serialize_note_card(note: Note, *, bookmarked: bool = False) -> dict[str, Any]:
    return {
        "id": note.id,
        "title": note.title,
        "subject": note.subject,
        "date": humanize_delta(note.created_at),
        "preview": note_preview(note),
        "flashcards": len(note.flashcards),
        "hasQuiz": note.quiz is not None and len(note.quiz.questions) > 0,
        "status": note.status,
        "bookmarked": bookmarked,
        "updated_at": note.updated_at.isoformat(),
    }


def serialize_flashcard(flashcard: Flashcard, *, bookmarked: bool = False) -> dict[str, Any]:
    return {
        "id": flashcard.id,
        "question": flashcard.question,
        "answer": flashcard.answer,
        "topic": flashcard.topic or "General",
        "review_count": flashcard.review_count,
        "got_it_count": flashcard.got_it_count,
        "almost_count": flashcard.almost_count,
        "still_learning_count": flashcard.still_learning_count,
        "mastery_level": flashcard.mastery_level,
        "last_rating": flashcard.last_rating,
        "next_review_at": flashcard.next_review_at.isoformat() if flashcard.next_review_at else None,
        "bookmarked": bookmarked,
    }


def serialize_quiz(quiz: Quiz, *, include_answers: bool, include_attempts: bool = False) -> dict[str, Any]:
    questions = []
    for question in quiz.questions:
        payload = {
            "id": question.id,
            "question": question.prompt,
            "options": json.loads(question.options_json or "[]"),
            "topic": question.topic or "General",
            "difficulty": question.difficulty or "medium",
            "explanation": question.explanation or "",
        }
        if include_answers:
            options = json.loads(question.options_json or "[]")
            payload["correct_index"] = question.correct_index
            payload["correct_answer"] = options[question.correct_index] if 0 <= question.correct_index < len(options) else ""
        questions.append(payload)

    data = {
        "id": quiz.id,
        "title": quiz.title,
        "question_count": len(questions),
        "duration_seconds": quiz.duration_seconds,
        "questions": questions,
    }
    if include_attempts:
        data["attempts"] = [serialize_attempt(attempt) for attempt in quiz.attempts]
    return data


def serialize_attempt(attempt: QuizAttempt) -> dict[str, Any]:
    return {
        "id": attempt.id,
        "score": attempt.score,
        "total_questions": attempt.total_questions,
        "percent": attempt.percent,
        "completion_seconds": attempt.completion_seconds,
        "wrong_answers": json.loads(attempt.wrong_answers_json or "[]"),
        "weak_topics": json.loads(attempt.weak_topics_json or "[]"),
        "recommendations": json.loads(attempt.recommendations_json or "[]"),
        "strengths": json.loads(attempt.strengths_json or "[]"),
        "weaknesses": json.loads(attempt.weaknesses_json or "[]"),
        "created_at": attempt.created_at.isoformat(),
    }


def serialize_history_event(event: HistoryEvent) -> dict[str, Any]:
    return {
        "id": event.id,
        "event_type": event.event_type,
        "content_type": event.content_type,
        "content_id": event.content_id,
        "title": event.title,
        "description": event.description,
        "date": humanize_delta(event.created_at),
        "created_at": event.created_at.isoformat(),
        "metadata": json.loads(event.metadata_json or "{}"),
    }


def list_notes_payload(db: Session, user_id: int, *, search: str | None = None, subject: str | None = None, sort: str = "newest") -> list[dict[str, Any]]:
    query = select(Note).where(Note.user_id == user_id, Note.status != "failed")
    if search:
        pattern = f"%{search.lower()}%"
        query = query.where(or_(func.lower(Note.title).like(pattern), func.lower(Note.subject).like(pattern)))
    if subject and subject != "All":
        query = query.where(Note.subject == subject)
    order_column = Note.created_at.desc() if sort != "oldest" else Note.created_at.asc()
    notes = db.scalars(query.order_by(order_column)).all()
    return [serialize_note_card(note, bookmarked=is_bookmarked(db, user_id, "note", note.id)) for note in notes]


def build_dashboard_payload(db: Session, user_id: int) -> dict[str, Any]:
    notes = db.scalars(select(Note).where(Note.user_id == user_id, Note.status == "ready").order_by(Note.created_at.desc())).all()
    latest_note = notes[0] if notes else None
    quizzes = db.scalars(select(Quiz).join(Note).where(Note.user_id == user_id).order_by(Quiz.created_at.desc())).all()
    attempts = db.scalars(select(QuizAttempt).where(QuizAttempt.user_id == user_id).order_by(QuizAttempt.created_at.desc())).all()
    flashcards = db.scalars(select(Flashcard).join(Note).where(Note.user_id == user_id)).all()

    due_flashcards = sorted(
        [card for card in flashcards if card.next_review_at is None or card.next_review_at <= datetime.now(UTC).replace(tzinfo=None)],
        key=lambda card: card.next_review_at or card.created_at,
    )
    average_score = round(sum(attempt.percent for attempt in attempts) / len(attempts)) if attempts else 0
    progress = build_progress_payload(db, user_id)

    return {
        "greeting_name": get_user_first_name(db, user_id),
        "summary": {
            "notes_created": len(notes),
            "flashcards_due": len(due_flashcards),
            "quiz_average": average_score,
            "study_streak": progress["stats"]["study_streak"],
        },
        "recent_notes": [serialize_note_card(note, bookmarked=is_bookmarked(db, user_id, "note", note.id)) for note in notes[:3]],
        "revision_tasks": [
            {
                "note_id": card.note_id,
                "title": card.note.title,
                "cards": len(card.note.flashcards),
            }
            for card in due_flashcards[:3]
        ],
        "continue_note": serialize_note_card(latest_note, bookmarked=is_bookmarked(db, user_id, "note", latest_note.id)) if latest_note else None,
    }


def build_upload_payload(db: Session, user_id: int) -> dict[str, Any]:
    notes = db.scalars(select(Note).where(Note.user_id == user_id, Note.status == "ready").order_by(Note.created_at.desc())).all()
    subjects = sorted({note.subject for note in notes})
    return {"subjects": subjects}


def build_processing_payload(db: Session, user_id: int, note_id: int | None) -> dict[str, Any]:
    if note_id is None:
        return {"note_id": None, "status": "missing"}
    note = require_note(db, user_id, note_id)
    return {"note_id": note.id, "status": note.status, "title": note.title, "error_message": note.error_message}


def build_notes_page_payload(db: Session, user_id: int) -> dict[str, Any]:
    items = list_notes_payload(db, user_id)
    subjects = ["All", *sorted({item["subject"] for item in items})]
    return {"items": items, "subjects": subjects}


def build_note_detail_payload(db: Session, user_id: int, note: Note) -> dict[str, Any]:
    note_payload = serialize_note_card(note, bookmarked=is_bookmarked(db, user_id, "note", note.id))
    note_payload.update(
        {
            "summary": note.summary,
            "notes_markdown": note.notes_markdown,
            "notes_html": markdown.markdown(note.notes_markdown or "", extensions=["fenced_code", "tables"]),
            "key_concepts": json.loads(note.key_concepts_json or "[]"),
            "difficulty_analysis": json.loads(note.difficulty_analysis_json or "{}"),
            "study_recommendations": json.loads(note.study_recommendations_json or "[]"),
            "generated_at": note.created_at.strftime("%d %b %Y"),
        }
    )

    return {
        "note": note_payload,
        "flashcards": [serialize_flashcard(card, bookmarked=is_bookmarked(db, user_id, "flashcard", card.id)) for card in note.flashcards],
        "quiz": serialize_quiz(note.quiz, include_answers=True, include_attempts=True) if note.quiz else None,
    }


def build_flashcards_payload(db: Session, user_id: int, note_id: int | None) -> dict[str, Any]:
    note = resolve_note_for_flashcards(db, user_id, note_id)
    if note is None:
        return {"note_id": None, "note_title": "Flashcard Review", "cards": []}
    return {
        "note_id": note.id,
        "note_title": note.title,
        "cards": [serialize_flashcard(card, bookmarked=is_bookmarked(db, user_id, "flashcard", card.id)) for card in note.flashcards],
        "back_link": f"/notes/{note.id}",
    }


def resolve_note_for_flashcards(db: Session, user_id: int, note_id: int | None) -> Note | None:
    if note_id is not None:
        return require_note(db, user_id, note_id)
    return db.scalar(select(Note).where(Note.user_id == user_id, Note.status == "ready").order_by(Note.updated_at.desc()))


def build_quizzes_page_payload(db: Session, user_id: int) -> dict[str, Any]:
    quizzes = db.scalars(select(Quiz).join(Note).where(Note.user_id == user_id).order_by(Quiz.created_at.desc())).all()
    attempted = {attempt.quiz_id: attempt for attempt in db.scalars(select(QuizAttempt).where(QuizAttempt.user_id == user_id).order_by(QuizAttempt.created_at.desc())).all()}
    items = []
    for quiz in quizzes:
        latest_attempt = attempted.get(quiz.id)
        items.append(
            {
                "id": quiz.id,
                "title": quiz.note.title,
                "subject": quiz.note.subject,
                "questions": len(quiz.questions),
                "attempted": latest_attempt is not None,
                "last_score": latest_attempt.percent if latest_attempt else None,
                "date": humanize_delta(quiz.created_at),
            }
        )

    average_score = round(sum(item["last_score"] for item in items if item["last_score"] is not None) / max(sum(1 for item in items if item["last_score"] is not None), 1)) if any(item["last_score"] is not None for item in items) else 0
    time_saved_hours = round(
        sum(max(quiz.duration_seconds - (attempted.get(quiz.id).completion_seconds if attempted.get(quiz.id) else 0), 0) for quiz in quizzes) / 3600,
        1,
    )
    return {
        "items": items,
        "stats": {
            "total_quizzes": len(items),
            "average_score": average_score,
            "time_saved_hours": time_saved_hours,
        },
    }


def build_quiz_payload(db: Session, user_id: int, quiz: Quiz) -> dict[str, Any]:
    return {
        "quiz": serialize_quiz(quiz, include_answers=True),
        "subject": quiz.note.subject,
        "title": quiz.note.title,
        "results_url": "/quiz-results",
    }


def build_quiz_results_payload(db: Session, user_id: int, *, attempt_id: int | None = None, quiz_id: int | None = None, note_id: int | None = None) -> dict[str, Any]:
    attempt = None
    if attempt_id is not None:
        attempt = db.scalar(select(QuizAttempt).join(Quiz).join(Note).where(QuizAttempt.id == attempt_id, Note.user_id == user_id))
    elif quiz_id is not None:
        attempt = db.scalar(select(QuizAttempt).join(Quiz).join(Note).where(Quiz.id == quiz_id, Note.user_id == user_id).order_by(QuizAttempt.created_at.desc()))
    elif note_id is not None:
        attempt = db.scalar(
            select(QuizAttempt)
            .join(Quiz)
            .join(Note)
            .where(Note.id == note_id, Note.user_id == user_id)
            .order_by(QuizAttempt.created_at.desc())
        )

    if attempt is None:
        latest = db.scalar(select(QuizAttempt).where(QuizAttempt.user_id == user_id).order_by(QuizAttempt.created_at.desc()))
        if latest is None:
            return {
                "result": {
                    "score": 0,
                    "total_questions": 0,
                    "percent": 0,
                    "feedback": "Take a quiz to see your results here.",
                    "wrong_answers": [],
                    "weak_topics": [],
                    "recommendations": [],
                    "strengths": [],
                    "weaknesses": [],
                }
            }
        attempt = latest

    result = serialize_attempt(attempt)
    result["feedback"] = build_feedback(result["percent"])
    result["quiz_id"] = attempt.quiz_id
    result["note_id"] = attempt.quiz.note_id
    return {"result": result}


def build_feedback(percent: int) -> str:
    if percent >= 85:
        return "Excellent work. You are close to mastery."
    if percent >= 65:
        return "Good progress. A targeted review will lift your score quickly."
    return "A quick revision pass will help reinforce the weaker areas."


def build_history_payload(db: Session, user_id: int) -> dict[str, Any]:
    events = db.scalars(select(HistoryEvent).where(HistoryEvent.user_id == user_id).order_by(HistoryEvent.created_at.desc()).limit(100)).all()
    return {"items": [serialize_history_event(event) for event in events]}


def build_bookmarks_payload(db: Session, user_id: int) -> dict[str, Any]:
    bookmarks = db.scalars(select(Bookmark).where(Bookmark.user_id == user_id).order_by(Bookmark.created_at.desc())).all()
    items: list[dict[str, Any]] = []
    for bookmark in bookmarks:
        title = lookup_content_title(db, bookmark.content_type, bookmark.content_id)
        items.append(
            {
                "id": bookmark.id,
                "content_type": bookmark.content_type,
                "content_id": bookmark.content_id,
                "title": title,
                "subject": lookup_content_subject(db, bookmark.content_type, bookmark.content_id),
                "meta": lookup_content_meta(db, bookmark.content_type, bookmark.content_id),
                "content": lookup_content_preview(db, bookmark.content_type, bookmark.content_id),
            }
        )
    return {"items": items}


def build_progress_payload(db: Session, user_id: int) -> dict[str, Any]:
    notes_count = db.scalar(select(func.count()).select_from(Note).where(Note.user_id == user_id, Note.status == "ready")) or 0
    flashcards = db.scalars(select(Flashcard).join(Note).where(Note.user_id == user_id)).all()
    attempts = db.scalars(select(QuizAttempt).where(QuizAttempt.user_id == user_id).order_by(QuizAttempt.created_at.asc())).all()
    bookmarks_count = db.scalar(select(func.count()).select_from(Bookmark).where(Bookmark.user_id == user_id)) or 0

    average_score = round(sum(attempt.percent for attempt in attempts) / len(attempts)) if attempts else 0
    cards_reviewed = sum(card.review_count for card in flashcards)
    topic_scores = calculate_topic_scores(db, user_id)
    history_dates = {
        event.created_at.date()
        for event in db.scalars(select(HistoryEvent).where(HistoryEvent.user_id == user_id)).all()
    }
    streak = calculate_streak(history_dates)
    score_trend = [
        {"name": attempt.created_at.strftime("%a"), "score": attempt.percent}
        for attempt in attempts[-7:]
    ]
    focus_areas = sorted(topic_scores, key=lambda item: item["score"])[:3]
    recent_history = [serialize_attempt(attempt) | {"title": attempt.quiz.note.title} for attempt in attempts[-5:][::-1]]

    return {
        "stats": {
            "quizzes_taken": len(attempts),
            "average_score": average_score,
            "cards_reviewed": cards_reviewed,
            "study_streak": streak,
            "notes_generated": notes_count,
            "bookmarks": bookmarks_count,
        },
        "score_trend": score_trend,
        "topic_performance": topic_scores,
        "focus_areas": focus_areas,
        "recent_quiz_history": recent_history,
    }


def calculate_topic_scores(db: Session, user_id: int) -> list[dict[str, Any]]:
    attempts = db.scalars(select(QuizAttempt).where(QuizAttempt.user_id == user_id)).all()
    topic_totals: dict[str, list[int]] = {}
    for attempt in attempts:
        weak_topics = json.loads(attempt.weak_topics_json or "[]")
        strengths = json.loads(attempt.strengths_json or "[]")
        for topic in weak_topics:
            name = topic["name"] if isinstance(topic, dict) else str(topic)
            topic_totals.setdefault(name, []).append(max(0, int(topic.get("accuracy", 0)) if isinstance(topic, dict) else 0))
        for topic in strengths:
            topic_totals.setdefault(str(topic), []).append(100)

    if not topic_totals:
        notes = db.scalars(select(Note).where(Note.user_id == user_id)).all()
        return [{"name": note.subject, "score": 75, "fill": "#2D6A4F"} for note in notes[:5]]

    scores = []
    for name, values in topic_totals.items():
        score = round(sum(values) / len(values)) if values else 0
        fill = "#2D6A4F" if score >= 80 else "#52796F" if score >= 60 else "#E2E2E2"
        scores.append({"name": name, "score": score, "fill": fill})
    return scores


def calculate_streak(history_dates: set[date]) -> int:
    streak = 0
    current = date.today()
    while current in history_dates:
        streak += 1
        current -= timedelta(days=1)
    return streak


def build_export_page_payload(db: Session, user_id: int, note_id: int | None) -> dict[str, Any]:
    notes = db.scalars(select(Note).where(Note.user_id == user_id, Note.status == "ready").order_by(Note.created_at.desc())).all()
    items = []
    selected_ids = []
    for note in notes:
        items.append({"id": note.id, "title": note.title, "type": "Note", "content_type": "note", "selection_key": f"note:{note.id}"})
        selected_ids.append(f"note:{note.id}")
        if note.quiz:
            items.append({"id": note.quiz.id, "title": f"{note.title} Quiz", "type": "Quiz", "content_type": "quiz", "selection_key": f"quiz:{note.quiz.id}"})
    if note_id is not None:
        selected_ids = [f"note:{note_id}"]
    settings = db.scalar(select(UserSetting).where(UserSetting.user_id == user_id)) or UserSetting(user_id=user_id)
    return {
        "items": items,
        "selected_ids": selected_ids[:3],
        "settings": {
            "export_format": settings.export_format,
            "export_font_size": settings.export_font_size,
            "export_include_cover": settings.export_include_cover,
            "export_include_toc": settings.export_include_toc,
            "export_include_concepts": settings.export_include_concepts,
            "export_include_pages": settings.export_include_pages,
            "export_header": settings.export_header or f"{get_user_first_name(db, user_id)}'s Study Notes",
        },
    }


def build_settings_payload(db: Session, user: User) -> dict[str, Any]:
    settings = get_or_create_settings(db, user)
    return {
        "profile": {
            "name": user.full_name,
            "email": user.email,
            "institution": settings.institution or "",
            "exam": settings.exam or "",
            "exam_date": settings.exam_date or "",
        },
        "ai": {
            "note_length": settings.note_length,
            "tone": settings.ai_tone,
            "auto_flashcards": settings.auto_flashcards,
            "auto_quiz": settings.auto_quiz,
            "questions_per_quiz": settings.questions_per_quiz,
            "difficulty": settings.difficulty,
            "reminders": settings.reminders,
            "reminder_time": settings.reminder_time,
        },
        "learning": {
            "spaced_repetition": settings.spaced_repetition,
            "show_hints": settings.show_hints,
            "progress_tracking": settings.progress_tracking,
        },
        "export": {
            "format": settings.export_format,
            "font_size": settings.export_font_size,
            "include_cover": settings.export_include_cover,
            "include_toc": settings.export_include_toc,
            "include_concepts": settings.export_include_concepts,
            "include_pages": settings.export_include_pages,
            "header": settings.export_header or "",
        },
        "theme": settings.theme,
    }


def get_or_create_settings(db: Session, user: User) -> UserSetting:
    settings = db.scalar(select(UserSetting).where(UserSetting.user_id == user.id))
    if settings is None:
        settings = UserSetting(user_id=user.id)
        db.add(settings)
        db.flush()
    return settings


def apply_settings_payload(user: User, settings: UserSetting, payload: SettingsPayload) -> None:
    if payload.full_name is not None:
        user.full_name = payload.full_name
    for field_name in {
        "institution",
        "exam",
        "exam_date",
        "theme",
        "spaced_repetition",
        "show_hints",
        "progress_tracking",
        "note_length",
        "ai_tone",
        "auto_flashcards",
        "auto_quiz",
        "questions_per_quiz",
        "difficulty",
        "reminders",
        "reminder_time",
        "export_format",
        "export_font_size",
        "export_include_cover",
        "export_include_toc",
        "export_include_concepts",
        "export_include_pages",
        "export_header",
    }:
        value = getattr(payload, field_name, None)
        if value is not None:
            setattr(settings, field_name, value)


def evaluate_quiz_attempt(quiz: Quiz, answers: dict[str, int], completion_seconds: int) -> dict[str, Any]:
    score = 0
    wrong_answers: list[dict[str, Any]] = []
    topic_totals: dict[str, dict[str, int]] = {}
    strengths: set[str] = set()

    for question in quiz.questions:
        selected_index = answers.get(str(question.id))
        options = json.loads(question.options_json or "[]")
        topic = question.topic or "General"
        stats = topic_totals.setdefault(topic, {"correct": 0, "total": 0})
        stats["total"] += 1

        if selected_index == question.correct_index:
            score += 1
            stats["correct"] += 1
            strengths.add(topic)
            continue

        wrong_answers.append(
            {
                "question": question.prompt,
                "yours": options[selected_index] if isinstance(selected_index, int) and 0 <= selected_index < len(options) else "Not answered",
                "correct": options[question.correct_index] if 0 <= question.correct_index < len(options) else "",
                "explanation": question.explanation or "",
            }
        )

    total_questions = len(quiz.questions)
    percent = round((score / total_questions) * 100) if total_questions else 0
    weak_topics = []
    weaknesses = []
    for topic, stats in topic_totals.items():
        accuracy = round((stats["correct"] / stats["total"]) * 100) if stats["total"] else 0
        if accuracy < 100:
            weak_topics.append({"name": topic, "accuracy": accuracy})
        if accuracy < 70:
            weaknesses.append(topic)

    recommendations = [f"Revisit {topic['name']}." for topic in weak_topics[:3]]
    if not recommendations:
        recommendations.append("Keep your momentum with one more practice round.")

    return {
        "score": score,
        "total_questions": total_questions,
        "percent": percent,
        "completion_seconds": completion_seconds,
        "wrong_answers": wrong_answers,
        "weak_topics": weak_topics,
        "recommendations": recommendations,
        "strengths": sorted(strengths - set(weaknesses)),
        "weaknesses": weaknesses,
    }


def build_export_document(db: Session, user_id: int, content_type: str, item_id: int) -> tuple[dict[str, Any], str]:
    if content_type == "quiz":
        quiz = require_quiz(db, user_id, item_id)
        note = quiz.note
    else:
        note = require_note(db, user_id, item_id)
        quiz = note.quiz

    note_payload = build_note_detail_payload(db, user_id, note)
    filename = slugify(note.title)
    document = {
        "note": note_payload["note"],
        "flashcards": note_payload["flashcards"],
        "quiz": serialize_quiz(quiz, include_answers=True, include_attempts=True) if quiz else None,
    }
    return document, filename


def store_export(db: Session, user_id: int, content_type: str, item_id: int, export_format: str) -> None:
    db.add(ExportRecord(user_id=user_id, content_type=content_type, content_id=item_id, format=export_format))
    record_history(db, user_id, "exported", content_type, item_id, lookup_content_title(db, content_type, item_id), export_format.upper())
    record_analytics(db, user_id, "export_created", content_type, item_id, lookup_content_subject(db, content_type, item_id), note_id=item_id if content_type == "note" else None)
    db.commit()


def lookup_content_title(db: Session, content_type: str, content_id: int) -> str:
    if content_type == "note":
        note = db.get(Note, content_id)
        return note.title if note else "Note"
    if content_type == "flashcard":
        card = db.get(Flashcard, content_id)
        return card.question if card else "Flashcard"
    quiz = db.get(Quiz, content_id)
    return quiz.title if quiz else "Quiz"


def lookup_content_subject(db: Session, content_type: str, content_id: int) -> str:
    if content_type == "note":
        note = db.get(Note, content_id)
        return note.subject if note else "General"
    if content_type == "flashcard":
        card = db.get(Flashcard, content_id)
        return card.note.subject if card else "General"
    quiz = db.get(Quiz, content_id)
    return quiz.note.subject if quiz else "General"


def lookup_content_meta(db: Session, content_type: str, content_id: int) -> str:
    if content_type == "note":
        note = db.get(Note, content_id)
        return humanize_delta(note.created_at) if note else ""
    if content_type == "flashcard":
        card = db.get(Flashcard, content_id)
        return f"{card.review_count} reviews" if card else ""
    quiz = db.get(Quiz, content_id)
    latest = quiz.attempts[0].percent if quiz and quiz.attempts else None
    return f"{latest}% score" if latest is not None else "Not attempted"


def lookup_content_preview(db: Session, content_type: str, content_id: int) -> str:
    if content_type == "note":
        note = db.get(Note, content_id)
        return note_preview(note) if note else ""
    if content_type == "flashcard":
        card = db.get(Flashcard, content_id)
        return card.answer if card else ""
    quiz = db.get(Quiz, content_id)
    if quiz and quiz.attempts:
        attempt = quiz.attempts[0]
        return f"Latest score: {attempt.percent}%"
    return "Quiz ready to attempt."


def slugify(value: str) -> str:
    normalized = "".join(char.lower() if char.isalnum() else "-" for char in value).strip("-")
    return "-".join(part for part in normalized.split("-") if part) or "study-export"


def record_history(db: Session, user_id: int, event_type: str, content_type: str, content_id: int, title: str, description: str) -> None:
    db.add(
        HistoryEvent(
            user_id=user_id,
            event_type=event_type,
            content_type=content_type,
            content_id=content_id,
            title=title,
            description=description,
            metadata_json="{}",
        )
    )


def record_study_session(
    db: Session,
    *,
    user_id: int,
    session_type: str,
    note_id: int | None = None,
    quiz_id: int | None = None,
    flashcard_id: int | None = None,
    duration_seconds: int = 0,
    status: str = "completed",
    metrics: dict[str, Any] | None = None,
) -> None:
    db.add(
        StudySession(
            user_id=user_id,
            note_id=note_id,
            quiz_id=quiz_id,
            flashcard_id=flashcard_id,
            session_type=session_type,
            status=status,
            duration_seconds=duration_seconds,
            metrics_json=json.dumps(metrics or {}),
        )
    )


def record_analytics(
    db: Session,
    user_id: int,
    event_name: str,
    content_type: str,
    content_id: int | None,
    subject: str | None,
    *,
    note_id: int | None = None,
    quiz_id: int | None = None,
    value: int = 0,
    metadata: dict[str, Any] | None = None,
) -> None:
    db.add(
        AnalyticsEvent(
            user_id=user_id,
            note_id=note_id,
            quiz_id=quiz_id,
            event_name=event_name,
            content_type=content_type,
            content_id=content_id,
            subject=subject,
            value=value,
            metadata_json=json.dumps(metadata or {}),
        )
    )


def get_user_first_name(db: Session, user_id: int) -> str:
    user = db.get(User, user_id)
    if user is None or not user.full_name:
        return "there"
    return user.full_name.split()[0]


def get_user_initials(full_name: str | None) -> str:
    if not full_name:
        return "NA"
    parts = [part[0] for part in full_name.split() if part]
    return "".join(parts[:2]).upper() or "NA"


def build_shell_payload(db: Session, user: User, page_title: str) -> dict[str, Any]:
    due_flashcards = db.scalar(
        select(func.count())
        .select_from(Flashcard)
        .join(Note)
        .where(
            Note.user_id == user.id,
            or_(Flashcard.next_review_at.is_(None), Flashcard.next_review_at <= datetime.now(UTC).replace(tzinfo=None)),
        )
    ) or 0
    latest_attempt = db.scalar(
        select(QuizAttempt)
        .where(QuizAttempt.user_id == user.id)
        .order_by(QuizAttempt.created_at.desc())
    )
    latest_note = db.scalar(select(Note).where(Note.user_id == user.id, Note.status == "ready").order_by(Note.created_at.desc()))
    notifications: list[dict[str, str | int]] = []
    if due_flashcards:
        notifications.append(
            {
                "title": f"{due_flashcards} flashcards due for review",
                "sub": "Continue your spaced repetition streak.",
                "time": "Now",
                "action": "Review now",
                "href": "/flashcards",
            }
        )
    if latest_attempt is not None:
        notifications.append(
            {
                "title": f"Latest quiz score: {latest_attempt.percent}%",
                "sub": latest_attempt.quiz.note.title,
                "time": humanize_delta(latest_attempt.created_at),
                "action": "View results",
                "href": f"/quiz-results?attempt_id={latest_attempt.id}",
            }
        )
    if latest_note is not None:
        notifications.append(
            {
                "title": "A study pack is ready",
                "sub": latest_note.title,
                "time": humanize_delta(latest_note.created_at),
                "action": "Open notes",
                "href": f"/notes/{latest_note.id}",
            }
        )
    return {
        "page_title": page_title,
        "user": {
            "full_name": user.full_name,
            "first_name": get_user_first_name(db, user.id),
            "initials": get_user_initials(user.full_name),
            "email": user.email,
        },
        "notifications": notifications[:3],
    }


def _store_processed_result(result: dict[str, Any]) -> str:
    result_id = secrets.token_hex(8)
    processed_results[result_id] = result
    return result_id


def _get_processed_result(result_id: str | None) -> dict[str, Any]:
    if not result_id:
        return {}
    return processed_results.get(result_id, {})


app = create_app()
