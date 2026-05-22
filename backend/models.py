from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import Boolean, DateTime, ForeignKey, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .database import Base


def utcnow_naive() -> datetime:
    return datetime.now(UTC).replace(tzinfo=None)


class TimestampMixin:
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow_naive)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=utcnow_naive,
        onupdate=utcnow_naive,
    )


class User(TimestampMixin, Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    full_name: Mapped[str] = mapped_column(String(255))
    email: Mapped[str] = mapped_column(String(255), unique=True, index=True)
    password_hash: Mapped[str] = mapped_column(String(512))

    notes: Mapped[list["Note"]] = relationship(back_populates="user", cascade="all, delete-orphan")
    quiz_attempts: Mapped[list["QuizAttempt"]] = relationship(back_populates="user", cascade="all, delete-orphan")
    bookmarks: Mapped[list["Bookmark"]] = relationship(back_populates="user", cascade="all, delete-orphan")
    history_events: Mapped[list["HistoryEvent"]] = relationship(back_populates="user", cascade="all, delete-orphan")
    settings: Mapped["UserSetting | None"] = relationship(back_populates="user", cascade="all, delete-orphan", uselist=False)
    exports: Mapped[list["ExportRecord"]] = relationship(back_populates="user", cascade="all, delete-orphan")
    uploaded_files: Mapped[list["UploadedFile"]] = relationship(back_populates="user", cascade="all, delete-orphan")
    study_sessions: Mapped[list["StudySession"]] = relationship(back_populates="user", cascade="all, delete-orphan")
    analytics_events: Mapped[list["AnalyticsEvent"]] = relationship(back_populates="user", cascade="all, delete-orphan")


class Note(TimestampMixin, Base):
    __tablename__ = "notes"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True)
    title: Mapped[str] = mapped_column(String(255), default="Untitled Note")
    subject: Mapped[str] = mapped_column(String(255), default="General")
    source_type: Mapped[str] = mapped_column(String(50), default="text")
    source_name: Mapped[str | None] = mapped_column(String(255), nullable=True)
    source_text: Mapped[str] = mapped_column(Text, default="")
    tags_json: Mapped[str] = mapped_column(Text, default="[]")
    status: Mapped[str] = mapped_column(String(50), default="processing", index=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    summary: Mapped[str] = mapped_column(Text, default="")
    notes_markdown: Mapped[str] = mapped_column(Text, default="")
    key_concepts_json: Mapped[str] = mapped_column(Text, default="[]")
    difficulty_analysis_json: Mapped[str] = mapped_column(Text, default="{}")
    study_recommendations_json: Mapped[str] = mapped_column(Text, default="[]")
    last_viewed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)

    user: Mapped["User"] = relationship(back_populates="notes")
    flashcards: Mapped[list["Flashcard"]] = relationship(back_populates="note", cascade="all, delete-orphan", order_by="Flashcard.position")
    quiz: Mapped["Quiz | None"] = relationship(back_populates="note", cascade="all, delete-orphan", uselist=False)
    uploaded_files: Mapped[list["UploadedFile"]] = relationship(back_populates="note", cascade="all, delete-orphan")
    study_sessions: Mapped[list["StudySession"]] = relationship(back_populates="note", cascade="all, delete-orphan")
    analytics_events: Mapped[list["AnalyticsEvent"]] = relationship(back_populates="note", cascade="all, delete-orphan")


class Flashcard(TimestampMixin, Base):
    __tablename__ = "flashcards"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    note_id: Mapped[int] = mapped_column(ForeignKey("notes.id"), index=True)
    position: Mapped[int] = mapped_column(Integer, default=0)
    question: Mapped[str] = mapped_column(Text)
    answer: Mapped[str] = mapped_column(Text)
    topic: Mapped[str | None] = mapped_column(String(255), nullable=True)
    review_count: Mapped[int] = mapped_column(Integer, default=0)
    got_it_count: Mapped[int] = mapped_column(Integer, default=0)
    almost_count: Mapped[int] = mapped_column(Integer, default=0)
    still_learning_count: Mapped[int] = mapped_column(Integer, default=0)
    mastery_level: Mapped[int] = mapped_column(Integer, default=0)
    last_rating: Mapped[str | None] = mapped_column(String(30), nullable=True)
    last_reviewed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    next_review_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)

    note: Mapped["Note"] = relationship(back_populates="flashcards")
    study_sessions: Mapped[list["StudySession"]] = relationship(back_populates="flashcard", cascade="all, delete-orphan")


class Quiz(TimestampMixin, Base):
    __tablename__ = "quizzes"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    note_id: Mapped[int] = mapped_column(ForeignKey("notes.id"), unique=True, index=True)
    title: Mapped[str] = mapped_column(String(255), default="Quiz")
    duration_seconds: Mapped[int] = mapped_column(Integer, default=600)
    weak_topics_json: Mapped[str] = mapped_column(Text, default="[]")

    note: Mapped["Note"] = relationship(back_populates="quiz")
    questions: Mapped[list["QuizQuestion"]] = relationship(back_populates="quiz", cascade="all, delete-orphan", order_by="QuizQuestion.position")
    attempts: Mapped[list["QuizAttempt"]] = relationship(back_populates="quiz", cascade="all, delete-orphan", order_by="desc(QuizAttempt.created_at)")
    study_sessions: Mapped[list["StudySession"]] = relationship(back_populates="quiz", cascade="all, delete-orphan")
    analytics_events: Mapped[list["AnalyticsEvent"]] = relationship(back_populates="quiz", cascade="all, delete-orphan")


class QuizQuestion(TimestampMixin, Base):
    __tablename__ = "quiz_questions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    quiz_id: Mapped[int] = mapped_column(ForeignKey("quizzes.id"), index=True)
    position: Mapped[int] = mapped_column(Integer, default=0)
    prompt: Mapped[str] = mapped_column(Text)
    options_json: Mapped[str] = mapped_column(Text, default="[]")
    correct_index: Mapped[int] = mapped_column(Integer, default=0)
    explanation: Mapped[str | None] = mapped_column(Text, nullable=True)
    topic: Mapped[str | None] = mapped_column(String(255), nullable=True)
    difficulty: Mapped[str | None] = mapped_column(String(50), nullable=True)

    quiz: Mapped["Quiz"] = relationship(back_populates="questions")


class QuizAttempt(TimestampMixin, Base):
    __tablename__ = "quiz_attempts"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    quiz_id: Mapped[int] = mapped_column(ForeignKey("quizzes.id"), index=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True)
    answers_json: Mapped[str] = mapped_column(Text, default="{}")
    score: Mapped[int] = mapped_column(Integer, default=0)
    total_questions: Mapped[int] = mapped_column(Integer, default=0)
    percent: Mapped[int] = mapped_column(Integer, default=0)
    completion_seconds: Mapped[int] = mapped_column(Integer, default=0)
    wrong_answers_json: Mapped[str] = mapped_column(Text, default="[]")
    weak_topics_json: Mapped[str] = mapped_column(Text, default="[]")
    recommendations_json: Mapped[str] = mapped_column(Text, default="[]")
    strengths_json: Mapped[str] = mapped_column(Text, default="[]")
    weaknesses_json: Mapped[str] = mapped_column(Text, default="[]")

    quiz: Mapped["Quiz"] = relationship(back_populates="attempts")
    user: Mapped["User"] = relationship(back_populates="quiz_attempts")


class Bookmark(TimestampMixin, Base):
    __tablename__ = "bookmarks"
    __table_args__ = (
        UniqueConstraint("user_id", "content_type", "content_id", name="uq_bookmark_user_content"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True)
    content_type: Mapped[str] = mapped_column(String(50))
    content_id: Mapped[int] = mapped_column(Integer)

    user: Mapped["User"] = relationship(back_populates="bookmarks")


class HistoryEvent(Base):
    __tablename__ = "history"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True)
    event_type: Mapped[str] = mapped_column(String(50), index=True)
    content_type: Mapped[str] = mapped_column(String(50))
    content_id: Mapped[int] = mapped_column(Integer)
    title: Mapped[str] = mapped_column(String(255))
    description: Mapped[str] = mapped_column(Text, default="")
    metadata_json: Mapped[str] = mapped_column(Text, default="{}")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow_naive, index=True)

    user: Mapped["User"] = relationship(back_populates="history_events")


class UserSetting(TimestampMixin, Base):
    __tablename__ = "settings"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), unique=True, index=True)
    theme: Mapped[str] = mapped_column(String(20), default="light")
    institution: Mapped[str | None] = mapped_column(String(255), nullable=True)
    exam: Mapped[str | None] = mapped_column(String(255), nullable=True)
    exam_date: Mapped[str | None] = mapped_column(String(50), nullable=True)
    spaced_repetition: Mapped[bool] = mapped_column(Boolean, default=True)
    show_hints: Mapped[bool] = mapped_column(Boolean, default=False)
    progress_tracking: Mapped[bool] = mapped_column(Boolean, default=True)
    note_length: Mapped[str] = mapped_column(String(50), default="Balanced")
    ai_tone: Mapped[str] = mapped_column(String(50), default="Academic")
    auto_flashcards: Mapped[bool] = mapped_column(Boolean, default=True)
    auto_quiz: Mapped[bool] = mapped_column(Boolean, default=True)
    questions_per_quiz: Mapped[int] = mapped_column(Integer, default=10)
    difficulty: Mapped[int] = mapped_column(Integer, default=65)
    reminders: Mapped[bool] = mapped_column(Boolean, default=True)
    reminder_time: Mapped[str] = mapped_column(String(20), default="09:00")
    export_format: Mapped[str] = mapped_column(String(20), default="pdf")
    export_font_size: Mapped[str] = mapped_column(String(20), default="Medium")
    export_include_cover: Mapped[bool] = mapped_column(Boolean, default=True)
    export_include_toc: Mapped[bool] = mapped_column(Boolean, default=True)
    export_include_concepts: Mapped[bool] = mapped_column(Boolean, default=False)
    export_include_pages: Mapped[bool] = mapped_column(Boolean, default=True)
    export_header: Mapped[str | None] = mapped_column(String(255), nullable=True)

    user: Mapped["User"] = relationship(back_populates="settings")


class ExportRecord(TimestampMixin, Base):
    __tablename__ = "exports"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True)
    content_type: Mapped[str] = mapped_column(String(50))
    content_id: Mapped[int] = mapped_column(Integer)
    format: Mapped[str] = mapped_column(String(20))

    user: Mapped["User"] = relationship(back_populates="exports")


class UploadedFile(TimestampMixin, Base):
    __tablename__ = "uploaded_files"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True)
    note_id: Mapped[int] = mapped_column(ForeignKey("notes.id"), index=True)
    source_type: Mapped[str] = mapped_column(String(50), default="file")
    original_filename: Mapped[str] = mapped_column(String(255))
    media_type: Mapped[str | None] = mapped_column(String(255), nullable=True)
    size_bytes: Mapped[int] = mapped_column(Integer, default=0)
    extracted_char_count: Mapped[int] = mapped_column(Integer, default=0)
    checksum: Mapped[str | None] = mapped_column(String(128), nullable=True)
    storage_path: Mapped[str | None] = mapped_column(String(512), nullable=True)
    status: Mapped[str] = mapped_column(String(50), default="processed")

    user: Mapped["User"] = relationship(back_populates="uploaded_files")
    note: Mapped["Note"] = relationship(back_populates="uploaded_files")


class StudySession(TimestampMixin, Base):
    __tablename__ = "study_sessions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True)
    note_id: Mapped[int | None] = mapped_column(ForeignKey("notes.id"), index=True, nullable=True)
    quiz_id: Mapped[int | None] = mapped_column(ForeignKey("quizzes.id"), index=True, nullable=True)
    flashcard_id: Mapped[int | None] = mapped_column(ForeignKey("flashcards.id"), index=True, nullable=True)
    session_type: Mapped[str] = mapped_column(String(50), index=True)
    status: Mapped[str] = mapped_column(String(50), default="completed")
    duration_seconds: Mapped[int] = mapped_column(Integer, default=0)
    metrics_json: Mapped[str] = mapped_column(Text, default="{}")

    user: Mapped["User"] = relationship(back_populates="study_sessions")
    note: Mapped["Note | None"] = relationship(back_populates="study_sessions")
    quiz: Mapped["Quiz | None"] = relationship(back_populates="study_sessions")
    flashcard: Mapped["Flashcard | None"] = relationship(back_populates="study_sessions")


class AnalyticsEvent(TimestampMixin, Base):
    __tablename__ = "analytics"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True)
    note_id: Mapped[int | None] = mapped_column(ForeignKey("notes.id"), index=True, nullable=True)
    quiz_id: Mapped[int | None] = mapped_column(ForeignKey("quizzes.id"), index=True, nullable=True)
    event_name: Mapped[str] = mapped_column(String(100), index=True)
    content_type: Mapped[str] = mapped_column(String(50), default="system")
    content_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    subject: Mapped[str | None] = mapped_column(String(255), nullable=True)
    value: Mapped[int] = mapped_column(Integer, default=0)
    metadata_json: Mapped[str] = mapped_column(Text, default="{}")

    user: Mapped["User"] = relationship(back_populates="analytics_events")
    note: Mapped["Note | None"] = relationship(back_populates="analytics_events")
    quiz: Mapped["Quiz | None"] = relationship(back_populates="analytics_events")
