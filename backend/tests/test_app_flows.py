from __future__ import annotations

import io
import json

import pytest

import backend.main as main_module
from backend.database import session_scope
from backend.models import AnalyticsEvent, Bookmark, ExportRecord, Flashcard, HistoryEvent, Note, QuizAttempt, StudySession, UploadedFile, User


def _signup(client, email: str = "user@example.com", password: str = "strong-password"):
    return client.post(
        "/api/auth/signup",
        json={"full_name": "Test User", "email": email, "password": password},
    )


def _login(client, email: str = "user@example.com", password: str = "strong-password"):
    return client.post("/api/auth/login", json={"email": email, "password": password})


def _create_note(client, monkeypatch, sample_generation_result):
    monkeypatch.setattr(main_module, "generate_full_content", lambda text: sample_generation_result)
    response = client.post(
        "/process",
        data={"source_type": "text", "text": "Cell biology notes", "subject": "Biology", "tags": "cells,organelles"},
    )
    assert response.status_code == 202
    return response.json()["note_id"]


def test_signup_login_logout_and_protected_routes(client):
    protected = client.get("/dashboard", follow_redirects=False)
    assert protected.status_code == 303
    assert protected.headers["location"] == "/login"

    signup_response = _signup(client)
    assert signup_response.status_code == 201
    body = signup_response.json()
    assert body["redirect_url"] == "/dashboard"

    dashboard = client.get("/api/dashboard")
    assert dashboard.status_code == 200
    assert dashboard.json()["greeting_name"] == "Test"

    logout_response = client.post("/logout", follow_redirects=False)
    assert logout_response.status_code == 303
    assert logout_response.headers["location"] == "/login"

    protected_after_logout = client.get("/api/dashboard")
    assert protected_after_logout.status_code == 401

    login_response = _login(client)
    assert login_response.status_code == 200
    assert login_response.json()["redirect_url"] == "/dashboard"


def test_duplicate_signup_is_rejected_gracefully(client):
    assert _signup(client).status_code == 201
    duplicate = _signup(client)
    assert duplicate.status_code == 409
    assert duplicate.json()["detail"] == "An account with that email already exists."


def test_upload_processing_and_dashboard_stats(client, monkeypatch, sample_generation_result):
    _signup(client)
    note_id = _create_note(client, monkeypatch, sample_generation_result)

    status_response = client.get(f"/api/processing/{note_id}")
    assert status_response.status_code == 200
    assert status_response.json()["status"] == "ready"

    note_response = client.get(f"/api/notes/{note_id}")
    assert note_response.status_code == 200
    note_body = note_response.json()
    assert note_body["title"] == "Cell Biology Basics"
    assert len(note_body["flashcards"]) == 2
    assert note_body["quiz"]["question_count"] == 2

    dashboard = client.get("/api/dashboard").json()
    assert dashboard["summary"]["notes_created"] == 1
    assert dashboard["summary"]["flashcards_due"] == 2
    assert dashboard["summary"]["quiz_average"] == 0
    assert dashboard["recent_notes"][0]["id"] == note_id


def test_file_upload_persists_uploaded_file_metadata(client, monkeypatch, sample_generation_result):
    _signup(client)
    monkeypatch.setattr(main_module, "generate_full_content", lambda text: sample_generation_result)

    upload_response = client.post(
        "/process",
        data={"source_type": "file", "subject": "Biology", "tags": "cells"},
        files={"file": ("cell-notes.txt", io.BytesIO(b"Cell notes from file upload"), "text/plain")},
    )
    assert upload_response.status_code == 202
    note_id = upload_response.json()["note_id"]

    with session_scope() as db:
        uploaded_file = db.query(UploadedFile).filter(UploadedFile.note_id == note_id).one_or_none()
        assert uploaded_file is not None
        assert uploaded_file.original_filename == "cell-notes.txt"
        assert uploaded_file.media_type == "text/plain"
        assert uploaded_file.size_bytes == len(b"Cell notes from file upload")
        assert uploaded_file.status == "processed"


def test_note_bookmark_delete_and_history(client, monkeypatch, sample_generation_result):
    _signup(client)
    note_id = _create_note(client, monkeypatch, sample_generation_result)

    bookmark_response = client.post(
        "/api/bookmarks/toggle",
        json={"content_type": "note", "content_id": note_id},
    )
    assert bookmark_response.status_code == 200
    assert bookmark_response.json()["bookmarked"] is True

    bookmarks = client.get("/api/bookmarks").json()
    assert bookmarks["items"][0]["content_id"] == note_id

    history = client.get("/api/history").json()
    assert any(item["event_type"] == "bookmarked" for item in history["items"])

    delete_response = client.delete(f"/api/notes/{note_id}")
    assert delete_response.status_code == 204
    assert client.get(f"/api/notes/{note_id}").status_code == 404


def test_flashcard_review_quiz_attempt_progress_and_exports(client, monkeypatch, sample_generation_result):
    _signup(client)
    note_id = _create_note(client, monkeypatch, sample_generation_result)
    note_payload = client.get(f"/api/notes/{note_id}").json()
    quiz = note_payload["quiz"]
    flashcard = note_payload["flashcards"][0]

    review_response = client.post(
        f"/api/flashcards/{flashcard['id']}/review",
        json={"rating": "got"},
    )
    assert review_response.status_code == 200
    review_payload = review_response.json()
    assert review_payload["review_count"] == 1
    assert review_payload["mastery_level"] >= 2
    assert review_payload["next_review_at"] is not None

    answers = {str(question["id"]): question["correct_index"] for question in quiz["questions"]}
    attempt_response = client.post(
        f"/api/quizzes/{quiz['id']}/attempts",
        json={"answers": answers, "completion_seconds": 90},
    )
    assert attempt_response.status_code == 201
    attempt_id = attempt_response.json()["attempt_id"]

    results = client.get(f"/api/quiz-attempts/{attempt_id}").json()
    assert results["score"] == 2
    assert results["percent"] == 100
    assert results["wrong_answers"] == []

    quiz_results_page_data = client.get(f"/quiz-results?attempt_id={attempt_id}")
    assert quiz_results_page_data.status_code == 200

    progress = client.get("/api/progress").json()
    assert progress["stats"]["quizzes_taken"] == 1
    assert progress["stats"]["average_score"] == 100
    assert progress["stats"]["cards_reviewed"] == 1
    assert progress["recent_quiz_history"][0]["percent"] == 100

    json_export = client.get(f"/export/json/{note_id}")
    assert json_export.status_code == 200
    assert json_export.headers["content-type"].startswith("application/json")
    assert json.loads(json_export.content.decode("utf-8"))["note"]["title"] == "Cell Biology Basics"

    text_export = client.get(f"/export/txt/{note_id}")
    assert text_export.status_code == 200
    assert "Cell Biology Basics" in text_export.text

    pdf_export = client.get(f"/export/pdf/{note_id}")
    assert pdf_export.status_code == 200
    assert pdf_export.headers["content-type"].startswith("application/pdf")
    assert pdf_export.content.startswith(b"%PDF")


def test_interactions_create_study_sessions_and_analytics_records(client, monkeypatch, sample_generation_result):
    _signup(client)
    note_id = _create_note(client, monkeypatch, sample_generation_result)
    note_payload = client.get(f"/api/notes/{note_id}").json()
    quiz = note_payload["quiz"]
    flashcard = note_payload["flashcards"][0]

    assert client.get(f"/notes/{note_id}").status_code == 200
    assert client.post(f"/api/flashcards/{flashcard['id']}/review", json={"rating": "got"}).status_code == 200
    assert client.post(
        f"/api/quizzes/{quiz['id']}/attempts",
        json={"answers": {str(question['id']): question['correct_index'] for question in quiz['questions']}, "completion_seconds": 75},
    ).status_code == 201
    assert client.get(f"/export/json/{note_id}").status_code == 200

    with session_scope() as db:
        assert db.query(StudySession).count() >= 2
        assert db.query(AnalyticsEvent).count() >= 4
        event_names = {event.event_name for event in db.query(AnalyticsEvent).all()}
        assert {"note_generated", "note_viewed", "flashcard_reviewed", "quiz_completed", "export_created"}.issubset(event_names)


def test_authenticated_html_pages_render_successfully(client, monkeypatch, sample_generation_result):
    _signup(client)
    note_id = _create_note(client, monkeypatch, sample_generation_result)
    note_payload = client.get(f"/api/notes/{note_id}").json()
    quiz_id = note_payload["quiz"]["id"]

    attempt_response = client.post(
        f"/api/quizzes/{quiz_id}/attempts",
        json={"answers": {str(question['id']): question['correct_index'] for question in note_payload['quiz']['questions']}, "completion_seconds": 30},
    )
    attempt_id = attempt_response.json()["attempt_id"]

    for path in [
        "/dashboard",
        "/upload",
        "/notes",
        f"/notes/{note_id}",
        f"/flashcards?note_id={note_id}",
        "/quizzes",
        f"/quiz/{quiz_id}",
        f"/quiz-results?attempt_id={attempt_id}",
        "/history",
        "/bookmarks",
        "/progress",
        f"/export?note_id={note_id}",
        "/settings",
    ]:
        response = client.get(path)
        assert response.status_code == 200, path
        assert "page-data" in response.text


def test_authenticated_pages_render_shared_sidebar_and_navbar(client, monkeypatch, sample_generation_result):
    _signup(client)
    note_id = _create_note(client, monkeypatch, sample_generation_result)
    note_payload = client.get(f"/api/notes/{note_id}").json()
    quiz_id = note_payload["quiz"]["id"]

    response = client.get("/notes")
    assert response.status_code == 200
    assert 'id="appSidebar"' in response.text
    assert 'id="appTopbar"' in response.text
    assert 'href="/dashboard"' in response.text
    assert 'href="/upload"' in response.text
    assert 'href="/settings"' in response.text
    assert 'aria-current="page"' in response.text

    for path in ["/dashboard", "/upload", f"/notes/{note_id}", f"/quiz/{quiz_id}", "/history", "/bookmarks", "/progress", "/export", "/settings"]:
        html = client.get(path).text
        assert 'id="appSidebar"' in html, path
        assert 'id="appTopbar"' in html, path


def test_settings_update_and_reset_progress(client, monkeypatch, sample_generation_result):
    _signup(client)
    note_id = _create_note(client, monkeypatch, sample_generation_result)
    note_payload = client.get(f"/api/notes/{note_id}").json()
    flashcard_id = note_payload["flashcards"][0]["id"]
    quiz_id = note_payload["quiz"]["id"]
    question = note_payload["quiz"]["questions"][0]

    settings_response = client.post(
        "/api/settings",
        json={
            "full_name": "Updated User",
            "institution": "OpenAI Academy",
            "exam": "Finals",
            "exam_date": "2026-06-01",
            "spaced_repetition": False,
            "auto_quiz": False,
            "questions_per_quiz": 5,
            "export_header": "Study Pack",
        },
    )
    assert settings_response.status_code == 200
    settings_body = settings_response.json()
    assert settings_body["profile"]["name"] == "Updated User"
    assert settings_body["learning"]["spaced_repetition"] is False
    assert settings_body["ai"]["auto_quiz"] is False
    assert settings_body["export"]["header"] == "Study Pack"

    assert client.post(f"/api/flashcards/{flashcard_id}/review", json={"rating": "almost"}).status_code == 200
    assert client.post(
        f"/api/bookmarks/toggle",
        json={"content_type": "note", "content_id": note_id},
    ).status_code == 200
    assert client.post(
        f"/api/quizzes/{quiz_id}/attempts",
        json={"answers": {str(question['id']): question['correct_index']}, "completion_seconds": 45},
    ).status_code == 201

    reset_response = client.post("/api/settings/reset-progress")
    assert reset_response.status_code == 200
    assert reset_response.json()["status"] == "reset"

    progress = client.get("/api/progress").json()
    assert progress["stats"]["quizzes_taken"] == 0
    assert progress["stats"]["cards_reviewed"] == 0
    assert progress["stats"]["bookmarks"] == 0


def test_login_accepts_legacy_hash_and_upgrades_it(client):
    legacy_hash = main_module.hash_legacy_password("strong-password")
    with session_scope() as db:
        user = User(full_name="Legacy User", email="legacy@example.com", password_hash=legacy_hash)
        db.add(user)
        db.flush()
        user_id = user.id

    login_response = client.post("/api/auth/login", json={"email": "legacy@example.com", "password": "strong-password"})
    assert login_response.status_code == 200

    with session_scope() as db:
        upgraded = db.get(User, user_id)
        assert upgraded is not None
        assert upgraded.password_hash != legacy_hash
        assert upgraded.password_hash.startswith("$2")
