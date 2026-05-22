from __future__ import annotations

import os

from fastapi.testclient import TestClient

import backend.database as database_module
import backend.main as main_module
from backend.main import create_app, hash_password, verify_password


def test_password_hash_uses_bcrypt():
    password_hash = hash_password("strong-password")
    assert password_hash.startswith("$2")
    assert verify_password("strong-password", password_hash) is True
    assert verify_password("wrong-password", password_hash) is False


def test_vercel_default_database_path(monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.setenv("VERCEL", "1")
    assert database_module._default_database_url() == "sqlite:////tmp/study_assistant.db"


def test_local_default_database_path(monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.delenv("VERCEL", raising=False)
    assert database_module._default_database_url().startswith("sqlite:///")


def test_production_session_defaults():
    app = create_app({"testing": True, "session_secret": "test-secret", "session_https_only": True})
    middleware = next(item for item in app.user_middleware if item.cls.__name__ == "SessionMiddleware")
    assert middleware.kwargs["https_only"] is True
    assert middleware.kwargs["same_site"] == "lax"
    assert middleware.kwargs["max_age"] > 0


def test_bookmarks_history_and_quiz_attempts_survive_restart(tmp_path, monkeypatch):
    database_url = f"sqlite:///{tmp_path / 'restart-persistence.db'}"
    sample_generation_result = {
        "title": "Restart Persistence",
        "summary": "Persistence summary.",
        "notes": "## Persistence\nThis survives restarts.",
        "key_concepts": [{"term": "Persistence", "explanation": "Data remains after restart."}],
        "flashcards": [{"question": "What survives?", "answer": "Bookmarks and attempts.", "topic": "Persistence"}],
        "quiz": [
            {
                "question": "What should survive app restart?",
                "options": ["Nothing", "Only notes", "Bookmarks and quiz attempts", "Only settings"],
                "answer": "Bookmarks and quiz attempts",
                "topic": "Persistence",
                "explanation": "Persistent storage keeps user data available.",
            }
        ],
    }
    monkeypatch.setattr(main_module, "generate_full_content", lambda text: sample_generation_result)

    first_app = create_app({"database_url": database_url, "process_inline": True, "testing": True, "session_secret": "restart-secret"})
    with TestClient(first_app) as client:
        assert client.post("/api/auth/signup", json={"full_name": "Restart User", "email": "restart@example.com", "password": "strong-password"}).status_code == 201
        note_id = client.post("/process", data={"source_type": "text", "text": "restart data", "subject": "Systems"}).json()["note_id"]
        note_payload = client.get(f"/api/notes/{note_id}").json()
        quiz = note_payload["quiz"]
        question = quiz["questions"][0]
        assert client.post("/api/bookmarks/toggle", json={"content_type": "note", "content_id": note_id}).status_code == 200
        assert client.post(
            f"/api/quizzes/{quiz['id']}/attempts",
            json={"answers": {str(question['id']): question['correct_index']}, "completion_seconds": 50},
        ).status_code == 201

    second_app = create_app({"database_url": database_url, "process_inline": True, "testing": True, "session_secret": "restart-secret"})
    with TestClient(second_app) as client:
        assert client.post("/api/auth/login", json={"email": "restart@example.com", "password": "strong-password"}).status_code == 200
        bookmarks = client.get("/api/bookmarks").json()
        history = client.get("/api/history").json()
        progress = client.get("/api/progress").json()
        assert len(bookmarks["items"]) == 1
        assert len(history["items"]) >= 2
        assert progress["stats"]["quizzes_taken"] == 1
