import json
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from backend.main import create_app


MOCK_GENERATED_CONTENT = {
    "title": "Photosynthesis",
    "summary": "Plants convert sunlight into stored energy.",
    "notes": "## Overview\nPhotosynthesis turns light into glucose.",
    "key_concepts": [
        {"term": "Chlorophyll", "explanation": "Green pigment that captures light."},
        {"term": "Calvin Cycle", "explanation": "Builds sugars from carbon dioxide."},
    ],
    "flashcards": [
        {"question": "What does chlorophyll do?", "answer": "Captures light energy."},
        {"question": "Where does the Calvin Cycle occur?", "answer": "In the stroma."},
    ],
    "quiz": [
        {
            "question": "Which pigment absorbs light for photosynthesis?",
            "options": ["Hemoglobin", "Chlorophyll", "Melanin", "Keratin"],
            "answer": "Chlorophyll",
            "topic": "Photosynthesis",
            "explanation": "Chlorophyll is the main light-absorbing pigment.",
        },
        {
            "question": "What is produced in the Calvin Cycle?",
            "options": ["ATP", "G3P", "Oxygen", "Water"],
            "answer": "G3P",
            "topic": "Calvin Cycle",
            "explanation": "G3P is the direct sugar product.",
        },
    ],
    "difficulty_analysis": {
        "overall": "medium",
        "challenging_topics": ["Calvin Cycle"],
    },
    "study_recommendations": [
        "Review the Calvin Cycle once more.",
        "Practice light-reaction terminology with flashcards.",
    ],
}


def _make_client(tmp_path: Path) -> TestClient:
    app = create_app(
        {
            "database_url": f"sqlite:///{tmp_path / 'test.db'}",
            "session_secret": "test-secret",
            "process_inline": True,
            "testing": True,
        }
    )
    return TestClient(app, raise_server_exceptions=True)


def test_auth_flow_and_protected_route_redirect(tmp_path: Path):
    client = _make_client(tmp_path)

    unauthenticated = client.get("/dashboard", follow_redirects=False)
    assert unauthenticated.status_code == 303
    assert unauthenticated.headers["location"] == "/login"

    signup = client.post(
        "/api/auth/signup",
        json={
            "full_name": "Arjun Sharma",
            "email": "arjun@example.com",
            "password": "strong-password",
        },
    )
    assert signup.status_code == 201
    assert signup.json()["user"]["email"] == "arjun@example.com"

    login = client.post(
        "/api/auth/login",
        json={"email": "arjun@example.com", "password": "strong-password"},
    )
    assert login.status_code == 200
    assert login.json()["redirect_url"] == "/dashboard"

    dashboard = client.get("/dashboard")
    assert dashboard.status_code == 200
    assert "Good morning" in dashboard.text

    logout = client.post("/logout", follow_redirects=False)
    assert logout.status_code == 303
    assert logout.headers["location"] == "/login"


def test_process_creates_persisted_note_flashcards_and_quiz(tmp_path: Path):
    client = _make_client(tmp_path)
    client.post(
        "/api/auth/signup",
        json={
            "full_name": "Arjun Sharma",
            "email": "arjun@example.com",
            "password": "strong-password",
        },
    )
    client.post(
        "/api/auth/login",
        json={"email": "arjun@example.com", "password": "strong-password"},
    )

    with patch("backend.main.generate_full_content", return_value=dict(MOCK_GENERATED_CONTENT)):
        response = client.post(
            "/process",
            data={
                "source_type": "text",
                "text": "Photosynthesis notes",
                "subject": "Biology",
                "tags": "plants,light",
            },
        )

    assert response.status_code == 202
    body = response.json()
    assert body["status"] == "processing"
    note_id = body["note_id"]

    detail = client.get(f"/api/notes/{note_id}")
    assert detail.status_code == 200
    note = detail.json()
    assert note["title"] == "Photosynthesis"
    assert len(note["flashcards"]) == 2
    assert note["quiz"]["question_count"] == 2
    assert note["status"] == "ready"

    notes_listing = client.get("/api/notes")
    assert notes_listing.status_code == 200
    assert notes_listing.json()["items"][0]["id"] == note_id


def test_bookmarks_progress_and_history_are_real(tmp_path: Path):
    client = _make_client(tmp_path)
    client.post(
        "/api/auth/signup",
        json={
            "full_name": "Arjun Sharma",
            "email": "arjun@example.com",
            "password": "strong-password",
        },
    )
    client.post(
        "/api/auth/login",
        json={"email": "arjun@example.com", "password": "strong-password"},
    )

    with patch("backend.main.generate_full_content", return_value=dict(MOCK_GENERATED_CONTENT)):
        process = client.post(
            "/process",
            data={"source_type": "text", "text": "Photosynthesis notes", "subject": "Biology"},
        )
    note_id = process.json()["note_id"]

    bookmark = client.post(
        "/api/bookmarks/toggle",
        json={"content_type": "note", "content_id": note_id},
    )
    assert bookmark.status_code == 200
    assert bookmark.json()["bookmarked"] is True

    progress = client.get("/api/progress")
    assert progress.status_code == 200
    progress_body = progress.json()
    assert progress_body["stats"]["notes_generated"] == 1
    assert progress_body["stats"]["bookmarks"] == 1

    history = client.get("/api/history")
    assert history.status_code == 200
    items = history.json()["items"]
    assert any(item["event_type"] == "generated" for item in items)
    assert any(item["event_type"] == "bookmarked" for item in items)


def test_quiz_attempt_and_json_export(tmp_path: Path):
    client = _make_client(tmp_path)
    client.post(
        "/api/auth/signup",
        json={
            "full_name": "Arjun Sharma",
            "email": "arjun@example.com",
            "password": "strong-password",
        },
    )
    client.post(
        "/api/auth/login",
        json={"email": "arjun@example.com", "password": "strong-password"},
    )

    with patch("backend.main.generate_full_content", return_value=dict(MOCK_GENERATED_CONTENT)):
        process = client.post(
            "/process",
            data={"source_type": "text", "text": "Photosynthesis notes", "subject": "Biology"},
        )
    note_id = process.json()["note_id"]

    note = client.get(f"/api/notes/{note_id}").json()
    quiz_id = note["quiz"]["id"]
    answers = {str(question["id"]): 1 for question in note["quiz"]["questions"]}

    attempt = client.post(
        f"/api/quizzes/{quiz_id}/attempts",
        json={"answers": answers, "completion_seconds": 87},
    )
    assert attempt.status_code == 201
    attempt_body = attempt.json()
    assert attempt_body["result"]["score"] >= 1
    assert "weak_topics" in attempt_body["result"]

    export = client.get(f"/export/json/{note_id}")
    assert export.status_code == 200
    assert export.headers["content-type"].startswith("application/json")
    exported = json.loads(export.content.decode("utf-8"))
    assert exported["note"]["id"] == note_id
    assert exported["quiz"]["attempts"][0]["completion_seconds"] == 87
