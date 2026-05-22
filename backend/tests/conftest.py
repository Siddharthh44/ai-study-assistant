from __future__ import annotations

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from backend.database import session_scope
from backend.main import create_app
from backend.models import User


@pytest.fixture()
def app(tmp_path) -> Iterator:
    database_path = tmp_path / "study_assistant_test.db"
    test_app = create_app(
        {
            "database_url": f"sqlite:///{database_path}",
            "process_inline": True,
            "testing": True,
            "session_secret": "test-secret",
            "session_https_only": False,
        }
    )
    yield test_app


@pytest.fixture()
def client(app) -> Iterator[TestClient]:
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture()
def sample_generation_result() -> dict:
    return {
        "title": "Cell Biology Basics",
        "summary": "Cells are the structural and functional unit of life.",
        "notes": "## Cells\nCells contain organelles that support life.",
        "key_concepts": [
            {"term": "Nucleus", "explanation": "Stores genetic material."},
            {"term": "Mitochondria", "explanation": "Produces usable cellular energy."},
        ],
        "difficulty_analysis": {"overall": "medium", "challenging_topics": ["Organelles"]},
        "study_recommendations": ["Review the summary before attempting the quiz."],
        "flashcards": [
            {"question": "What is the nucleus?", "answer": "The cell control center.", "topic": "Organelles"},
            {"question": "What do mitochondria make?", "answer": "ATP energy.", "topic": "Organelles"},
        ],
        "quiz": [
            {
                "question": "Which organelle stores genetic material?",
                "options": ["Ribosome", "Nucleus", "Golgi body", "Cytoplasm"],
                "answer": "Nucleus",
                "topic": "Organelles",
                "explanation": "The nucleus houses DNA.",
                "difficulty": "easy",
            },
            {
                "question": "What is the main role of mitochondria?",
                "options": ["Store water", "Make ATP", "Digest proteins", "Build cell walls"],
                "answer": "Make ATP",
                "topic": "Organelles",
                "explanation": "Mitochondria are the powerhouse of the cell.",
                "difficulty": "easy",
            },
        ],
    }


@pytest.fixture()
def auth_headers(client: TestClient) -> dict:
    response = client.post(
        "/api/auth/signup",
        json={
            "full_name": "Test User",
            "email": "user@example.com",
            "password": "strong-password",
        },
    )
    assert response.status_code == 201
    return {"x-test-user": "user@example.com"}


def create_user(email: str, password_hash: str, *, full_name: str = "Legacy User") -> int:
    with session_scope() as db:
        user = User(full_name=full_name, email=email, password_hash=password_hash)
        db.add(user)
        db.flush()
        return user.id
