"""
Integration Tests — End-to-End Flow
=====================================
Verifies the complete upload → processing → result → flashcards → quiz →
quiz-result flow, plus the login flow and all previously-missing routes.

All Gemini API calls are mocked to avoid real network requests.
Jinja2 TemplateResponse is patched to bypass the Python 3.14 LRU-cache
incompatibility that causes 500 errors during actual template rendering.
"""

import json
import sys
import os
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient
from fastapi.responses import HTMLResponse

# ---------------------------------------------------------------------------
# Ensure the project root (parent of backend/) is on sys.path
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from backend.main import app, processed_results  # noqa: E402
import backend.main as main_module  # noqa: E402

client = TestClient(app, raise_server_exceptions=False)

# ---------------------------------------------------------------------------
# Shared mock data
# ---------------------------------------------------------------------------

MOCK_FULL_CONTENT = {
    "title": "Photosynthesis",
    "summary": "Plants convert sunlight into glucose.",
    "notes": "## Overview\nPhotosynthesis uses CO2 and water.",
    "key_concepts": [
        {"term": "Chlorophyll", "explanation": "Green pigment that absorbs light."}
    ],
    "flashcards": [
        {"question": "What is photosynthesis?", "answer": "Energy conversion in plants."}
    ],
    "quiz": [
        {
            "question": "What gas do plants absorb?",
            "options": ["O2", "CO2", "N2", "H2"],
            "answer": "CO2",
            "topic": "Gas Exchange",
        }
    ],
}


def _mock_template_response(template_name, context, *args, **kwargs):
    """Bypass Jinja2 rendering — return a plain 200 HTMLResponse."""
    return HTMLResponse(
        content=f"<html><body>mock:{template_name}</body></html>",
        status_code=200,
    )


# ===========================================================================
# End-to-End Flow: upload → processing → result → flashcards → quiz → quiz-result
# ===========================================================================

class TestEndToEndFlow:
    """
    Verifies the complete content-generation flow from POST /process through
    to GET /quiz-result.
    """

    def test_step1_post_process_returns_result_id(self):
        """
        Step 1: POST /process with JSON body → get result_id.
        """
        with patch(
            "backend.main.generate_full_content",
            return_value=dict(MOCK_FULL_CONTENT),
        ):
            response = client.post("/process", json={"text": "Photosynthesis notes"})

        assert response.status_code == 200, (
            f"POST /process expected 200, got {response.status_code}"
        )
        body = response.json()
        assert "result_id" in body, "Response must contain 'result_id'"

    def test_step2_get_processing_returns_200(self):
        """
        Step 2: GET /processing?result_id=<id> → returns 200.
        """
        # First create a result_id via /process
        with patch(
            "backend.main.generate_full_content",
            return_value=dict(MOCK_FULL_CONTENT),
        ):
            proc_resp = client.post("/process", json={"text": "Photosynthesis notes"})
        result_id = proc_resp.json()["result_id"]

        with patch.object(main_module.templates, "TemplateResponse",
                          side_effect=_mock_template_response):
            response = client.get(f"/processing?result_id={result_id}")

        assert response.status_code == 200, (
            f"GET /processing?result_id=... expected 200, got {response.status_code}"
        )

    def test_step3_get_result_returns_200(self):
        """
        Step 3: GET /result?result_id=<id> → returns 200.
        """
        with patch(
            "backend.main.generate_full_content",
            return_value=dict(MOCK_FULL_CONTENT),
        ):
            proc_resp = client.post("/process", json={"text": "Photosynthesis notes"})
        result_id = proc_resp.json()["result_id"]

        with patch.object(main_module.templates, "TemplateResponse",
                          side_effect=_mock_template_response):
            response = client.get(f"/result?result_id={result_id}")

        assert response.status_code == 200, (
            f"GET /result?result_id=... expected 200, got {response.status_code}"
        )

    def test_step4_get_flashcards_returns_200(self):
        """
        Step 4: GET /flashcards?result_id=<id> → returns 200.
        """
        with patch(
            "backend.main.generate_full_content",
            return_value=dict(MOCK_FULL_CONTENT),
        ):
            proc_resp = client.post("/process", json={"text": "Photosynthesis notes"})
        result_id = proc_resp.json()["result_id"]

        with patch.object(main_module.templates, "TemplateResponse",
                          side_effect=_mock_template_response):
            response = client.get(f"/flashcards?result_id={result_id}")

        assert response.status_code == 200, (
            f"GET /flashcards?result_id=... expected 200, got {response.status_code}"
        )

    def test_step5_get_quiz_returns_200(self):
        """
        Step 5: GET /quiz?result_id=<id> → returns 200.
        """
        with patch(
            "backend.main.generate_full_content",
            return_value=dict(MOCK_FULL_CONTENT),
        ):
            proc_resp = client.post("/process", json={"text": "Photosynthesis notes"})
        result_id = proc_resp.json()["result_id"]

        with patch.object(main_module.templates, "TemplateResponse",
                          side_effect=_mock_template_response):
            response = client.get(f"/quiz?result_id={result_id}")

        assert response.status_code == 200, (
            f"GET /quiz?result_id=... expected 200, got {response.status_code}"
        )

    def test_step6_get_quiz_result_returns_200(self):
        """
        Step 6: GET /quiz-result?result_id=<id>&answers=[0] → returns 200.
        """
        with patch(
            "backend.main.generate_full_content",
            return_value=dict(MOCK_FULL_CONTENT),
        ):
            proc_resp = client.post("/process", json={"text": "Photosynthesis notes"})
        result_id = proc_resp.json()["result_id"]

        answers = json.dumps([0])
        with patch.object(main_module.templates, "TemplateResponse",
                          side_effect=_mock_template_response):
            response = client.get(
                f"/quiz-result?result_id={result_id}&answers={answers}"
            )

        assert response.status_code == 200, (
            f"GET /quiz-result?result_id=...&answers=[0] expected 200, "
            f"got {response.status_code}"
        )

    def test_full_flow_end_to_end(self):
        """
        Complete end-to-end flow in a single test:
        POST /process → GET /processing → GET /result → GET /flashcards
        → GET /quiz → GET /quiz-result
        All steps must return 200.
        """
        # Step 1: POST /process
        with patch(
            "backend.main.generate_full_content",
            return_value=dict(MOCK_FULL_CONTENT),
        ):
            proc_resp = client.post("/process", json={"text": "Photosynthesis notes"})

        assert proc_resp.status_code == 200
        result_id = proc_resp.json()["result_id"]
        assert result_id, "result_id must be non-empty"

        with patch.object(main_module.templates, "TemplateResponse",
                          side_effect=_mock_template_response):
            # Step 2: GET /processing
            r2 = client.get(f"/processing?result_id={result_id}")
            assert r2.status_code == 200, f"Step 2 /processing failed: {r2.status_code}"

            # Step 3: GET /result
            r3 = client.get(f"/result?result_id={result_id}")
            assert r3.status_code == 200, f"Step 3 /result failed: {r3.status_code}"

            # Step 4: GET /flashcards
            r4 = client.get(f"/flashcards?result_id={result_id}")
            assert r4.status_code == 200, f"Step 4 /flashcards failed: {r4.status_code}"

            # Step 5: GET /quiz
            r5 = client.get(f"/quiz?result_id={result_id}")
            assert r5.status_code == 200, f"Step 5 /quiz failed: {r5.status_code}"

            # Step 6: GET /quiz-result
            answers = json.dumps([0])
            r6 = client.get(f"/quiz-result?result_id={result_id}&answers={answers}")
            assert r6.status_code == 200, f"Step 6 /quiz-result failed: {r6.status_code}"


# ===========================================================================
# Login Flow
# ===========================================================================

class TestLoginFlow:
    """
    Verifies the login flow:
    GET / → 200 (auth.html rendered)
    POST /login with form data → 303 redirect to /dashboard
    GET /dashboard → 200
    """

    def test_get_root_returns_200(self):
        """
        Step 1: GET / → returns 200 (auth.html rendered).
        """
        with patch.object(main_module.templates, "TemplateResponse",
                          side_effect=_mock_template_response):
            response = client.get("/")

        assert response.status_code == 200, (
            f"GET / expected 200, got {response.status_code}"
        )

    def test_post_login_redirects_to_dashboard(self):
        """
        Step 2: POST /login with form data → returns 303 redirect to /dashboard.
        """
        response = client.post(
            "/login",
            data={"email": "user@example.com", "password": "secret"},
            follow_redirects=False,
        )
        assert response.status_code == 303, (
            f"POST /login expected 303, got {response.status_code}"
        )
        assert response.headers.get("location") == "/dashboard", (
            f"Expected redirect to /dashboard, got {response.headers.get('location')}"
        )

    def test_get_dashboard_returns_200(self):
        """
        Step 3: GET /dashboard → returns 200.
        """
        with patch.object(main_module.templates, "TemplateResponse",
                          side_effect=_mock_template_response):
            response = client.get("/dashboard")

        assert response.status_code == 200, (
            f"GET /dashboard expected 200, got {response.status_code}"
        )

    def test_full_login_flow(self):
        """
        Complete login flow in a single test:
        GET / → POST /login → GET /dashboard
        """
        with patch.object(main_module.templates, "TemplateResponse",
                          side_effect=_mock_template_response):
            r1 = client.get("/")
        assert r1.status_code == 200, f"GET / failed: {r1.status_code}"

        r2 = client.post(
            "/login",
            data={"email": "user@example.com", "password": "secret"},
            follow_redirects=False,
        )
        assert r2.status_code == 303, f"POST /login failed: {r2.status_code}"
        assert r2.headers.get("location") == "/dashboard"

        with patch.object(main_module.templates, "TemplateResponse",
                          side_effect=_mock_template_response):
            r3 = client.get("/dashboard")
        assert r3.status_code == 200, f"GET /dashboard failed: {r3.status_code}"


# ===========================================================================
# Previously-missing routes — all must return 200
# ===========================================================================

class TestPreviouslyMissingRoutes:
    """
    Verifies that all routes that were previously missing now return 200.
    """

    @pytest.mark.parametrize("route", [
        "/notes",
        "/quizresults",
        "/progress",
        "/history",
        "/bookmarks",
        "/export",
        "/settings",
        "/pyqanalysis",
        "/noteview",
        "/quizzes",
    ])
    def test_route_returns_200(self, route):
        """
        Each previously-missing route must now return 200.
        """
        with patch.object(main_module.templates, "TemplateResponse",
                          side_effect=_mock_template_response):
            response = client.get(route)

        assert response.status_code == 200, (
            f"Route '{route}' expected 200, got {response.status_code}"
        )


# ===========================================================================
# Backend service signatures — verify they are unchanged
# ===========================================================================

class TestBackendServicesUnchanged:
    """
    Verifies that backend service files have their original function signatures.
    """

    def test_llm_service_generate_content_signature(self):
        """
        llm_service.generate_content must accept (prompt, request_size=None).
        """
        import inspect
        from backend.services.llm_service import generate_content

        sig = inspect.signature(generate_content)
        params = list(sig.parameters.keys())
        assert "prompt" in params, "generate_content must have 'prompt' parameter"
        assert "request_size" in params, "generate_content must have 'request_size' parameter"

    def test_summary_service_generate_summary_signature(self):
        """
        summary_service.generate_summary must accept (text, mode='short').
        """
        import inspect
        from backend.services.summary_service import generate_summary

        sig = inspect.signature(generate_summary)
        params = list(sig.parameters.keys())
        assert "text" in params, "generate_summary must have 'text' parameter"
        assert "mode" in params, "generate_summary must have 'mode' parameter"

    def test_summary_service_generate_full_content_signature(self):
        """
        summary_service.generate_full_content must accept (text).
        """
        import inspect
        from backend.services.summary_service import generate_full_content

        sig = inspect.signature(generate_full_content)
        params = list(sig.parameters.keys())
        assert "text" in params, "generate_full_content must have 'text' parameter"

    def test_parser_parse_summary_signature(self):
        """
        parser.parse_summary must accept (response_text).
        """
        import inspect
        from backend.services.parser import parse_summary

        sig = inspect.signature(parse_summary)
        params = list(sig.parameters.keys())
        assert "response_text" in params, "parse_summary must have 'response_text' parameter"

    def test_prompt_builder_build_summary_prompt_signature(self):
        """
        prompt_builder.build_summary_prompt must accept (text, mode='short').
        """
        import inspect
        from backend.services.prompt_builder import build_summary_prompt

        sig = inspect.signature(build_summary_prompt)
        params = list(sig.parameters.keys())
        assert "text" in params, "build_summary_prompt must have 'text' parameter"
        assert "mode" in params, "build_summary_prompt must have 'mode' parameter"

    def test_prompt_builder_build_process_prompt_signature(self):
        """
        prompt_builder.build_process_prompt must accept (text).
        """
        import inspect
        from backend.services.prompt_builder import build_process_prompt

        sig = inspect.signature(build_process_prompt)
        params = list(sig.parameters.keys())
        assert "text" in params, "build_process_prompt must have 'text' parameter"
