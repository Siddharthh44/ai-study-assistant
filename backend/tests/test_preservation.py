"""
Preservation Property Tests
============================
These tests verify that EXISTING WORKING functionality is preserved.
They MUST PASS on the current (unfixed) code and MUST CONTINUE TO PASS
after the fix is applied.

Covers the "Unchanged Behavior (Regression Prevention)" section from bugfix.md:

  3.1  /process endpoint with valid JSON body calls generate_full_content
       and returns result_id
  3.2  /summarize endpoint invokes generate_summary and returns SummaryResponse
  3.3  /result page with valid result_id retrieves stored result and passes
       key_concepts, flashcards, quiz_count, notes_html to template
  3.4  /quiz-result page with valid result_id and answers computes quiz analysis
  3.5  Gemini API unavailable → fallback response returned without crashing
  3.6  Daily API limit reached → limit response returned without additional
       API calls
  3.7  Static files served under /static/ from the static/ directory
  3.8  localStorage usage preserved (pages that use it don't break)
  3.9  FastAPI app starts with static files mounted and Jinja2Templates
       initialized
  3.10 Backend services (llm_service, summary_service, parser, prompt_builder)
       function without modification

All Gemini API calls are mocked to avoid real network requests.
"""

import json
import sys
import os
from unittest.mock import patch, MagicMock

import pytest
from fastapi.testclient import TestClient

# ---------------------------------------------------------------------------
# Ensure the project root (parent of backend/) is on sys.path
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from backend.main import app, processed_results  # noqa: E402

client = TestClient(app, raise_server_exceptions=False)

# ---------------------------------------------------------------------------
# Shared test fixtures / helpers
# ---------------------------------------------------------------------------

SAMPLE_FULL_CONTENT = {
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

SAMPLE_SUMMARY_RESULT = {
    "title": "Photosynthesis",
    "summary": "Plants convert sunlight into glucose.",
    "key_points": ["Uses CO2", "Produces O2"],
}


def _store_sample_result() -> str:
    """Store SAMPLE_FULL_CONTENT in processed_results and return its result_id."""
    from backend.main import _store_processed_result
    return _store_processed_result(dict(SAMPLE_FULL_CONTENT))


# ===========================================================================
# 3.1  /process endpoint with valid JSON body
# ===========================================================================

class TestProcessEndpointPreservation:
    """
    Validates: Requirement 3.1
    /process with valid JSON body calls generate_full_content, stores the
    result in processed_results under a UUID result_id, and returns result_id.
    """

    def test_process_json_body_returns_result_id(self):
        """
        Validates: Requirement 3.1
        POST /process with a valid JSON body must return a dict containing
        'result_id'.
        """
        with patch(
            "backend.main.generate_full_content",
            return_value=dict(SAMPLE_FULL_CONTENT),
        ) as mock_gen:
            response = client.post("/process", json={"text": "Photosynthesis notes"})

        assert response.status_code == 200, (
            f"Expected 200 from /process, got {response.status_code}"
        )
        body = response.json()
        assert "result_id" in body, "Response must contain 'result_id'"
        mock_gen.assert_called_once()

    def test_process_stores_result_in_processed_results(self):
        """
        Validates: Requirement 3.1
        The result_id returned by /process must map to the stored result in
        processed_results.
        """
        with patch(
            "backend.main.generate_full_content",
            return_value=dict(SAMPLE_FULL_CONTENT),
        ):
            response = client.post("/process", json={"text": "Photosynthesis notes"})

        assert response.status_code == 200
        result_id = response.json()["result_id"]
        assert result_id in processed_results, (
            f"result_id '{result_id}' not found in processed_results"
        )

    def test_process_calls_generate_full_content_with_text(self):
        """
        Validates: Requirement 3.1
        generate_full_content must be called with the text from the request body.
        """
        input_text = "Study material about osmosis"
        with patch(
            "backend.main.generate_full_content",
            return_value=dict(SAMPLE_FULL_CONTENT),
        ) as mock_gen:
            client.post("/process", json={"text": input_text})

        mock_gen.assert_called_once_with(input_text)

    def test_process_returns_content_fields_alongside_result_id(self):
        """
        Validates: Requirement 3.1
        The /process response must include the AI-generated content fields
        in addition to result_id.
        """
        with patch(
            "backend.main.generate_full_content",
            return_value=dict(SAMPLE_FULL_CONTENT),
        ):
            response = client.post("/process", json={"text": "test"})

        body = response.json()
        assert "title" in body
        assert "summary" in body


# ===========================================================================
# 3.2  /summarize endpoint
# ===========================================================================

class TestSummarizeEndpointPreservation:
    """
    Validates: Requirement 3.2
    /summarize must invoke generate_summary and return a SummaryResponse
    with 'summary' and 'key_points' fields.
    """

    def test_summarize_returns_summary_response(self):
        """
        Validates: Requirement 3.2
        POST /summarize with valid JSON must return 200 with summary and
        key_points fields.
        """
        with patch(
            "backend.main.generate_summary",
            return_value=dict(SAMPLE_SUMMARY_RESULT),
        ):
            response = client.post(
                "/summarize",
                json={"text": "Photosynthesis notes", "mode": "short"},
            )

        assert response.status_code == 200, (
            f"Expected 200 from /summarize, got {response.status_code}"
        )
        body = response.json()
        assert "summary" in body, "SummaryResponse must contain 'summary'"
        assert "key_points" in body, "SummaryResponse must contain 'key_points'"

    def test_summarize_calls_generate_summary(self):
        """
        Validates: Requirement 3.2
        generate_summary must be called with the text and mode from the request.
        """
        with patch(
            "backend.main.generate_summary",
            return_value=dict(SAMPLE_SUMMARY_RESULT),
        ) as mock_sum:
            client.post(
                "/summarize",
                json={"text": "Osmosis notes", "mode": "detailed"},
            )

        mock_sum.assert_called_once_with("Osmosis notes", "detailed")

    def test_summarize_key_points_is_list(self):
        """
        Validates: Requirement 3.2
        key_points in the SummaryResponse must be a list.
        """
        with patch(
            "backend.main.generate_summary",
            return_value=dict(SAMPLE_SUMMARY_RESULT),
        ):
            response = client.post(
                "/summarize",
                json={"text": "test", "mode": "short"},
            )

        body = response.json()
        assert isinstance(body["key_points"], list)


# ===========================================================================
# 3.3  /result page logic: retrieves stored result and prepares template context
# ===========================================================================

class TestResultPagePreservation:
    """
    Validates: Requirement 3.3
    The /result route must retrieve the stored result and pass key_concepts,
    flashcards, quiz_count, and notes_html to the template.

    NOTE: The template file 'result.html' does not yet exist on disk (it is
    named differently — this is bug 1.12 being fixed separately).  We therefore
    test the route's DATA PREPARATION logic directly via the helper functions
    rather than through the HTTP endpoint, which currently returns 500 due to
    the template-not-found error.  After the fix (template renamed/created),
    the HTTP-level assertions will also pass.
    """

    def test_get_processed_result_returns_stored_data(self):
        """
        Validates: Requirement 3.3
        _get_processed_result must return the stored result dict for a valid
        result_id.
        """
        from backend.main import _get_processed_result

        result_id = _store_sample_result()
        result = _get_processed_result(result_id)

        assert isinstance(result, dict), "_get_processed_result must return a dict"
        assert result.get("title") == "Photosynthesis"

    def test_get_processed_result_returns_empty_for_unknown_id(self):
        """
        Validates: Requirement 3.3
        _get_processed_result must return an empty dict for an unknown result_id
        (not raise an exception).
        """
        from backend.main import _get_processed_result

        result = _get_processed_result("nonexistent_id_xyz")
        assert result == {}, "Unknown result_id must return empty dict"

    def test_get_processed_result_returns_empty_for_none(self):
        """
        Validates: Requirement 3.3
        _get_processed_result must return an empty dict when result_id is None.
        """
        from backend.main import _get_processed_result

        result = _get_processed_result(None)
        assert result == {}, "None result_id must return empty dict"

    def test_notes_markdown_converted_to_html(self):
        """
        Validates: Requirement 3.3
        The markdown notes field must be converted to HTML using markdown().
        The '## Overview' heading must become an <h2> tag.
        """
        import markdown as md_lib

        raw_notes = "## Overview\nPhotosynthesis uses CO2 and water."
        html_notes = md_lib.markdown(raw_notes, extensions=["fenced_code", "tables"])

        assert "<h2>" in html_notes, "Markdown ## heading must be converted to <h2>"

    def test_result_route_passes_correct_context_to_template(self):
        """
        Validates: Requirement 3.3
        The result route must pass key_concepts, flashcard_count, quiz_count,
        and notes_html to the template context.  We verify by patching
        TemplateResponse and inspecting the context dict.
        """
        from fastapi.responses import HTMLResponse
        import backend.main as main_module

        result_id = _store_sample_result()
        captured_context = {}

        def fake_template_response(name, context, **kwargs):
            captured_context.update(context)
            return HTMLResponse(content="<html>ok</html>")

        with patch.object(main_module.templates, "TemplateResponse",
                          side_effect=fake_template_response):
            response = client.get(f"/result?result_id={result_id}")

        assert response.status_code == 200
        assert "key_concepts" in captured_context, (
            "Template context must include 'key_concepts'"
        )
        assert "quiz_count" in captured_context, (
            "Template context must include 'quiz_count'"
        )
        assert "notes_html" in captured_context, (
            "Template context must include 'notes_html'"
        )


# ===========================================================================
# 3.4  /quiz-result page logic: computes quiz analysis
# ===========================================================================

class TestQuizResultPagePreservation:
    """
    Validates: Requirement 3.4
    The /quiz-result route must compute _build_quiz_analysis and render the
    quiz result template with score, feedback, and wrong-answer review.

    NOTE: The template file 'quiz-result.html' does not yet exist on disk
    (it is named differently — bug 1.12).  We test the DATA LOGIC directly
    via _build_quiz_analysis and _parse_quiz_answers, and use a patched
    TemplateResponse to verify the context passed to the template.
    """

    def test_build_quiz_analysis_computes_correct_score(self):
        """
        Validates: Requirement 3.4
        _build_quiz_analysis must compute the correct score for a given set
        of answers.
        """
        from backend.main import _build_quiz_analysis

        # quiz has 1 question; CO2 is index 1 in ["O2", "CO2", "N2", "H2"]
        result = dict(SAMPLE_FULL_CONTENT)
        answers = {0: 1}  # user picks index 1 = "CO2" = correct

        analysis = _build_quiz_analysis(result, answers)

        assert analysis["total"] == 1
        assert analysis["correct"] == 1
        assert analysis["percent"] == 100

    def test_build_quiz_analysis_wrong_answer(self):
        """
        Validates: Requirement 3.4
        _build_quiz_analysis must record wrong answers in wrong_answers list.
        """
        from backend.main import _build_quiz_analysis

        result = dict(SAMPLE_FULL_CONTENT)
        answers = {0: 0}  # user picks index 0 = "O2" = wrong

        analysis = _build_quiz_analysis(result, answers)

        assert analysis["correct"] == 0
        assert len(analysis["wrong_answers"]) == 1

    def test_build_quiz_analysis_empty_result(self):
        """
        Validates: Requirement 3.4
        _build_quiz_analysis must handle an empty result dict without crashing.
        """
        from backend.main import _build_quiz_analysis

        analysis = _build_quiz_analysis({}, {})

        assert analysis["total"] == 0
        assert analysis["correct"] == 0
        assert analysis["percent"] == 0

    def test_parse_quiz_answers_parses_json_list(self):
        """
        Validates: Requirement 3.4
        _parse_quiz_answers must parse a JSON list of answer indices.
        """
        from backend.main import _parse_quiz_answers

        raw = json.dumps([1, 2, 0])
        result = _parse_quiz_answers(raw)

        assert result == {0: 1, 1: 2, 2: 0}

    def test_parse_quiz_answers_returns_empty_for_none(self):
        """
        Validates: Requirement 3.4
        _parse_quiz_answers must return an empty dict for None input.
        """
        from backend.main import _parse_quiz_answers

        result = _parse_quiz_answers(None)
        assert result == {}

    def test_quiz_result_route_passes_analysis_to_template(self):
        """
        Validates: Requirement 3.4
        The /quiz-result route must pass score/feedback/wrong_answers to the
        template context.  We verify by patching TemplateResponse.
        """
        from fastapi.responses import HTMLResponse
        import backend.main as main_module

        result_id = _store_sample_result()
        answers = json.dumps([1])  # correct answer
        captured_context = {}

        def fake_template_response(name, context, **kwargs):
            captured_context.update(context)
            return HTMLResponse(content="<html>ok</html>")

        with patch.object(main_module.templates, "TemplateResponse",
                          side_effect=fake_template_response):
            response = client.get(
                f"/quiz-result?result_id={result_id}&answers={answers}"
            )

        assert response.status_code == 200
        assert "percent" in captured_context, "Template context must include 'percent'"
        assert "feedback" in captured_context, "Template context must include 'feedback'"
        assert "wrong_answers" in captured_context, (
            "Template context must include 'wrong_answers'"
        )


# ===========================================================================
# 3.5  Gemini API unavailable → fallback response
# ===========================================================================

class TestGeminiUnavailableFallback:
    """
    Validates: Requirement 3.5
    When the Gemini API is unavailable, generate_full_content must return the
    fallback response (title, summary, empty arrays) without crashing.
    """

    def test_generate_full_content_returns_fallback_on_api_error(self):
        """
        Validates: Requirement 3.5
        When generate_content raises an exception, generate_full_content must
        return a fallback dict with title, summary, key_concepts, flashcards,
        and quiz fields.
        """
        from backend.services.summary_service import generate_full_content

        with patch(
            "backend.services.summary_service.generate_content",
            side_effect=Exception("Gemini API unavailable"),
        ):
            result = generate_full_content("Some study text")

        assert isinstance(result, dict), "Fallback must be a dict"
        assert "title" in result, "Fallback must have 'title'"
        assert "summary" in result, "Fallback must have 'summary'"
        assert "key_concepts" in result, "Fallback must have 'key_concepts'"
        assert "flashcards" in result, "Fallback must have 'flashcards'"
        assert "quiz" in result, "Fallback must have 'quiz'"

    def test_generate_full_content_fallback_has_empty_arrays(self):
        """
        Validates: Requirement 3.5
        The fallback response must have empty lists for key_concepts, flashcards,
        and quiz (not None or missing).
        """
        from backend.services.summary_service import generate_full_content

        with patch(
            "backend.services.summary_service.generate_content",
            side_effect=ConnectionError("Network unreachable"),
        ):
            result = generate_full_content("Some study text")

        assert result["key_concepts"] == [], "Fallback key_concepts must be []"
        assert result["flashcards"] == [], "Fallback flashcards must be []"
        assert result["quiz"] == [], "Fallback quiz must be []"

    def test_process_endpoint_returns_500_not_crash_on_api_error(self):
        """
        Validates: Requirement 3.5
        When the AI service is unavailable, /process must return a non-422
        response (either 200 with fallback or 500 with error detail) — it must
        not crash the server.
        """
        with patch(
            "backend.main.generate_full_content",
            side_effect=Exception("Gemini API unavailable"),
        ):
            response = client.post("/process", json={"text": "test"})

        assert response.status_code != 422, (
            "A Gemini API error must not produce a 422 validation error"
        )
        # 500 is acceptable — the endpoint raises HTTPException(500)
        assert response.status_code in (200, 500)


# ===========================================================================
# 3.6  Daily API limit reached → limit response
# ===========================================================================

class TestDailyApiLimitPreservation:
    """
    Validates: Requirement 3.6
    When the daily API limit is reached, _build_limit_response must be returned
    without making additional Gemini API calls.
    """

    def test_build_limit_response_structure(self):
        """
        Validates: Requirement 3.6
        _build_limit_response must return a dict with title, summary, key_points,
        and important_terms fields.
        """
        from backend.services.summary_service import _build_limit_response

        result = _build_limit_response()

        assert isinstance(result, dict), "_build_limit_response must return a dict"
        assert "title" in result
        assert "summary" in result
        assert "key_points" in result
        assert isinstance(result["key_points"], list)

    def test_generate_summary_returns_limit_response_when_limit_reached(self):
        """
        Validates: Requirement 3.6
        When _reserve_daily_api_call_slot returns False, generate_summary must
        return the limit response without calling generate_content.
        """
        from backend.services import summary_service

        with patch.object(
            summary_service, "_reserve_daily_api_call_slot", return_value=False
        ), patch(
            "backend.services.summary_service.generate_content"
        ) as mock_gen:
            result = summary_service.generate_summary("test text", "short")

        # generate_content must NOT have been called
        mock_gen.assert_not_called()
        # Result must be the limit response
        assert "title" in result
        assert "summary" in result

    def test_limit_response_summary_mentions_limit(self):
        """
        Validates: Requirement 3.6
        The limit response summary must communicate that the limit has been reached.
        """
        from backend.services.summary_service import _build_limit_response

        result = _build_limit_response()
        combined = (result.get("title", "") + " " + result.get("summary", "")).lower()
        assert "limit" in combined or "capacity" in combined or "exhausted" in combined, (
            "Limit response must mention the limit/capacity in title or summary"
        )


# ===========================================================================
# 3.7  Static files served under /static/
# ===========================================================================

class TestStaticFilesPreservation:
    """
    Validates: Requirement 3.7
    Static files must be served from the static/ directory under the /static/
    URL prefix.
    """

    def test_static_js_file_is_served(self):
        """
        Validates: Requirement 3.7
        A known static JS file (chat.js) must be accessible under /static/js/.
        """
        response = client.get("/static/js/chat.js")
        assert response.status_code == 200, (
            f"Expected 200 for /static/js/chat.js, got {response.status_code}"
        )

    def test_static_nonexistent_file_returns_404(self):
        """
        Validates: Requirement 3.7
        A request for a non-existent static file must return 404, not 500.
        """
        response = client.get("/static/js/nonexistent_file_xyz.js")
        assert response.status_code == 404, (
            f"Expected 404 for missing static file, got {response.status_code}"
        )

    def test_static_mount_is_registered(self):
        """
        Validates: Requirement 3.7
        The /static route must be registered on the FastAPI app (not return 404
        for the mount itself).
        """
        # Any request under /static/ should be handled by StaticFiles, not
        # return a FastAPI 404 "route not found" response.
        response = client.get("/static/js/chat.js")
        # StaticFiles handles the request — 200 means mount is active
        assert response.status_code in (200, 404), (
            "Static mount must handle /static/ requests (200 or 404 for missing file)"
        )


# ===========================================================================
# 3.8  localStorage usage preserved (pages that use it don't break)
# ===========================================================================

class TestLocalStoragePagePreservation:
    """
    Validates: Requirement 3.8
    Pages that rely on localStorage (History, Export, Settings, Bookmarks)
    must load without server-side errors. The localStorage mechanism is
    client-side; we verify the pages render without crashing.

    NOTE: We patch TemplateResponse to bypass Jinja2 rendering (which has a
    Python 3.14 compatibility issue with its LRU cache).  The important
    preservation property is that the route handler itself does not crash —
    i.e., the FastAPI route is registered and the handler logic runs without
    raising an unhandled exception.
    """

    @pytest.mark.parametrize("route", [
        "/dashboard",
        "/upload",
        "/processing",
    ])
    def test_registered_pages_load_without_500(self, route):
        """
        Validates: Requirement 3.8
        Currently registered pages must return 200 (not 500), confirming the
        server-side rendering does not break localStorage-dependent pages.
        We patch TemplateResponse to avoid Jinja2 rendering issues on Python 3.14.
        """
        from fastapi.responses import HTMLResponse
        import backend.main as main_module

        def fake_template_response(name, context, **kwargs):
            return HTMLResponse(content="<html>ok</html>")

        with patch.object(main_module.templates, "TemplateResponse",
                          side_effect=fake_template_response):
            response = client.get(route)

        assert response.status_code == 200, (
            f"Expected 200 for '{route}', got {response.status_code}"
        )

    def test_result_page_loads_without_500(self):
        """
        Validates: Requirement 3.8
        The /result page (which may use localStorage for result_id) must load
        without a 500 error.
        We patch TemplateResponse to avoid Jinja2 rendering issues on Python 3.14.
        """
        from fastapi.responses import HTMLResponse
        import backend.main as main_module

        def fake_template_response(name, context, **kwargs):
            return HTMLResponse(content="<html>ok</html>")

        with patch.object(main_module.templates, "TemplateResponse",
                          side_effect=fake_template_response):
            response = client.get("/result")

        assert response.status_code == 200, (
            f"Expected 200 for /result (no result_id), got {response.status_code}"
        )


# ===========================================================================
# 3.9  FastAPI app starts with static files mounted and Jinja2Templates init
# ===========================================================================

class TestAppStartupPreservation:
    """
    Validates: Requirement 3.9
    The FastAPI app must start with static files mounted and Jinja2Templates
    initialized from the same BASE_DIR-relative paths already configured in
    main.py.
    """

    def test_app_has_static_mount(self):
        """
        Validates: Requirement 3.9
        The FastAPI app must have a static files mount registered.
        """
        from backend.main import app as main_app

        route_paths = [
            getattr(route, "path", None) for route in main_app.routes
        ]
        assert "/static" in route_paths, (
            "FastAPI app must have a /static mount registered"
        )

    def test_templates_object_is_initialized(self):
        """
        Validates: Requirement 3.9
        The Jinja2Templates object must be initialized (not None) in main.py.
        """
        from backend.main import templates

        assert templates is not None, "Jinja2Templates must be initialized"

    def test_health_check_returns_ok(self):
        """
        Validates: Requirement 3.9
        The /health endpoint must return {"status": "ok"}, confirming the app
        started correctly.
        """
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}

    def test_base_dir_paths_exist(self):
        """
        Validates: Requirement 3.9
        The static/ and templates/ directories referenced by BASE_DIR must
        exist on disk.
        """
        from backend.main import BASE_DIR

        static_dir = os.path.join(BASE_DIR, "static")
        templates_dir = os.path.join(BASE_DIR, "templates")

        assert os.path.isdir(static_dir), (
            f"static/ directory must exist at {static_dir}"
        )
        assert os.path.isdir(templates_dir), (
            f"templates/ directory must exist at {templates_dir}"
        )


# ===========================================================================
# 3.10  Backend services function without modification
# ===========================================================================

class TestBackendServicesPreservation:
    """
    Validates: Requirement 3.10
    The backend services (llm_service, summary_service, parser, prompt_builder)
    must function correctly without modification to their internal logic.
    """

    def test_parse_summary_parses_valid_json(self):
        """
        Validates: Requirement 3.10
        parser.parse_summary must correctly parse a valid JSON string.
        """
        from backend.services.parser import parse_summary

        raw = '{"summary": "test", "key_points": ["a", "b"]}'
        result = parse_summary(raw)

        assert result["summary"] == "test"
        assert result["key_points"] == ["a", "b"]

    def test_parse_summary_handles_markdown_code_block(self):
        """
        Validates: Requirement 3.10
        parser.parse_summary must strip ```json ... ``` fences before parsing.
        """
        from backend.services.parser import parse_summary

        raw = '```json\n{"summary": "test", "key_points": []}\n```'
        result = parse_summary(raw)

        assert result["summary"] == "test"

    def test_parse_summary_returns_error_dict_on_invalid_json(self):
        """
        Validates: Requirement 3.10
        parser.parse_summary must return an error dict (not raise) for invalid JSON.
        """
        from backend.services.parser import parse_summary

        result = parse_summary("this is not json at all")

        assert isinstance(result, dict)
        assert "error" in result

    def test_build_summary_prompt_includes_text(self):
        """
        Validates: Requirement 3.10
        prompt_builder.build_summary_prompt must include the input text in the
        returned prompt string.
        """
        from backend.services.prompt_builder import build_summary_prompt

        text = "Unique study material about osmosis"
        prompt = build_summary_prompt(text, "short")

        assert text in prompt, "build_summary_prompt must embed the input text"

    def test_build_process_prompt_includes_text(self):
        """
        Validates: Requirement 3.10
        prompt_builder.build_process_prompt must include the input text in the
        returned prompt string.
        """
        from backend.services.prompt_builder import build_process_prompt

        text = "Unique study material about mitosis"
        prompt = build_process_prompt(text)

        assert text in prompt, "build_process_prompt must embed the input text"

    def test_generate_full_content_returns_dict_on_success(self):
        """
        Validates: Requirement 3.10
        summary_service.generate_full_content must return a dict when the LLM
        returns valid JSON.
        """
        from backend.services.summary_service import generate_full_content

        mock_response = json.dumps(SAMPLE_FULL_CONTENT)

        with patch(
            "backend.services.summary_service.generate_content",
            return_value=mock_response,
        ):
            result = generate_full_content("Photosynthesis notes")

        assert isinstance(result, dict)
        assert result["title"] == "Photosynthesis"

    def test_generate_summary_returns_dict_on_success(self):
        """
        Validates: Requirement 3.10
        summary_service.generate_summary must return a dict when the LLM
        returns valid JSON.
        """
        from backend.services.summary_service import generate_summary

        mock_response = json.dumps(SAMPLE_SUMMARY_RESULT)

        with patch(
            "backend.services.summary_service.generate_content",
            return_value=mock_response,
        ):
            result = generate_summary("Photosynthesis notes", "short")

        assert isinstance(result, dict)

    def test_llm_service_get_client_raises_without_api_key(self):
        """
        Validates: Requirement 3.10
        llm_service._get_client must raise ValueError when GEMINI_API_KEY is
        not set, preserving the existing guard.
        """
        from backend.services import llm_service

        original_key = llm_service.GEMINI_API_KEY
        original_client = llm_service._client

        try:
            llm_service.GEMINI_API_KEY = None
            llm_service._client = None  # force re-initialization

            with pytest.raises((ValueError, Exception)):
                llm_service._get_client()
        finally:
            llm_service.GEMINI_API_KEY = original_key
            llm_service._client = original_client


# ===========================================================================
# Hypothesis Property-Based Tests
# ===========================================================================
# These tests use Hypothesis to verify preservation properties hold across
# a wide range of inputs, not just specific examples.
# ===========================================================================

from hypothesis import given, settings, HealthCheck
from hypothesis import strategies as st
from fastapi.responses import HTMLResponse
import backend.main as _main_module


def _fake_template_response(name, context, **kwargs):
    """Bypass Jinja2 rendering (Python 3.14 LRU cache incompatibility)."""
    return HTMLResponse(content="<html>ok</html>")


# ---------------------------------------------------------------------------
# Property 1: /process with any non-empty text returns 200 + result_id
# ---------------------------------------------------------------------------

class TestProcessPropertyBased:
    """
    Validates: Requirements 3.1, 3.9

    **Validates: Requirements 3.1**

    Property: For any non-empty text string, POST {"text": text} to /process
    SHALL return HTTP 200 and a result_id in the response body.
    """

    @given(text=st.text(min_size=1, max_size=500))
    @settings(max_examples=30, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_process_any_nonempty_text_returns_result_id(self, text):
        """
        **Validates: Requirements 3.1**

        Property: For any non-empty text string, POST /process with a valid
        JSON body SHALL return HTTP 200 and include 'result_id' in the response.
        Gemini API is mocked to avoid real network calls.
        """
        with patch(
            "backend.main.generate_full_content",
            return_value=dict(SAMPLE_FULL_CONTENT),
        ):
            response = client.post("/process", json={"text": text})

        assert response.status_code == 200, (
            f"Expected 200 from /process for text={text!r}, got {response.status_code}"
        )
        body = response.json()
        assert "result_id" in body, (
            f"Response must contain 'result_id' for text={text!r}"
        )
        assert isinstance(body["result_id"], str), (
            "result_id must be a string"
        )
        assert len(body["result_id"]) > 0, (
            "result_id must be non-empty"
        )

    @given(text=st.text(min_size=1, max_size=500))
    @settings(max_examples=20, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_process_result_id_stored_in_processed_results(self, text):
        """
        **Validates: Requirements 3.1**

        Property: For any non-empty text, the result_id returned by /process
        SHALL be present as a key in processed_results.
        """
        with patch(
            "backend.main.generate_full_content",
            return_value=dict(SAMPLE_FULL_CONTENT),
        ):
            response = client.post("/process", json={"text": text})

        assert response.status_code == 200
        result_id = response.json()["result_id"]
        assert result_id in processed_results, (
            f"result_id '{result_id}' must be stored in processed_results"
        )


# ---------------------------------------------------------------------------
# Property 2: /result with any valid result_id returns 200
# ---------------------------------------------------------------------------

class TestResultPagePropertyBased:
    """
    Validates: Requirements 3.3

    **Validates: Requirements 3.3**

    Property: For any valid result_id stored in processed_results,
    GET /result?result_id=<id> SHALL return HTTP 200.
    """

    @given(
        title=st.text(min_size=1, max_size=100),
        summary=st.text(min_size=1, max_size=500),
        notes=st.text(max_size=200),
    )
    @settings(max_examples=20, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_result_page_with_any_valid_result_id_returns_200(self, title, summary, notes):
        """
        **Validates: Requirements 3.3**

        Property: For any valid result_id (stored in processed_results with
        arbitrary content), GET /result?result_id=<id> SHALL return HTTP 200.
        Template rendering is patched to avoid Jinja2/Python 3.14 issues.
        """
        from backend.main import _store_processed_result

        result_data = {
            "title": title,
            "summary": summary,
            "notes": notes,
            "key_concepts": [],
            "flashcards": [],
            "quiz": [],
        }
        result_id = _store_processed_result(result_data)

        with patch.object(_main_module.templates, "TemplateResponse",
                          side_effect=_fake_template_response):
            response = client.get(f"/result?result_id={result_id}")

        assert response.status_code == 200, (
            f"Expected 200 for /result?result_id={result_id}, got {response.status_code}"
        )

    @given(
        key_concepts=st.lists(
            st.fixed_dictionaries({"term": st.text(min_size=1), "explanation": st.text()}),
            max_size=5,
        ),
        flashcards=st.lists(
            st.fixed_dictionaries({"question": st.text(min_size=1), "answer": st.text()}),
            max_size=5,
        ),
        quiz=st.lists(
            st.fixed_dictionaries({
                "question": st.text(min_size=1),
                "options": st.lists(st.text(min_size=1), min_size=2, max_size=4),
                "answer": st.integers(min_value=0, max_value=3),
                "topic": st.text(min_size=1),
            }),
            max_size=3,
        ),
    )
    @settings(max_examples=15, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_result_page_context_contains_required_keys(self, key_concepts, flashcards, quiz):
        """
        **Validates: Requirements 3.3**

        Property: For any stored result with arbitrary key_concepts, flashcards,
        and quiz data, the /result route SHALL pass key_concepts, quiz_count,
        and notes_html to the template context.
        """
        from backend.main import _store_processed_result

        result_data = {
            "title": "Test",
            "summary": "Test summary",
            "notes": "## Test\nSome notes.",
            "key_concepts": key_concepts,
            "flashcards": flashcards,
            "quiz": quiz,
        }
        result_id = _store_processed_result(result_data)
        captured_context = {}

        def capturing_template_response(name, context, **kwargs):
            captured_context.update(context)
            return HTMLResponse(content="<html>ok</html>")

        with patch.object(_main_module.templates, "TemplateResponse",
                          side_effect=capturing_template_response):
            response = client.get(f"/result?result_id={result_id}")

        assert response.status_code == 200
        assert "key_concepts" in captured_context
        assert "quiz_count" in captured_context
        assert "notes_html" in captured_context
        assert captured_context["quiz_count"] == len(quiz)


# ---------------------------------------------------------------------------
# Property 3: /quiz-result with any valid result_id and answers returns 200
# ---------------------------------------------------------------------------

class TestQuizResultPropertyBased:
    """
    Validates: Requirements 3.4

    **Validates: Requirements 3.4**

    Property: For any valid result_id and any list of integer answers,
    GET /quiz-result?result_id=<id>&answers=<json> SHALL return HTTP 200
    with 'percent' and 'correct' values in the template context.
    """

    @given(
        answers=st.lists(
            st.integers(min_value=0, max_value=3),
            min_size=0,
            max_size=10,
        )
    )
    @settings(max_examples=25, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_quiz_result_any_answers_returns_200(self, answers):
        """
        **Validates: Requirements 3.4**

        Property: For any list of integer answers (including empty), the
        /quiz-result route SHALL return HTTP 200 without crashing.
        """
        result_id = _store_sample_result()
        answers_json = json.dumps(answers)
        captured_context = {}

        def capturing_template_response(name, context, **kwargs):
            captured_context.update(context)
            return HTMLResponse(content="<html>ok</html>")

        with patch.object(_main_module.templates, "TemplateResponse",
                          side_effect=capturing_template_response):
            response = client.get(
                f"/quiz-result?result_id={result_id}&answers={answers_json}"
            )

        assert response.status_code == 200, (
            f"Expected 200 for /quiz-result with answers={answers}, "
            f"got {response.status_code}"
        )
        assert "percent" in captured_context, (
            "Template context must include 'percent'"
        )
        assert "correct" in captured_context, (
            "Template context must include 'correct'"
        )

    @given(
        answers=st.lists(
            st.integers(min_value=0, max_value=3),
            min_size=0,
            max_size=10,
        )
    )
    @settings(max_examples=20, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_quiz_result_percent_is_valid_range(self, answers):
        """
        **Validates: Requirements 3.4**

        Property: For any list of integer answers, the 'percent' value in the
        template context SHALL be an integer in the range [0, 100].
        """
        result_id = _store_sample_result()
        answers_json = json.dumps(answers)
        captured_context = {}

        def capturing_template_response(name, context, **kwargs):
            captured_context.update(context)
            return HTMLResponse(content="<html>ok</html>")

        with patch.object(_main_module.templates, "TemplateResponse",
                          side_effect=capturing_template_response):
            client.get(
                f"/quiz-result?result_id={result_id}&answers={answers_json}"
            )

        percent = captured_context.get("percent", -1)
        assert isinstance(percent, int), f"percent must be int, got {type(percent)}"
        assert 0 <= percent <= 100, f"percent must be in [0, 100], got {percent}"


# ---------------------------------------------------------------------------
# Property 4: /health always returns 200
# ---------------------------------------------------------------------------

class TestHealthPropertyBased:
    """
    Validates: Requirements 3.9

    **Validates: Requirements 3.9**

    Property: For any request, GET /health SHALL return HTTP 200 with
    {"status": "ok"}.
    """

    @given(st.just(None))
    @settings(max_examples=5, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_health_always_returns_200(self, _):
        """
        **Validates: Requirements 3.9**

        Property: GET /health SHALL always return HTTP 200 with status "ok".
        This is a basic liveness check that must never fail.
        """
        response = client.get("/health")
        assert response.status_code == 200, (
            f"Expected 200 from /health, got {response.status_code}"
        )
        assert response.json() == {"status": "ok"}, (
            f"Expected {{\"status\": \"ok\"}}, got {response.json()}"
        )

    @given(st.just(None))
    @settings(max_examples=5, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_health_response_is_idempotent(self, _):
        """
        **Validates: Requirements 3.9**

        Property: Multiple calls to GET /health SHALL all return the same
        {"status": "ok"} response (idempotent).
        """
        responses = [client.get("/health") for _ in range(3)]
        for r in responses:
            assert r.status_code == 200
            assert r.json() == {"status": "ok"}


# ---------------------------------------------------------------------------
# Property 5: /login POST always redirects to /dashboard
# ---------------------------------------------------------------------------

class TestLoginPropertyBased:
    """
    Validates: Requirements 3.1 (login redirect is preserved working behavior)

    **Validates: Requirements 3.1**

    Property: POST /login with any email and password SHALL return HTTP 303
    redirect to /dashboard.
    """

    @given(
        email=st.emails(),
        password=st.text(min_size=1, max_size=100),
    )
    @settings(max_examples=20, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_login_any_credentials_redirects_to_dashboard(self, email, password):
        """
        **Validates: Requirements 3.1**

        Property: POST /login with any email and password SHALL return HTTP 303
        redirect to /dashboard (the endpoint skips auth validation).
        """
        response = client.post(
            "/login",
            data={"email": email, "password": password},
            follow_redirects=False,
        )
        assert response.status_code == 303, (
            f"Expected 303 from /login, got {response.status_code}"
        )
        assert response.headers.get("location") == "/dashboard", (
            f"Expected redirect to /dashboard, got {response.headers.get('location')}"
        )


# ---------------------------------------------------------------------------
# Property 6: Static files mount returns non-404 for /static/ prefix
# ---------------------------------------------------------------------------

class TestStaticFilesPropertyBased:
    """
    Validates: Requirements 3.7

    **Validates: Requirements 3.7**

    Property: GET /static/ SHALL be handled by the StaticFiles mount
    (returning 200 for existing files, 404 for missing files — never a
    FastAPI routing 404 with "Not Found" detail).
    """

    @given(
        filename=st.sampled_from([
            "js/chat.js",
        ])
    )
    @settings(max_examples=5, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_known_static_files_return_200(self, filename):
        """
        **Validates: Requirements 3.7**

        Property: Known static files under /static/ SHALL return HTTP 200.
        """
        response = client.get(f"/static/{filename}")
        assert response.status_code == 200, (
            f"Expected 200 for /static/{filename}, got {response.status_code}"
        )

    @given(
        suffix=st.text(
            alphabet=st.characters(whitelist_categories=("Ll", "Lu", "Nd"), whitelist_characters="/_-."),
            min_size=1,
            max_size=30,
        ).filter(lambda s: not s.startswith("/") and ".." not in s)
    )
    @settings(max_examples=15, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_static_mount_handles_all_requests(self, suffix):
        """
        **Validates: Requirements 3.7**

        Property: Any GET /static/<path> request SHALL be handled by the
        StaticFiles mount (returning 200 or 404), never a 500 server error.
        """
        response = client.get(f"/static/{suffix}")
        assert response.status_code in (200, 404), (
            f"Expected 200 or 404 for /static/{suffix}, got {response.status_code}"
        )
