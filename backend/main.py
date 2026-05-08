import json
import os
from datetime import datetime
from uuid import uuid4
import markdown

from fastapi import FastAPI, HTTPException, Request, Form
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from pydantic import BaseModel, ConfigDict, Field

# OLD (kept for fallback)
from backend.services.summary_service import generate_summary

# NEW
from backend.services.summary_service import generate_full_content

app = FastAPI()

# -----------------------------
# PATH SETUP
# -----------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static")

templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))
processed_results: dict[str, dict] = {}

# -----------------------------
# MODELS
# -----------------------------

# OLD MODEL (kept)
class SummaryRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str = Field(..., min_length=1)
    mode: str


class SummaryResponse(BaseModel):
    summary: str
    key_points: list[str]


# NEW MODEL
class ProcessRequest(BaseModel):
    text: str


def _store_processed_result(result: dict) -> str:
    result = dict(result)
    result.setdefault("generated_at", datetime.now().strftime("%d %b %Y, %I:%M %p"))
    result_id = uuid4().hex
    processed_results[result_id] = result
    return result_id


def _get_processed_result(result_id: str | None) -> dict:
    if not result_id:
        return {}
    return processed_results.get(result_id, {})


def _get_note_lines(notes: str) -> list[str]:
    return [line.strip() for line in str(notes).splitlines() if line.strip()]


def _get_correct_index(question: dict) -> int:
    options = question.get("options", [])
    answer = question.get("answer")

    if isinstance(answer, int):
        return answer if 0 <= answer < len(options) else -1

    if isinstance(answer, str):
        for index, option in enumerate(options):
            if option == answer or option.startswith(answer):
                return index

    return -1


def _parse_quiz_answers(raw_answers: str | None) -> dict[int, int]:
    if not raw_answers:
        return {}

    try:
        parsed = json.loads(raw_answers)
    except (json.JSONDecodeError, TypeError):
        return {}

    if isinstance(parsed, list):
        return {
            index: value
            for index, value in enumerate(parsed)
            if isinstance(value, int)
        }

    if not isinstance(parsed, dict):
        return {}

    answers: dict[int, int] = {}
    for key, value in parsed.items():
        try:
            question_index = int(key)
        except (TypeError, ValueError):
            continue

        if isinstance(value, int):
            answers[question_index] = value

    return answers


def _build_quiz_feedback(percent: int) -> str:
    if percent > 80:
        return "Great work! You're close to mastery."
    if percent > 60:
        return "Nice! A bit more revision and you'll ace it."
    return "Keep practicing - you're improving."


def _build_quiz_analysis(result: dict, answers: dict[int, int]) -> dict:
    questions = result.get("quiz") if isinstance(result.get("quiz"), list) else []
    total = len(questions)
    correct = 0
    topic_stats: dict[str, dict[str, int]] = {}
    wrong_answers: list[dict[str, str]] = []

    for index, question in enumerate(questions):
        options = question.get("options") if isinstance(question.get("options"), list) else []
        correct_index = _get_correct_index(question)
        user_index = answers.get(index)
        topic = question.get("topic") or "General"

        if topic not in topic_stats:
            topic_stats[topic] = {"total": 0, "correct": 0}

        topic_stats[topic]["total"] += 1

        if user_index == correct_index and correct_index >= 0:
            correct += 1
            topic_stats[topic]["correct"] += 1
            continue

        wrong_answers.append(
            {
                "question": str(question.get("question", "")),
                "user_answer": options[user_index] if isinstance(user_index, int) and 0 <= user_index < len(options) else "Not answered",
                "correct_answer": options[correct_index] if 0 <= correct_index < len(options) else "",
                "explanation": str(question.get("explanation") or "No explanation available"),
            }
        )

    percent = round((correct / total) * 100) if total else 0
    topics_to_review = []

    for topic, stats in topic_stats.items():
        accuracy = round((stats["correct"] / stats["total"]) * 100) if stats["total"] else 0
        if accuracy < 100:
            topics_to_review.append({"name": topic, "accuracy": accuracy})

    return {
        "questions": questions,
        "correct": correct,
        "total": total,
        "percent": percent,
        "feedback": _build_quiz_feedback(percent),
        "topics_to_review": topics_to_review,
        "wrong_answers": wrong_answers,
    }


# -----------------------------
# HEALTH CHECK
# -----------------------------
@app.get("/health")
def health_check():
    return {"status": "ok"}


# -----------------------------
# OLD ROUTE (optional fallback)
# -----------------------------
@app.post("/summarize", response_model=SummaryResponse)
def summarize(request: SummaryRequest):
    try:
        result = generate_summary(request.text, request.mode)

        return SummaryResponse(
            summary=result["summary"],
            key_points=result["key_points"],
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# -----------------------------
# NEW MAIN ROUTE (IMPORTANT)
# -----------------------------
@app.post("/process")
def process(request: ProcessRequest):
    try:
        result = generate_full_content(request.text)

        if not isinstance(result, dict):
            raise HTTPException(status_code=500, detail="Invalid response from AI")

        if result.get("error"):
            raise HTTPException(status_code=500, detail=result["error"])

        result_id = _store_processed_result(result)
        return {"result_id": result_id, **result}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# -----------------------------
# PAGES
# -----------------------------
@app.get("/", response_class=HTMLResponse)
def login_page(request: Request):
    return templates.TemplateResponse("login.html", {"request": request})

@app.post("/login")
def login(email: str = Form(...), password: str = Form(...)):
    # TEMP: skip auth
    return RedirectResponse(url="/dashboard", status_code=303)

@app.get("/dashboard", response_class=HTMLResponse)
def dashboard_page(request: Request):
    return templates.TemplateResponse("dashboard.html", {"request": request})

@app.get("/upload", response_class=HTMLResponse)
def upload_page(request: Request):
    return templates.TemplateResponse("upload.html", {"request": request})

@app.get("/processing", response_class=HTMLResponse)
def processing_page(request: Request):
    return templates.TemplateResponse("processing.html", {"request": request})

@app.get("/result", response_class=HTMLResponse)
def result_page(request: Request, result_id: str | None = None):
    result = _get_processed_result(result_id)

    key_concepts = result.get("key_concepts") if isinstance(result.get("key_concepts"), list) else []
    flashcards = result.get("flashcards") if isinstance(result.get("flashcards"), list) else []
    quiz = result.get("quiz") if isinstance(result.get("quiz"), list) else []
    raw_notes = result.get("notes", "")
    html_notes = markdown.markdown(raw_notes, extensions=["fenced_code", "tables"])

    return templates.TemplateResponse(
        "result.html",
        {
            "request": request,
            "result": result,
            "result_id": result_id,
            "notes_html": html_notes,
            "key_concepts": key_concepts,
            "flashcard_count": len(flashcards),
            "quiz_count": len(quiz),
        },
    )

@app.get("/flashcards", response_class=HTMLResponse)
def flashcards_page(request: Request, result_id: str | None = None):
    result = _get_processed_result(result_id)
    flashcards = result.get("flashcards") if isinstance(result.get("flashcards"), list) else []
    return templates.TemplateResponse(
        "flashcard.html",
        {
            "request": request,
            "result_id": result_id,
            "flashcards": flashcards,
        },
    )

@app.get("/quiz", response_class=HTMLResponse)
def quiz_page(request: Request, result_id: str | None = None):
    result = _get_processed_result(result_id)
    quiz_questions = result.get("quiz") if isinstance(result.get("quiz"), list) else []
    return templates.TemplateResponse(
        "quiz.html",
        {
            "request": request,
            "result_id": result_id,
            "quiz_questions": quiz_questions,
        },
    )

@app.get("/quiz-result", response_class=HTMLResponse)
def quiz_result_page(request: Request, result_id: str | None = None, answers: str | None = None):
    result = _get_processed_result(result_id)
    analysis = _build_quiz_analysis(result, _parse_quiz_answers(answers))
    return templates.TemplateResponse(
        "quiz-result.html",
        {
            "request": request,
            "result_id": result_id,
            **analysis,
        },
    )

@app.get("/mynotes", response_class=HTMLResponse)
def mynotes_page(request: Request):

    # TEMP DATA (replace later with AI/backend output)
    notes = [
        {
            "id": "1",
            "title": "Photosynthesis & Plant Biology",
            "subject": "Biology",
            "date": "2h ago",
            "preview": "Photosynthesis is the process by which green plants use sunlight...",
            "flashcards": 12,
            "hasQuiz": True
        },
        {
            "id": "2",
            "title": "Organic Chemistry: Alkanes & Alkenes",
            "subject": "Chemistry",
            "date": "Yesterday",
            "preview": "Alkanes are saturated hydrocarbons...",
            "flashcards": 18,
            "hasQuiz": True
        },
        {
            "id": "3",
            "title": "World War II: Key Events & Timeline",
            "subject": "History",
            "date": "2 days ago",
            "preview": "World War II began in September 1939...",
            "flashcards": 24,
            "hasQuiz": False
        }
    ]

    return templates.TemplateResponse(
        "mynotes.html",
        {
            "request": request,
            "notes": notes
        }
    )