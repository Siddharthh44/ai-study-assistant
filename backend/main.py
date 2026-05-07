import os

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

        return result

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
def result_page(request: Request):
    return templates.TemplateResponse("result.html", {"request": request})

@app.get("/flashcards", response_class=HTMLResponse)
def flashcards_page(request: Request):
    return templates.TemplateResponse("flashcards.html", {"request": request})

@app.get("/quiz", response_class=HTMLResponse)
def quiz_page(request: Request):
    return templates.TemplateResponse("quiz.html", {"request": request})

@app.get("/quiz-result", response_class=HTMLResponse)
def quiz_result_page(request: Request):
    return templates.TemplateResponse("quiz-result.html", {"request": request})