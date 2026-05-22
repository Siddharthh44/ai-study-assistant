# AI Study Assistant

A production-ready AI-powered study platform built with FastAPI, Jinja templates, Tailwind CSS, and vanilla JavaScript.

It converts uploaded study material into structured notes, flashcards, quizzes, and progress insights with persistent user accounts and session authentication.

## Features

- Session-based authentication (`signup`, `login`, `logout`) with bcrypt password hashing.
- Upload support for `TXT`, `PDF`, and `DOCX` plus direct pasted text.
- AI-generated study content:
  - concise summary
  - structured notes
  - key concepts
  - flashcards
  - MCQ quizzes
- Persistent SQLite/SQLAlchemy data model for:
  - users
  - notes
  - flashcards
  - quizzes + attempts
  - bookmarks
  - history
  - exports
  - uploaded files
  - study sessions
  - analytics events
- Dynamic dashboard, progress analytics, and history tracking.
- Export endpoints for `PDF`, `TXT`, and `JSON`.
- Vercel-compatible serverless entrypoint (`api/index.py`).

## Tech Stack

- Backend: FastAPI, SQLAlchemy, Jinja2, Pydantic
- Frontend: HTML templates, Tailwind CSS (CDN), vanilla JavaScript
- Database: SQLite (local, `/tmp` fallback on Vercel)
- AI: Google Gemini via `google-genai`
- Auth: Starlette `SessionMiddleware` + passlib bcrypt

## Architecture

- `backend/main.py`: app creation, routes, page payload builders, API handlers
- `backend/models.py`: SQLAlchemy models and relationships
- `backend/database.py`: engine/session config and runtime DB defaults
- `backend/services/`: AI generation, prompt building, parsing, extraction, export helpers
- `templates/`: Jinja pages and shared partials
- `static/js/`: page-level frontend integration scripts
- `api/index.py`: Vercel entrypoint exposing `app`

## Local Development

### 1) Create and activate a virtual environment

```bash
python -m venv venv
# Windows PowerShell
venv\Scripts\Activate.ps1
# macOS/Linux
source venv/bin/activate
```

### 2) Install dependencies

```bash
pip install -r requirements.txt
```

### 3) Configure environment variables

Create `backend/.env`:

```env
GEMINI_API_KEY=your_gemini_api_key
SESSION_SECRET=replace_with_a_long_random_secret
```

Optional:

```env
DATABASE_URL=sqlite:///backend/study_assistant.db
SESSION_SAME_SITE=lax
SESSION_HTTPS_ONLY=false
SESSION_MAX_AGE_SECONDS=1209600
```

### 4) Run the app

```bash
uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

Open: [http://localhost:8000](http://localhost:8000)

## Deployment (Vercel)

1. Push this repository to GitHub.
2. Import it in Vercel.
3. Ensure the project root contains:
   - `vercel.json`
   - `runtime.txt`
   - `requirements.txt`
   - `api/index.py`
4. Set required environment variables in Vercel:
   - `GEMINI_API_KEY`
   - `SESSION_SECRET`
   - optional session flags (`SESSION_SAME_SITE`, `SESSION_MAX_AGE_SECONDS`)
5. Deploy.

### Vercel Notes

- In Vercel, default DB path falls back to `/tmp/study_assistant.db` when `DATABASE_URL` is not set.
- Session cookies run with `https_only=True` when `VERCEL` is present.
- Static assets are served by FastAPI (`/static`) via mounted `StaticFiles`.

## Environment Variables

- `GEMINI_API_KEY` (required): Gemini API key for content generation.
- `SESSION_SECRET` (required in production): session signing secret.
- `DATABASE_URL` (optional): SQLAlchemy connection string.
- `SESSION_SAME_SITE` (optional, default: `lax`).
- `SESSION_HTTPS_ONLY` (optional; defaults to `true` on Vercel).
- `SESSION_MAX_AGE_SECONDS` (optional).
- `SUMMARY_CACHE_ENABLED` (optional, default: `true`).
- `SUMMARY_CACHE_TTL_SECONDS` (optional, default: `600`).
- `SUMMARY_DAILY_API_LIMIT` (optional, default: `20`).
- `SUMMARY_DAILY_API_LIMIT_BUFFER` (optional, default: `2`).

## Testing

Run all tests:

```bash
pytest -q
```

## Screenshots

- `docs/screenshots/dashboard.png`
- `docs/screenshots/upload.png`
- `docs/screenshots/note-view.png`
- `docs/screenshots/quiz.png`
- `docs/screenshots/results.png`

## Future Improvements

- Background queue workers for long-running AI jobs.
- Optional PostgreSQL production profile.
- Smarter spaced-repetition scheduling.
- Collaborative note sharing.
- Usage/observability dashboards for ops.
