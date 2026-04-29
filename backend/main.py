from inspect import signature
from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from .services.summary_service import generate_summary


app = FastAPI()


class SummaryRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str = Field(..., min_length=1)
    mode: Literal["short", "detailed", "exam"]


class SummaryResponse(BaseModel):
    summary: str
    key_points: list[str]
    #difficulty: str


def _generate_summary(text: str, mode: str):
    service_signature = signature(generate_summary)

    if "mode" in service_signature.parameters:
        return generate_summary(text, mode)

    return generate_summary(text)


def _build_summary_response(result: dict, requested_mode: str) -> SummaryResponse:
    if not isinstance(result, dict):
        raise HTTPException(status_code=502, detail="Summary service returned an invalid response.")

    if result.get("error"):
        raise HTTPException(status_code=502, detail=result["error"])

    summary = result.get("summary")
    key_points = result.get("key_points")
    #difficulty = result.get("difficulty") or requested_mode

    if not isinstance(summary, str) or not summary.strip():
        raise HTTPException(status_code=502, detail="Summary service returned an invalid summary.")

    if not isinstance(key_points, list) or not all(isinstance(point, str) for point in key_points):
        raise HTTPException(status_code=502, detail="Summary service returned invalid key points.")

    return SummaryResponse(
        summary=summary.strip(),
        key_points=key_points,
        #difficulty=difficulty,
    )


@app.get("/")
def read_root():
    return {"message": "AI Study Assistant Backend Running"}


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/summarize", response_model=SummaryResponse)
def summarize(request: SummaryRequest):
    try:
        result = generate_summary(request.text, request.mode)

        return SummaryResponse(
            summary=result["summary"],
            key_points=result["key_points"],
            #difficulty=result["difficulty"]
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
