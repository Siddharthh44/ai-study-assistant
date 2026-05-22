from __future__ import annotations

import io
import json
from typing import Any

from reportlab.lib.pagesizes import A4
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.pdfgen import canvas


def build_text_export(document: dict[str, Any]) -> str:
    note = document["note"]
    sections = [
        note["title"],
        "",
        f"Subject: {note['subject']}",
        f"Summary: {note['summary']}",
        "",
        "Notes",
        note["notes_markdown"],
        "",
        "Key Concepts",
    ]

    for concept in note["key_concepts"]:
        sections.append(f"- {concept['term']}: {concept['explanation']}")

    sections.append("")
    sections.append("Flashcards")
    for card in document["flashcards"]:
        sections.append(f"Q: {card['question']}")
        sections.append(f"A: {card['answer']}")
        sections.append("")

    quiz = document.get("quiz")
    if quiz:
        sections.append("Quiz")
        for index, question in enumerate(quiz["questions"], start=1):
            sections.append(f"{index}. {question['question']}")
            for option in question["options"]:
                sections.append(f"   - {option}")
            sections.append(f"   Answer: {question['correct_answer']}")
            if question.get("explanation"):
                sections.append(f"   Explanation: {question['explanation']}")
            sections.append("")

    attempts = quiz.get("attempts", []) if quiz else []
    if attempts:
        latest = attempts[0]
        sections.append("Latest Quiz Result")
        sections.append(f"Score: {latest['score']} / {latest['total_questions']} ({latest['percent']}%)")
        sections.append(f"Time: {latest['completion_seconds']} seconds")

    return "\n".join(sections).strip() + "\n"


def build_json_export(document: dict[str, Any]) -> bytes:
    return json.dumps(document, indent=2).encode("utf-8")


def build_pdf_export(document: dict[str, Any], *, header: str | None = None) -> bytes:
    buffer = io.BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4
    x = 50
    y = height - 60

    header_text = header or document["note"]["title"]
    pdf.setFont("Helvetica-Bold", 18)
    pdf.drawString(x, y, header_text)
    y -= 30

    pdf.setFont("Helvetica", 11)
    lines = build_text_export(document).splitlines()
    for line in lines:
        if y < 50:
            pdf.showPage()
            pdf.setFont("Helvetica", 11)
            y = height - 50

        wrapped_lines = _wrap_line(line, max_width=width - 100, font_name="Helvetica", font_size=11)
        for wrapped in wrapped_lines:
            pdf.drawString(x, y, wrapped)
            y -= 16
            if y < 50:
                pdf.showPage()
                pdf.setFont("Helvetica", 11)
                y = height - 50

    pdf.save()
    return buffer.getvalue()


def _wrap_line(text: str, *, max_width: float, font_name: str, font_size: int) -> list[str]:
    if not text:
        return [""]

    words = text.split()
    lines: list[str] = []
    current = words[0]

    for word in words[1:]:
        candidate = f"{current} {word}"
        if stringWidth(candidate, font_name, font_size) <= max_width:
            current = candidate
        else:
            lines.append(current)
            current = word

    lines.append(current)
    return lines
