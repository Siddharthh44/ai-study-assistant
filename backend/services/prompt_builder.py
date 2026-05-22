def build_summary_prompt(text: str, mode: str = "short") -> str:
    if mode == "short":
        instruction = "Give a very concise summary in 3-4 lines."
    elif mode == "detailed":
        instruction = "Give a detailed explanation with clear concepts."
    elif mode == "exam":
        instruction = "Give exam-oriented bullet points for quick revision."
    else:
        instruction = "Give a balanced summary."

    return f"""
You are an AI study assistant.

Instruction:
{instruction}

Return JSON:
{{
"title": "...",
"summary": "...",
"key_points": ["..."]
}}

Text:
{text}
"""


def build_process_prompt(text: str) -> str:
    return f"""
You are an advanced AI study assistant.
Analyze the study material and generate structured learning content.

Return STRICTLY valid JSON in this exact format:

{{
"title": "...",
"summary": "...",
"notes": "...",
"key_concepts": [
  {{ "term": "...", "explanation": "..." }}
],
"flashcards": [
  {{ "question": "...", "answer": "..." }}
],
"quiz": [
  {{
    "question": "...",
    "options": ["A","B","C","D"],
    "answer": "A"
  }}
]
}}

Text:
{text}

Return ONLY JSON.
"""
