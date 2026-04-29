def build_summary_prompt(text: str, mode: str = "short") -> str:
    """
    Builds a prompt that asks the LLM to return structured study notes
    based on the selected mode.
    """

    if mode == "short":
        instruction = "Give a very concise summary in 3-4 lines."
    elif mode == "detailed":
        instruction = "Give a detailed explanation with clear concepts."
    elif mode == "exam":
        instruction = "Give exam-oriented bullet points for quick revision."
    else:
        instruction = "Give a balanced summary."

    prompt = f"""
You are an AI study assistant.

Analyze the following study material.

Instruction:
{instruction}

Return the response STRICTLY in this JSON format:

{{
"title": "Topic title",
"summary": "Short paragraph summary",
"key_points": ["point1", "point2", "point3"],
"important_terms": ["term1", "term2", "term3"]
}}

Study Material:
{text}

Return ONLY valid JSON.
"""

    return prompt