import json
import re


def parse_summary(response_text: str):
    """
    Converts LLM response into JSON.
    Handles markdown code blocks like ```json ... ```
    """

    try:
        cleaned = re.sub(r"```json|```", "", response_text).strip()

        parsed = json.loads(cleaned)

        return parsed

    except Exception:
        return {
            "error": "Failed to parse AI response",
            "raw_output": response_text
        }