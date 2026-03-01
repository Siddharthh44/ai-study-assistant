import os
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found")

# IMPORTANT: Force Gemini Developer API (v1beta)
client = genai.Client(
    api_key=GEMINI_API_KEY,
    http_options=types.HttpOptions(api_version="v1beta")
)

def test_connection():
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents="Say hello in one short sentence."
    )
    return response.text