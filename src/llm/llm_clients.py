
import os
from dotenv import load_dotenv
from google import genai


load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
MODEL = "gemini-2.5-flash"

_client = None

def _get_client() -> genai.Client:
    global _client
    if _client is None:
        if not GEMINI_API_KEY:
            raise EnvironmentError("Gemini API key not set.")
        _client = genai.Client(api_key=GEMINI_API_KEY)
    return _client