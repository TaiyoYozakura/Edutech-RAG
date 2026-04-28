from google.genai import types
from llm.llm_clients import _get_client, MODEL
import logging
logger = logging.getLogger(__name__)



def call_gemini(query: str) -> str:
    client = _get_client()
    response = client.models.generate_content(
        model=MODEL,
        contents=query,
        config=types.GenerateContentConfig(
            temperature=0.7,
            max_output_tokens=1024,
        )
    )
    res = (response.text or "").strip()
    logger.info(f"Gemini response: {res}")
    return res