import logging
from dotenv import load_dotenv
from llm.gemini_flash_2_5 import call_gemini

from llm.llm_clients import (
   _get_client as gemini_client,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def generate_response(input_prompt: str) -> str:
    """
    Generate a response using  the Gemini LLM.
    """
    if not gemini_client:
        logger.error("Gemini skipped: API key not configured.")
        raise EnvironmentError("Gemini API key not set")

    result = call_gemini(input_prompt)
    if not result or not result.strip():
        raise RuntimeError("Gemini returned an empty response")

    logger.info("Response received from Gemini")
    return result.strip()

if __name__ == "__main__":
    load_dotenv()  # Load environment variables from .env file
    prompt = "What is the capital of France?"
    try:
        response = generate_response(prompt)
        print("Final Response:", response)
    except RuntimeError as e:
        logger.error(e)