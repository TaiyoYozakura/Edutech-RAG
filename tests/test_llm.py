import pytest
from unittest.mock import patch, MagicMock

from llm.gemini_flash_2_5 import call_gemini
from llm.llm_orchestrator import generate_response


MOCK_RESPONSE = "Test response"


# -------- Fixtures -------- #

@pytest.fixture
def mock_gemini_response():
    mock = MagicMock()
    mock.text = MOCK_RESPONSE
    return mock


# -------- Gemini Tests -------- #

class TestGemini:

    @patch("llm.gemini_flash_2_5._get_client")
    def test_success(self, mock_get_client, mock_gemini_response):
        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = mock_gemini_response
        mock_get_client.return_value = mock_client

        result = call_gemini("Hello")
        assert result == MOCK_RESPONSE

    @patch("llm.gemini_flash_2_5._get_client")
    def test_missing_client(self, mock_get_client):
        mock_get_client.side_effect = EnvironmentError("Gemini API key not set.")
        with pytest.raises(EnvironmentError, match="Gemini API key not set"):
            call_gemini("Hello")

    @patch("llm.gemini_flash_2_5._get_client")
    def test_api_error(self, mock_get_client):
        mock_client = MagicMock()
        mock_client.models.generate_content.side_effect = Exception("API error")
        mock_get_client.return_value = mock_client

        with pytest.raises(Exception, match="API error"):
            call_gemini("Hello")

    @patch("llm.gemini_flash_2_5._get_client")
    def test_strips_whitespace(self, mock_get_client, mock_gemini_response):
        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = mock_gemini_response
        mock_get_client.return_value = mock_client

        mock_gemini_response.text = "  spaced response  "
        result = call_gemini("Hello")
        assert result == "spaced response"


# -------- Orchestrator Tests -------- #

class TestGenerateResponse:

    @patch("llm.llm_orchestrator.call_gemini", return_value=MOCK_RESPONSE)
    @patch("llm.llm_orchestrator.gemini_client", MagicMock())
    def test_returns_gemini(self, *_):
        assert generate_response("Hello") == MOCK_RESPONSE

    @patch("llm.llm_orchestrator.gemini_client", None)
    def test_missing_gemini_key(self):
        with pytest.raises(EnvironmentError, match="Gemini API key not set"):
            generate_response("Hello")

    @patch("llm.llm_orchestrator.call_gemini", return_value="  ")
    @patch("llm.llm_orchestrator.gemini_client", MagicMock())
    def test_empty_response_raises(self, *_):
        with pytest.raises(RuntimeError, match="Gemini returned an empty response"):
            generate_response("Hello")

    @patch("llm.llm_orchestrator.call_gemini")
    @patch("llm.llm_orchestrator.gemini_client", MagicMock())
    def test_gemini_error_bubbles(self, mock_call_gemini, *_):
        mock_call_gemini.side_effect = Exception("API error")
        with pytest.raises(Exception, match="API error"):
            generate_response("Hello")
