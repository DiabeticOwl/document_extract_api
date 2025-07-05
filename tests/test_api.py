import io
import pytest

from api.main import app
from core.vector_db import get_vector_db_client, VectorDBClient
from fastapi.testclient import TestClient
from pathlib import Path
from unittest.mock import MagicMock

# This file contains integration tests for the FastAPI application.
# Integration tests are designed to check how different parts of the system
# (like the API endpoints and the core logic modules) work together.

# The TestClient is a special object from FastAPI that allows us to send
# HTTP requests to our application in a testing environment without needing
# to run a live server.

client = TestClient(app)
FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture
def mock_db_client_override():
    """
    A pytest fixture to override the VectorDBClient dependency.

    This is the recommended way to handle dependencies in tests. It creates a
    mock client, injects it into the FastAPI app for the duration of a single
    test, and then cleans up by clearing the override afterwards.
    """
    mock_client = MagicMock(spec=VectorDBClient)

    # Apply the dependency override to the app.
    app.dependency_overrides[get_vector_db_client] = lambda: mock_client

    # Yield the mock to the test function so it can be configured.
    # The code in the test function will run at this point.
    yield mock_client

    # Clear the dependency override after the test is complete.
    app.dependency_overrides.clear()


def test_extract_entities_api_full_success(mocker, mock_db_client_override):
    """
    Tests the "happy path" for the full API endpoint.

    This test uses the fixture to inject a mock DB client and `mocker` to patch
    the other functions, verifying that the endpoint correctly orchestrates a
    successful response.
    """
    # Configure the behavior of all our mocks for this test.
    mock_db_client_override.find_document_type.return_value = {"document_type": "invoice", "confidence": 0.95}
    mocker.patch('api.endpoints.extract_text_from_document', return_value="Sample OCR text")
    mocker.patch('api.endpoints.extract_entities_with_llm', return_value={"invoice_number": "API-TEST-123"})

    # Send a request to the API endpoint.
    file_path = FIXTURES_DIR / "invoice-template.pdf"
    with open(file_path, "rb") as f:
        files = {'file': (file_path.name, f, 'application/pdf')}
        response = client.post("/extract_entities/", files=files)

    # Check the response and that our mocks were called correctly.
    assert response.status_code == 200
    data = response.json()
    assert data['document_type'] == 'invoice'
    assert data['entities']['invoice_number'] == 'API-TEST-123'
    mock_db_client_override.find_document_type.assert_called_once_with("Sample OCR text")


def test_extract_entities_api_unsupported_file_type():
    """
    Tests the API's input validation. This test doesn't need mocks as it
    should fail before any core logic is called.
    """
    dummy_file = io.BytesIO(b"this is a text file")
    files = {'file': ('test.txt', dummy_file, 'text/plain')}
    response = client.post("/extract_entities/", files=files)
    assert response.status_code == 400
    assert "Invalid file type" in response.json()['detail']


def test_extract_entities_api_ocr_failure(mocker):
    """
    Tests how the API handles a document from which no text can be extracted.
    """
    # Mock the OCR function to return an empty string
    mocker.patch('api.endpoints.extract_text_from_document', return_value="  ")

    file_path = FIXTURES_DIR / "blank.pdf" # Assumes you have a blank PDF
    with open(file_path, "rb") as f:
        files = {'file': (file_path.name, f, 'application/pdf')}
        response = client.post("/extract_entities/", files=files)

    assert response.status_code == 422
    assert response.json() == {'detail': 'Could not extract any text from the document.'}


def test_extract_entities_api_db_failure(mocker, mock_db_client_override):
    """
    Tests the API's response when the database service returns an error.
    """
    #Configure the mock DB client to return an error dictionary.
    mock_db_client_override.find_document_type.return_value = {"error": "Database is offline."}
    mocker.patch('api.endpoints.extract_text_from_document', return_value="Sample OCR text")
    mocker.patch('api.endpoints.extract_entities_with_llm', return_value={})

    file_path = FIXTURES_DIR / "invoice-template.pdf"
    with open(file_path, "rb") as f:
        files = {'file': (file_path.name, f, 'application/pdf')}
        response = client.post("/extract_entities/", files=files)

    assert response.status_code == 500
    assert response.json() == {'detail': 'Database is offline.'}


def test_extract_entities_api_llm_failure(mocker, mock_db_client_override):
    """
    Tests how the API handles a failure from the LLM service.

    Purpose: To ensure that if the final LLM extraction step fails, the API
             returns a 500 Internal Server Error with a relevant message,
             rather than crashing.
    """
    # Configure all mocks for a successful run up until the LLM step.
    mock_db_client_override.find_document_type.return_value = {
        "document_type": "invoice",
        "confidence": 0.95
    }
    mocker.patch(
        'api.endpoints.extract_text_from_document',
        return_value="Sample OCR text"
    )
    # Mock the LLM function to return an error dictionary, simulating a failure.
    mocker.patch(
        'api.endpoints.extract_entities_with_llm',
        return_value={"error": "LLM service is down."}
    )

    # Send a request to the endpoint.
    file_path = FIXTURES_DIR / "invoice-template.pdf"
    with open(file_path, "rb") as f:
        files = {'file': (file_path.name, f, 'application/pdf')}
        response = client.post("/extract_entities/", files=files)

    assert response.status_code == 500
    assert response.json() == {'detail': 'LLM service is down.'}
