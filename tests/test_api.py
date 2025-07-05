import io

from api.main import app
from fastapi.testclient import TestClient
from pathlib import Path

# Create a client instance for testing our FastAPI app
client = TestClient(app)
FIXTURES_DIR = Path(__file__).parent / "fixtures"


def test_extract_entities_api_full_success(mocker):
    """
    Tests the full API endpoint with mocks for the core logic.
    This is an integration test for the API layer.
    """
    # 1. Mock all the core service functions
    mocker.patch('api.endpoints.extract_text_from_document', return_value="Sample OCR text from invoice")
    mocker.patch('api.endpoints.find_document_type', return_value={"document_type": "invoice", "confidence": 0.95})
    mocker.patch('api.endpoints.extract_entities_with_llm', return_value={"invoice_number": "API-TEST-123", "total_amount": "$100"})

    # 2. Prepare the file for upload
    file_path = FIXTURES_DIR / "invoice-template.pdf" # Assumes you have a test invoice in your fixtures
    with open(file_path, "rb") as f:
        files = {'file': (file_path.name, f, 'application/pdf')}
        response = client.post("/extract_entities/", files=files)

    # 3. Assert the response
    assert response.status_code == 200
    data = response.json()
    assert data['document_type'] == 'invoice'
    assert data['confidence'] == 0.95
    assert data['entities']['invoice_number'] == 'API-TEST-123'
    assert 'processing_time' in data


def test_extract_entities_api_unsupported_file_type():
    """
    Tests the API's validation for incorrect file content types.
    """
    # Create a dummy text file in memory
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


def test_extract_entities_api_llm_failure(mocker):
    """
    Tests how the API handles a failure from the LLM service.
    """
    mocker.patch('api.endpoints.extract_text_from_document', return_value="Sample OCR text")
    mocker.patch('api.endpoints.find_document_type', return_value={"document_type": "invoice", "confidence": 0.95})
    # Mock the LLM function to return an error
    mocker.patch('api.endpoints.extract_entities_with_llm', return_value={"error": "LLM service is down."})

    file_path = FIXTURES_DIR / "invoice-template.pdf"
    with open(file_path, "rb") as f:
        files = {'file': (file_path.name, f, 'application/pdf')}
        response = client.post("/extract_entities/", files=files)

    assert response.status_code == 500
    assert response.json() == {'detail': 'LLM service is down.'}
