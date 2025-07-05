import requests
from core.llm import extract_entities_with_llm

# Sample text to be used in tests
SAMPLE_INVOICE_TEXT = "Invoice #INV-007 from ACME Corp. to John Doe for $1500.50 due on 2025-12-31."


def test_extract_entities_llm_success(mocker):
    """
    Tests the happy path for LLM entity extraction.
    Verifies that the prompt is correctly formatted and the JSON response is parsed.
    """
    # 1. Setup the mock for requests.post
    mock_response = mocker.Mock()
    # Ollama's format=json returns a JSON object where the 'response' key contains a stringified JSON.
    mock_response.json.return_value = {
        'response': '{"invoice_number": "INV-007", "vendor_name": "ACME Corp.", "total_amount": "$1500.50"}'
    }
    mock_response.raise_for_status.return_value = None
    mocker.patch('requests.post', return_value=mock_response)

    # 2. Call the function
    entities = extract_entities_with_llm(SAMPLE_INVOICE_TEXT, "invoice")

    # 3. Assert the results
    assert entities == {"invoice_number": "INV-007", "vendor_name": "ACME Corp.", "total_amount": "$1500.50"}

    # Assert that the prompt sent to the LLM was correctly formatted
    sent_payload = requests.post.call_args.kwargs['json']
    assert "document of type 'invoice'" in sent_payload['prompt']
    assert "invoice_number" in sent_payload['prompt']
    assert SAMPLE_INVOICE_TEXT in sent_payload['prompt']


def test_extract_entities_unsupported_type():
    """
    Tests that the function returns an empty dict for an unsupported document type.
    """
    entities = extract_entities_with_llm(SAMPLE_INVOICE_TEXT, "unsupported_type")
    assert entities == {}


def test_extract_entities_llm_connection_error(mocker):
    """
    Tests the failure case where the request to the Ollama API fails.
    """
    # Setup the mock to raise a connection error
    mocker.patch('requests.post', side_effect=requests.exceptions.RequestException("Connection failed"))

    entities = extract_entities_with_llm(SAMPLE_INVOICE_TEXT, "invoice")

    assert "error" in entities
    assert entities["error"] == "Failed to connect to the local LLM service."


def test_extract_entities_llm_bad_json_response(mocker):
    """
    Tests handling of a malformed (non-JSON) string in the LLM response.
    """
    mock_response = mocker.Mock()
    # Simulate the LLM returning plain text instead of a JSON string
    mock_response.json.return_value = {'response': 'This is not valid JSON.'}
    mock_response.raise_for_status.return_value = None
    mocker.patch('requests.post', return_value=mock_response)

    entities = extract_entities_with_llm(SAMPLE_INVOICE_TEXT, "invoice")

    assert "error" in entities
    assert entities["error"] == "LLM returned a malformed response."
