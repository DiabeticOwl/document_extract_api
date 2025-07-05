import pytest

from unittest.mock import MagicMock
from core.vector_db import VectorDBClient, get_vector_db_client

# Sample text to be used across tests
SAMPLE_TEXT = "This is a test document about an invoice."


@pytest.fixture
def mock_db_client(mocker):
    """
    A pytest fixture that creates a mocked VectorDBClient instance.

    This is the cornerstone of our unit tests. It allows us to test the logic
    of the VectorDBClient class in complete isolation, without the slow and
    unreliable overhead of loading real AI models or connecting to a real database.

    Args:
        mocker: The mocker fixture provided by the pytest-mock plugin.

    Returns:
        A fully mocked instance of the VectorDBClient.
    """
    # We use mocker.patch to replace the `_load` method with a function that
    # does nothing. This stops the client from trying to load models or connect
    # to ChromaDB.
    mocker.patch('core.vector_db.VectorDBClient._load', return_value=None)

    client = VectorDBClient()

    # Manually attach mock objects for the model and collection
    client.embedding_model = MagicMock()
    client.embedding_model.encode.return_value = [0.1, 0.2, 0.3]

    client.collection = MagicMock()

    return client


def test_find_document_type_success(mock_db_client):
    """
    Tests the happy path where a document type is successfully found.

    Purpose:
        Verify that the function correctly processes a successful query result
        from the vector database and returns the expected document type
        and a calculated confidence score.

    Mocks:
        - `embedding_model.encode`: To simulate generating an embedding vector.
        - `collection.query`: To simulate a successful response from ChromaDB,
          returning a match with metadata and a distance score.
    """
    mock_db_client.collection.query.return_value = {
        'ids': [['id1']],
        'distances': [[0.15]], # A low distance means high similarity.
        'metadatas': [[{'document_type': 'invoice'}]]
    }

    result = mock_db_client.find_document_type(SAMPLE_TEXT)

    assert result['document_type'] == 'invoice'
    # Confidence should be 1 - distance = 1 - 0.15 = 0.85
    assert result['confidence'] == 0.85


def test_find_document_type_no_match(mock_db_client):
    """
    Tests the scenario where the database returns no matching documents.

    Purpose:
        Ensure the function handles cases where the vector search yields
        no results, returning 'Unknown' with zero confidence.

    Mocks:
        - `collection.query`: To simulate an empty response from ChromaDB.
    """
    # Configure the mock query to return an empty list for the 'ids',
    # simulating a scenario where no similar documents were found.
    mock_db_client.collection.query.return_value = {
        'ids': [[]],
        'distances': [[]],
        'metadatas': [[]]
    }

    result = mock_db_client.find_document_type(SAMPLE_TEXT)

    assert result['document_type'] == 'Unknown'
    assert result['confidence'] == 0.0


def test_find_document_type_metadata_key_missing(mock_db_client):
    """
    Tests the edge case where a document is found, but its metadata is
    missing the required 'document_type' key.

    Purpose: To ensure the code is robust and doesn't crash with a KeyError
             if the database contains malformed metadata.
    """
    mock_db_client.collection.query.return_value = {
        'ids': [['id1']],
        'distances': [[0.2]],
        'metadatas': [[{'other_key': 'some_value'}]]
    }

    # A robust implementation of find_document_type should not raise a KeyError here.
    # It should handle the missing key gracefully.
    result = mock_db_client.find_document_type(SAMPLE_TEXT)

    # Assert that it defaults to 'Unknown' when the key is not found.
    assert result['document_type'] == 'Unknown'
    # It still has a confidence score because a document was found.
    assert result['confidence'] == 0.80


def test_find_document_type_db_not_available(mocker):
    """
    Tests the scenario where the database collection failed to initialize.

    Purpose:
        Verify that the function fails gracefully if the global 'collection'
        object is None, which would happen if the initial connection to
        ChromaDB failed on API startup.

    Mocks:
        - `core.vector_db.collection`: Patched to be None.
    """
    # We don't use the fixture here because we need to modify the instance
    # before the `_load` method is even considered.
    mocker.patch('core.vector_db.VectorDBClient._load', return_value=None)
    client = VectorDBClient()
    # Manually simulates a failed initialization.
    client.collection = None

    result = client.find_document_type(SAMPLE_TEXT)

    assert "error" in result
    assert result["error"] == "Vector database collection is not available."



def test_find_document_type_query_exception(mock_db_client):
    """
    Tests the function's error handling when the database query itself fails.

    Purpose:
        Ensure that any unexpected exception raised during the `collection.query`
        call is caught and handled, returning a standardized error message.

    Mocks:
        - `collection.query`: Patched to raise a generic Exception.
    """
    # Configures the mock `query` method to raise an Exception when called.
    # `side_effect` is the attribute used to make a mock raise an exception.
    mock_db_client.collection.query.side_effect = Exception("Database connection lost.")

    result = mock_db_client.find_document_type(SAMPLE_TEXT)

    assert "error" in result
    assert result["error"] == "Failed to query the vector database."


def test_get_vector_db_client_is_singleton():
    """
    Tests that the dependency factory function `get_vector_db_client`
    always returns the exact same instance of the client.

    Purpose: To verify that our use of `@lru_cache` is working correctly,
             which is critical for the performance of the running API.
    """
    # Clears the cache before the test to ensure a clean state
    get_vector_db_client.cache_clear()

    # Calls the factory function twice.
    client1 = get_vector_db_client()
    client2 = get_vector_db_client()

    # `is` checks if two variables point to the exact same object in memory.
    assert client1 is client2
