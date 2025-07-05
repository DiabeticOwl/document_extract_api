from core.vector_db import find_document_type

# Sample text to be used across tests
SAMPLE_TEXT = "This is a test document about an invoice."


def test_find_document_type_success(mocker):
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
    # 1. Setup Mocks
    # Mock the embedding model's encode method
    mocker.patch('core.vector_db.embedding_model.encode', return_value=[0.1, 0.2, 0.3])

    # Mock the collection object's query method
    mock_collection = mocker.patch('core.vector_db.collection')
    mock_collection.query.return_value = {
        'ids': [['id1']],
        'distances': [[0.15]], # Cosine distance of 0.15
        'metadatas': [[{'document_type': 'invoice'}]]
    }

    # 2. Call the function
    result = find_document_type(SAMPLE_TEXT)
    print(result)

    # 3. Assert the results
    assert result['document_type'] == 'invoice'
    # Confidence should be 1 - distance = 1 - 0.15 = 0.85
    assert result['confidence'] == 0.85


def test_find_document_type_no_match(mocker):
    """
    Tests the scenario where the database returns no matching documents.

    Purpose:
        Ensure the function handles cases where the vector search yields
        no results, returning 'Unknown' with zero confidence.

    Mocks:
        - `collection.query`: To simulate an empty response from ChromaDB.
    """
    mocker.patch('core.vector_db.embedding_model.encode', return_value=[0.1, 0.2, 0.3])
    mock_collection = mocker.patch('core.vector_db.collection')
    # Simulate a query that returns no valid IDs
    mock_collection.query.return_value = {'ids': [[]], 'distances': [[]], 'metadatas': [[]]}

    result = find_document_type(SAMPLE_TEXT)

    assert result['document_type'] == 'Unknown'
    assert result['confidence'] == 0.0


def test_find_document_type_metadata_key_missing(mocker):
    """
    Tests the scenario where a document is found, but its metadata
    is missing the 'document_type' key. This is a crucial edge case.
    """
    mocker.patch('core.vector_db.embedding_model.encode', return_value=[0.1, 0.2, 0.3])
    mock_collection = mocker.patch('core.vector_db.collection')
    # Simulate a result where the metadata dictionary is empty or missing the key.
    mock_collection.query.return_value = {
        'ids': [['id1']],
        'distances': [[0.2]],
        'metadatas': [[{'other_key': 'some_value'}]] # Note: 'document_type' key is missing
    }

    # A robust implementation of find_document_type should not raise a KeyError here.
    # It should handle the missing key gracefully.
    result = find_document_type(SAMPLE_TEXT)

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
    mocker.patch('core.vector_db.collection', None)

    result = find_document_type(SAMPLE_TEXT)

    assert "error" in result
    assert result["error"] == "Vector database collection is not available."


def test_find_document_type_query_exception(mocker):
    """
    Tests the function's error handling when the database query itself fails.

    Purpose:
        Ensure that any unexpected exception raised during the `collection.query`
        call is caught and handled, returning a standardized error message.

    Mocks:
        - `collection.query`: Patched to raise a generic Exception.
    """
    mocker.patch('core.vector_db.embedding_model.encode', return_value=[0.1, 0.2, 0.3])
    mock_collection = mocker.patch('core.vector_db.collection')
    mock_collection.query.side_effect = Exception("Database connection lost")

    result = find_document_type(SAMPLE_TEXT)

    assert "error" in result
    assert result["error"] == "Failed to query the vector database."
