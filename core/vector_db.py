import chromadb
from pathlib import Path
from sentence_transformers import SentenceTransformer

DB_PATH = Path("data/chroma_db")
COLLECTION_NAME = "document_types"
MODEL_NAME = 'all-MiniLM-L6-v2'

try:
    print("Initializing vector database client and embedding model for querying...")
    # Initialize the embedding model.
    embedding_model = SentenceTransformer(MODEL_NAME)
    # Connect to the persistent ChromaDB client.
    client = chromadb.PersistentClient(path=str(DB_PATH))
    # Get the existing collection.
    collection = client.get_collection(name=COLLECTION_NAME)
    print("Vector database and model initialized successfully.")
except Exception as e:
    print(f"FATAL: Could not initialize vector database. Error: {e}")
    print("Please ensure you have run the 'build_vector_db.py' script successfully.")
    collection = None


def find_document_type(text: str) -> dict:
    """
    Performs a semantic search to find the most likely document type.

    Args:
        text (str): The extracted text from the uploaded document.

    Returns:
        dict: A dictionary containing the 'document_type' and a 'confidence'
              score, or an error message if the query fails.
    """
    if collection is None:
        return {"error": "Vector database collection is not available."}

    try:
        # 1. Generate an embedding for the input text.
        query_embedding = embedding_model.encode(text)

        # 2. Query the collection to find the single most similar document.
        # The result includes the metadata and the distance (similarity score).
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=1
        )

        # 3. Process the results.
        if not results or not results['ids'][0]:
            return {"document_type": "Unknown", "confidence": 0.0}

        # The distance is cosine distance. A smaller distance means more similar.
        # We can convert this to a confidence score (0.0 to 1.0).
        distance = results['distances'][0][0]
        confidence = 1 - distance

        # Extract the document type from the metadata of the closest match.
        metadata = results['metadatas'][0][0]
        # If the key is missing, it will default to 'Unknown' instead of raising KeyError.
        doc_type = metadata.get('document_type', 'Unknown')

        return {"document_type": doc_type, "confidence": round(confidence, 2)}

    except Exception as e:
        print(f"An error occurred during vector database query: {e}")
        return {"error": "Failed to query the vector database."}
