import chromadb

from functools import lru_cache
from pathlib import Path
from sentence_transformers import SentenceTransformer


class VectorDBClient:
    """
    A client class to handle all interactions with the ChromaDB vector database.

    This class encapsulates the logic for initializing the database connection,
    loading the embedding model, and querying for document types. This approach
    prevents the models from being loaded on simple module imports and provides
    a clean, reusable interface for the API.
    """
    def __init__(self):
        """
        Initializes the VectorDBClient.

        This method loads the embedding model and connects to the ChromaDB
        collection. It's designed to be called once when the API starts.
        """
        self.db_path = Path(__file__).parent.parent / "data/chroma_db"
        self.collection_name = "document_types"
        self.model_name = 'all-MiniLM-L6-v2'

        # These will be populated by the _load() method.
        self.embedding_model = None
        self.collection = None

        # Call the private load method to initialize resources.
        self._load()

    def _load(self):
        """
        Private method to load the model and connect to the database.
        """
        try:
            print("Initializing vector database client and embedding model...")
            self.embedding_model = SentenceTransformer(self.model_name)
            client = chromadb.PersistentClient(path=str(self.db_path))
            self.collection = client.get_collection(name=self.collection_name)
            print("Vector database and model initialized successfully.")
        except Exception as e:
            print(f"FATAL: Could not initialize vector database. Error: {e}")
            print("Please ensure you have run the 'build_vector_db.py' script successfully.")
            # If initialization fails, the collection will remain None.
            self.collection = None


    def find_document_type(self, text: str) -> dict:
        """
        Performs a semantic search to find the most likely document type.

        Args:
            text (str): The extracted text from the uploaded document.

        Returns:
            dict: A dictionary containing the 'document_type' and a 'confidence'
                score, or an error message if the query fails.
        """
        if self.collection is None:
            return {"error": "Vector database collection is not available."}

        try:
            # 1. Generate an embedding for the input text.
            query_embedding = self.embedding_model.encode(text)

            # 2. Query the collection to find the single most similar document.
            # The result includes the metadata and the distance (similarity score).
            results = self.collection.query(
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


# The @lru_cache decorator is a simple and powerful way to turn a function
# into a cached singleton factory. The first time this function is called, it
# will create and return a new VectorDBClient instance. On all subsequent calls
# within the application's lifecycle, it will instantly return the *exact same*
# instance without re-running the function.
@lru_cache
def get_vector_db_client() -> VectorDBClient:
    """
    Dependency factory for the VectorDBClient.
    Using lru_cache ensures that only one instance of the client is created.
    """
    return VectorDBClient()
