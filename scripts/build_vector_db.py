import chromadb
from pathlib import Path
from sentence_transformers import SentenceTransformer
from core.ocr import extract_text_from_document

DB_PATH = Path("data/chroma_db")
# This is the directory where your sample documents are organized into
# subfolders by type.
SAMPLE_DOCS_PATH = Path("data")
COLLECTION_NAME = "document_types"

print("Initializing database client and embedding model...")

# Initialize the ChromaDB client.
# `PersistentClient` ensures that the database is saved to disk at the
# specified path.
# This means the indexed data will persist even after the script finishes
# running.
client = chromadb.PersistentClient(path=str(DB_PATH))

# We are using the 'all-MiniLM-L12-v2' model from the SentenceTransformers
# library.
# This model is chosen for several key reasons:
# 1.  **Performance:** It is highly optimized to be small and fast, making it
#     ideal for local development and production use without requiring massive
#     computational resources.
# 2.  **Effectiveness:** Despite its size, it is very effective at creating
#     high-quality
#     semantic embeddings for sentences and paragraphs. This means it's great at
#     understanding the "meaning" of text.
# 3.  **General Purpose:** It's trained on a wide variety of text data, making
#     it suitable for general-purpose tasks like classifying the content of
#     diverse documents.
#
# --- Alternative Models ---
# - For Higher Quality (at the cost of speed): 'all-mpnet-base-v2'
#   This is a larger, more powerful model that often yields more accurate results.
#   Use this if embedding quality is more critical than processing speed.
# - A Middle Ground: 'all-MiniLM-L12-v2'
#   This model has 12 layers instead of 6 (like the default). It offers better
#   accuracy than L6 while still being faster than larger models like mpnet.
# - For Multilingual Documents: 'paraphrase-multilingual-MiniLM-L12-v2'
#   If your documents contain multiple languages, this model is specifically trained
#   to handle them effectively.
embedding_model = SentenceTransformer('all-MiniLM-L12-v2')

# Get or create a collection in ChromaDB. A collection is similar to a table in
# a traditional database and will store our document embeddings.
collection = client.get_or_create_collection(
    name=COLLECTION_NAME,
    # Cosine similarity is the best choice for comparing text embeddings because
    # it measures the angle between two vectors, not their magnitude. This means
    # it determines if two pieces of text are pointing in the same "semantic direction,"
    # regardless of their length.
    # For example, "Total amount due" and "Final invoice total" will have very
    # similar cosine scores even though their lengths differ.
    metadata={"hnsw:space": "cosine"}
)

# --- 2. Processing and Indexing Documents ---

print("Starting document processing and indexing...")
doc_id_counter = 0

for type_path in SAMPLE_DOCS_PATH.iterdir():
    # We only process subdirectories (e.g., 'invoices', 'receipts').
    if type_path.is_dir():
        # The directory name itself is our document type label.
        doc_type = type_path.name
        print(f"\nProcessing document type: '{doc_type}'")

        for doc_path in type_path.iterdir():
            print(f"\t- Processing file: {doc_path.name}")

            # Read the entire file into memory as bytes.
            with open(doc_path, "rb") as f:
                file_bytes = f.read()

            # Use our robust, updated OCR function to extract text.
            # This function handles both PDFs and images, returning clean,
            # structured text.
            text = extract_text_from_document(file_bytes, doc_path.name)

            # Only proceed if the OCR process successfully extracted text.
            if text.strip():
                # An embedding (or vector) is a list of numbers that represents
                # the semantic meaning of the text. The embedding_model converts
                # the extracted text string into this numerical format. Once in
                # this format, we can perform mathematical comparisons
                # (like cosine similarity) to find texts with similar meanings.
                embedding = embedding_model.encode(text).tolist()

                # The `collection.add()` method takes the generated embedding
                # and saves it to the persistent database along with its
                # associated metadata and a unique ID.
                collection.add(
                    embeddings=[embedding],
                    # Metadata allows us to store extra information with each
                    # vector.
                    # Here, we store the document's type, which is crucial for
                    # our classification task.
                    metadatas=[{"document_type": doc_type}],
                    # Each item in the database needs a unique ID.
                    ids=[f"id_{doc_id_counter}"]
                )
                doc_id_counter += 1
            else:
                print(f"\t- WARNING: No text extracted from {doc_path.name}. Skipping.")

print(f"\nVector database build complete. Total documents indexed: {doc_id_counter}")
