import chromadb
import concurrent.futures
import easyocr
import multiprocessing
import numpy as np
import time

from core.ocr import extract_text_from_document, SUPPORTED_FORMATS
from pathlib import Path
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

DB_PATH = Path("data/chroma_db")
SAMPLE_DOCS_PATH = Path("data/sample_docs")
COLLECTION_NAME = "document_types"
MODEL_NAME = 'all-MiniLM-L6-v2'

# MAX_WORKERS limits the number of parallel processes for both OCR and encoding.
# A lower number reduces memory (VRAM) and CPU load but is slower.
# A higher number is faster but requires more resources.
MAX_WORKERS = 4

worker_ocr_reader = None


def init_ocr_worker():
    """
    Initializer function for each worker process in the pool.
    This runs ONCE per worker, loading the EasyOCR model into that process's memory.
    """
    global worker_ocr_reader
    worker_ocr_reader = easyocr.Reader(['en'])


def ocr_worker(doc_path: Path):
    """
    A worker function for OCR. It reads its assigned file and uses the
    pre-initialized model via the centralized `extract_text_from_document`
    function.
    """
    global worker_ocr_reader
    try:
        with open(doc_path, "rb") as f:
            file_bytes = f.read()

        text = extract_text_from_document(file_bytes, doc_path.name, reader=worker_ocr_reader)

        if text.strip():
            doc_type = doc_path.parent.name
            return {"text": text, "metadata": {"document_type": doc_type}}
        return None
    except Exception as e:
        tqdm.write(f"  - ERROR processing {doc_path.name}: {e}")
        return None


def main():
    """
    Main function to orchestrate the building of the vector database.

    This script operates in three distinct phases for maximum efficiency:
    1. OCR Phase: Sequentially read all documents and extract their text.
    2. Parallel Encoding Phase: Use all available CPU cores to convert the
       extracted texts into vector embeddings simultaneously.
    3. Database Insertion Phase: Add all the generated embeddings to the
       database in a single, efficient batch operation.
    """
    start_time = time.time()

    # --- 1. OCR Phase: Extract Text from All Documents ---
    # In this phase, we iterate through every document one by one to perform OCR.
    # While the OCR for each file is sequential, it's generally fast enough,
    # and this approach simplifies the data preparation for the next, more
    # computationally intensive phase.
    print(f"--- Phase 1: Starting Parallel OCR (max_workers={MAX_WORKERS}) ---")

    if not SAMPLE_DOCS_PATH.is_dir():
        print(f"Error: The specified sample documents directory does not exist: {SAMPLE_DOCS_PATH}")
        return

    all_doc_paths = [
        p for p in SAMPLE_DOCS_PATH.rglob('*')
        if p.is_file()
           and p.parent.name != DB_PATH.name
           and p.suffix.lower() in SUPPORTED_FORMATS
    ]

    if not all_doc_paths:
        print("Error: No documents found in the sample directory. Exiting.")
        return

    print(f"Found {len(all_doc_paths)} documents to process.")

    texts_to_process = []
    with multiprocessing.Pool(processes=MAX_WORKERS, initializer=init_ocr_worker) as pool:
        results = list(tqdm(
            pool.imap_unordered(ocr_worker, all_doc_paths),
            total=len(all_doc_paths),
            desc="Phase 1: Parallel OCR"
        ))
        texts_to_process = [result for result in results if result]

    if not texts_to_process:
        print("Error: No text could be extracted from any documents. Exiting.")
        return

    print(f"\nOCR Phase complete. Found text in {len(texts_to_process)} documents.")

    print("\n--- Phase 2: Starting batched embedding generation ---")
    try:
        print("Initializing SentenceTransformer model...")
        embedding_model = SentenceTransformer(MODEL_NAME)

        sentences = [item["text"] for item in texts_to_process]

        # `encode` takes the list of sentences and distributes them among the worker
        # processes in the pool. It automatically handles batching and scheduling
        # to maximize CPU utilization.
        #
        # An embedding (or vector) is a list of numbers that represents
        # the semantic meaning of the text. The embedding_model converts
        # the extracted text string into this numerical format.
        print(f"Encoding {len(sentences)} documents in a single batch...")
        embeddings = embedding_model.encode(
            sentences,
            batch_size=32,
            show_progress_bar=True
        )

        print("Embedding generation complete.")

    except Exception as e:
        print(f"FATAL ERROR during embedding phase: {e}")
        return

    # --- Phase 3: Database Insertion ---
    # In the final phase, we connect to the database and add all the processed
    # data in a single, highly efficient batch operation.
    print("\n--- Phase 3: Initializing database and adding documents ---")
    try:
        client = chromadb.PersistentClient(path=str(DB_PATH))
        collection = client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"}
        )

        # Adding data in one large batch is significantly faster than adding each
        # document individually, as it minimizes database transaction overhead.
        print(f"Adding {len(texts_to_process)} embeddings to the ChromaDB collection...")
        collection.add(
            # Convert the numpy array to a standard list
            embeddings=embeddings.tolist(),
            metadatas=[item["metadata"] for item in texts_to_process],
            ids=[f"id_{i}" for i in range(len(texts_to_process))]
        )
    except Exception as e:
        print(f"FATAL ERROR during database insertion: {e}")
        return

    end_time = time.time()
    print("\n--------------------------------------------------")
    print(f"Vector database build complete.")
    print(f"Total documents successfully indexed: {len(texts_to_process)}")
    print(f"Total time taken: {end_time - start_time:.2f} seconds")
    print("--------------------------------------------------")


if __name__ == "__main__":
    # This is crucial for libraries that use CUDA (like PyTorch, which powers
    # EasyOCR and SentenceTransformers) to work correctly in parallel processes.
    # 'spawn' creates a fresh process, avoiding CUDA initialization conflicts
    # that occur with the default 'fork' method on Linux/macOS.
    # This line must be inside the `if __name__ == "__main__":` block.
    multiprocessing.set_start_method('spawn', force=True)

    main()
