# Intelligent Document Understanding API

## Overview

The **Intelligent Document Understanding API** is designed to extract structured information from unstructured documents. This API accepts document uploads (images or PDFs) and processes them to return structured data in a standardized JSON format. The main components of the API include:

- **OCR Implementation**: Extracting text from documents.
- **Vector Database**: Storing and retrieving document embeddings.
- **API Development**: Built using FastAPI.
- **Document Classification**: Identifying document types.
- **Entity Extraction**: Using Large Language Models (LLMs) to extract structured data fields.

## Project Structure
```
.
└── document_extract_api/
    ├── core/
    │   ├── __init__.py
    │   ├── ocr.py
    │   └── vector_db.py
    ├── data/
    ├── scripts/
    │   └── build_vector_db.py
    ├── tests/
    │   ├── fixtures/
    │   ├── test_ocr.py
    │   └── test_vector_db.py
    ├── README.md
    └── requirements.txt
```

## Setup & Usage

### 1. Environment setup
Utilize our `requirements.txt` file to install the needed packages within your environment.

### 2. Prepare the sample documents
In order to train your model you will need to create a [vector database](https://en.wikipedia.org/wiki/Vector_database), which is a specialized type of database designed to find items based on their semantic meaning rather than just exact matches.

1. Create a directory called `data/sample_docs`.
2. Organize your documents into subdirectories named by their type. For example:
```
data/sample_docs/
├── invoices/
│   ├── inv1.pdf
│   └── inv2.png
├── receipts/
│   └── rec1.jpg
```

A dataset used while building this project comes from [Shahebaz Mohammad's dataset in Kaggle](https://www.kaggle.com/datasets/shaz13/real-world-documents-collections).

### Handling Low-Quality OCR with Data Augmentation
A significant challenge in real-world document processing is handling low-quality inputs, such as skewed scans, poor lighting, or noisy artifacts. To make our system more robust against these issues, we have implemented an online data augmentation strategy during the vector database creation process.

The Strategy: Random Preprocessing
Instead of creating multiple static versions of each image, we apply one of several advanced preprocessing transformations randomly to each document before it is processed by the OCR engine. This means that on any given run of the build_vector_db.py script, the OCR engine is fed a slightly different, augmented version of the dataset.

This approach makes the resulting text embeddings in our vector database more robust. The database learns to associate not just the "perfect" version of a document with its type, but also versions that have been cleaned and optimized in different ways. This improves the system's ability to classify real-world, imperfect documents.

Implemented Transformations
Noise Reduction (Median Blur): This filter is applied to remove "salt-and-pepper" noise common in scanned documents. It smooths out random speckles without significantly blurring the sharp edges of the text, leading to cleaner input for the OCR engine.

Adaptive Thresholding: This technique converts the image to pure black and white by calculating an optimal brightness threshold for different regions of the image. It is highly effective at handling documents with shadows or uneven lighting, creating a high-contrast image that is ideal for OCR.

Deskewing: The image is algorithmically rotated to ensure the lines of text are perfectly horizontal, which can dramatically improve OCR accuracy on tilted scans.

None (Original Image): We also include the option to process the original, unaltered image. This ensures that the "perfect" version of the document is always represented in the database.

## 3. Build your Vector Database
The build_vector_db.py script will process all your sample documents and create a searchable vector database from them.

Run the script from your terminal:
```bash
python build_vector_db.py
```
The script is optimized to use multiple CPU cores for the OCR phase to process large datasets quickly. It will:

1. Scan the data/sample_docs/ directory for all supported files.
2. Use a pool of worker processes to perform OCR in parallel.
3. Use a highly efficient Sentence Transformer model to generate embeddings for the extracted text.
4. Create and save a persistent ChromaDB database in the data/chroma_db/ directory.
5. After the script finishes, your vector database is ready for the next stage of the project: building the API for document classification.
