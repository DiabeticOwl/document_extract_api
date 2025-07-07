# Intelligent Document Understanding API

## Index

1. [Overview](#overview)
2. [Project Architecture](#project-architecture)
3. [Project Structure](#project-structure)
4. [Setup & Usage](#setup--usage)
    - [1. Environment setup](#1-environment-setup)
    - [2. Prepare the sample documents](#2-prepare-the-sample-documents)
    - [3. Build the Vector Database](#3-build-the-vector-database)
    - [4. Run the API Server](#4-run-the-api-server)
    - [5. API usage](#5-api-usage)
5. [Testing Procedures](#testing-procedures)
6. [Docker Implementation](#docker-implementation)
    - [1. Build the Docker Image](#1-build-the-docker-image)
    - [2. Run the Docker Container](#2-run-the-docker-container)
    - [3. Access the Application](#3-access-the-application)
    - [4. Stop the Container](#4-stop-the-container)
7. [Data Augmentation for OCR](#data-augmentation-for-ocr)
    - [Implemented Transformations](#implemented-transformations)
    - [Transformations example](#transformations-example)

## Overview

The **Intelligent Document Understanding API** is designed to extract structured information from unstructured documents. This API accepts document uploads (images or PDFs) and processes them to return structured data in a standardized JSON format. The main components of the API include:

- **OCR Implementation**: Extracting text from documents.
- **Vector Database**: Storing and retrieving document embeddings.
- **API Development**: Built using FastAPI.
- **Document Classification**: Identifying document types.
- **Entity Extraction**: Using Large Language Models (LLMs) to extract structured data fields.

## Project Architecture

| **Component** 	| **Technology** 	| **Purpose** 	|
|:---:	|:---:	|:---:	|
| **API Framework** 	| FastAPI 	| For building a robust, async API with automatic documentation. 	|
| **OCR Engine** 	| EasyOCR & PyMuPDF 	| For accurate text extraction from images and PDFs. 	|
| **Embedding Model** 	| Sentence-Transformers 	| To convert text into meaningful numerical vectors for search. 	|
| **Vector Database** 	| ChromaDB 	| To store and perform efficient similarity searches on embeddings. 	|
| **LLM Service** 	| Ollama & Phi-3 Mini 	| To serve a local LLM for fast and accurate entity extraction. 	|
| **Parallel Processing** 	| `multiprocessing` 	| To significantly speed up the initial data processing pipeline. 	|
| **Image Augmentation** 	| OpenCV & Pillow 	| To apply preprocessing transformations for a more robust dataset. 	|

## Project Structure
```
.
├── api/
│   ├── endpoints.py
│   ├── main.py
│   └── schemas.py
├── core/
│   ├── llm.py
│   ├── ocr.py
│   ├── utils.py
│   └── vector_db.py
├── data/
│   ├── sample_docs/
│   └── chroma_db/
├── frontend/
│   └── index.html
├── scripts/
│   ├── build_vector_db.py
│   └── manual_test.py
├── tests/
│   ├── fixtures/
│   ├── test_api.py
│   ├── test_llm.py
│   ├── test_ocr.py
│   └── test_vector_db.py
├── .gitignore
├── Dockerfile
├── README.md
├── requirements.txt
└── start.sh
```

## Setup & Usage

### 1. Environment setup
Utilize our `requirements.txt` file to install the needed packages within your environment.

```bash
pip install -r requirements.txt
```

Install and run [Ollama](https://ollama.com/) and pull the required model locally:
```bash
ollama pull phi3:mini
```

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

### 3. Build the Vector Database
This one-time, resumable script processes your sample documents and builds the `chroma_db` vector database.

```bash
python -m scripts.build_vector_db
```

### 4. Run the API Server
Once the database is built, run the FastAPI server.
```bash
uvicorn api.main:app --reload
```
Once the server is running, the API is ready at `http://127.0.0.1:8000`.

### 5. API usage
The endpoint: `POST /extract_entities/` accepts a multipart/form-data request with a file field and returns a JSON object with the extracted information.

- Example `curl` request:
    ```bash
    curl -X POST "http://127.0.0.1:8000/extract_entities/" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@/path/to/your/invoice.pdf"
    ```
- Success response (200 OK):
    ```json
    {
        "document_type": "invoice",
        "confidence": 0.92,
        "entities": {
            "invoice_number": {
                "value": "INV-12345",
                "confidence": 0.99
            },
            "total_amount": {
                "value": "$450.00",
                "confidence": 0.95
            }
        },
        "processing_time": "5.72s"
    }
    ```

- When running through your web browser you can consume the LLM in a similar but more user friendly way:

    ![API Response through Frontend](<API Response through Frontend.png>)


## Testing Procedures
The project includes a comprehensive test suite using `pytest` to ensure code quality, reliability, and maintainability. Our testing strategy is divided into two main categories: unit tests for individual components and integration tests for the complete API workflow. This suite makes extensive use of mocking via the `pytest-mock` library to isolate components from external dependencies.

To run the tests, execute the following command from the project's root directory to run all test across the entire project:
```bash
pytest
```

Or, this one for selecting a specific module:
```bash
pytest tests/test_ocr.py
```
## Docker implementation
Docker is used to containerize this API, packaging the application, its dependencies, and the Ollama LLM server into a single, portable image. This is the recommended way to test the application in a production-like environment.

The `start.sh` script is essential for the initialization of this service under a server, as it manages the startup of both the `ollama` server (in the background) and the `uvicorn` web server (in the foreground) within the container.

1. **Build the Docker Image**: This command reads the Dockerfile and executes each step to build a self-contained, portable image of the application named document-api. This process may take several minutes on the first run as it downloads the base images and installs all dependencies.
```bash
docker build -t document-api .
```

2. **Run the Docker Container**: This command starts the container from the image you just built. It includes several important flags:

    - `--rm`: Automatically removes the container when it stops, keeping your system clean.
    - `-p 8000:7860`: Maps port 8000 on your local machine to port 7860 inside the container, where the Uvicorn server is listening.
    - `-v "$(pwd)/data:/app/data"`: Mounts your project's local `data` folder (containing `chroma_db`) into the `/app/data` folder inside the container. This is highly efficient as it allows the container to use your actual local database without copying it.

    ```bash
    docker run --rm -p 8000:7860 -v "$(pwd)/data:/app/data" --name document-api-container document-api
    ```
3. **Access the Application**: Once the container is running, open your browser and navigate to `http://localhost:8000`. You should see the web interface and can begin testing.

4. **Stop the Container**: Press `CTRL+C` in the terminal where the container is running. This will gracefully stop the Uvicorn server and, because of the `--rm` flag, the container will be automatically removed.

## Data Augmentation for OCR
To handle low-quality real-world documents, the `build_vector_db.py` script employs an online data augmentation strategy. This is a critical step for building a system that is resilient to the imperfections of scanned or photographed documents. During the build process, multiple advanced preprocessing transformations is applied to each document before it is passed to the OCR engine.

This approach is more efficient than offline augmentation (pre-generating thousands of transformed images), as it saves significant disk space and time. More importantly, it makes the resulting vector database more robust. The system learns to associate a document's core semantic meaning with a variety of visual representations, not just a single "perfect" version. This greatly improves its ability to classify real-world documents affected by skewed scans, digital noise, and uneven lighting.

#### Implemented Transformations:
- **Noise Reduction (Median Blur)**: This filter is specifically chosen to remove "salt-and-pepper" noise, which is common in scanned documents. It works by replacing each pixel with the median value of its neighbors, effectively smoothing out random speckles while preserving the sharp edges of the text, which is crucial for OCR accuracy.

- **Adaptive Thresholding**: This advanced technique converts an image to pure black and white, which is ideal for OCR. Unlike simple thresholding, it calculates an optimal brightness cutoff for different regions of the image independently. This makes it highly effective at handling documents with shadows or uneven lighting, ensuring that faint text in one area and clear text in another are both processed correctly.

- **Deskewing**: Even a slight tilt in a scanned document can significantly degrade OCR performance. This function algorithmically detects the general orientation of the text block and rotates the image to ensure the text lines are perfectly horizontal, presenting the OCR engine with the ideal alignment it was trained on.

- **None (Original Image)**

#### Transformations example:
| **None (Original Image)** | **Noise Reduction (Median Blur)** 	| **Adaptive Thresholding** 	| **Deskewing** 	|
|---	|---	|---	|---	|
| ![None (Original Image)](data/sample_transformations/original.png) 	| ![Noise Reduction (Median Blur)](data/sample_transformations/processed_noise_reduction.png) 	| ![Adaptive Thresholding](data/sample_transformations/processed_adaptive_threshold.png) 	| ![Deskewing](data/sample_transformations/processed_deskewed.png) 	|

