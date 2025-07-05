import time

from api.schemas import ExtractionResponse
from core.ocr import extract_text_from_document, SUPPORTED_MIME_TYPES
from core.llm import extract_entities_with_llm
from core.vector_db import find_document_type
from fastapi import APIRouter, UploadFile, File, HTTPException

# Create an APIRouter. This helps in organizing endpoints and can be included
# in the main FastAPI app instance.
router = APIRouter()

@router.post(
    "/extract_entities/",
    response_model=ExtractionResponse,
    tags=["Document Processing"],
    summary="Extract entities from a document"
)
async def extract_entities(file: UploadFile = File(..., description="The document file (PDF or image) to be processed.")):
    """
    This endpoint performs the full document processing pipeline:
    1.  Accepts a document upload.
    2.  Performs OCR to extract text.
    3.  Classifies the document type using a vector database.
    4.  Extracts structured entities using an LLM.
    5.  Returns a standardized JSON response.
    """
    start_time = time.time()

    # 1. Read file and validate its content type.
    if not file.content_type in SUPPORTED_MIME_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type. Supported types are: {', '.join(SUPPORTED_MIME_TYPES)}"
        )

    file_bytes = await file.read()

    # 2. Perform OCR to extract text from the document.
    extracted_text = extract_text_from_document(file_bytes, file.filename)
    if not extracted_text.strip():
        raise HTTPException(
            status_code=422,
            detail="Could not extract any text from the document."
        )

    # 3. Classify the document type using the vector database.
    classification_result = find_document_type(extracted_text)
    if "error" in classification_result:
        raise HTTPException(
            status_code=500,
            detail=classification_result["error"]
        )

    doc_type = classification_result['document_type']
    confidence = classification_result['confidence']

    # 4. Extract structured entities using the LLM.
    entities = extract_entities_with_llm(extracted_text, doc_type)
    if "error" in entities:
        raise HTTPException(status_code=500, detail=entities["error"])

    processing_time = f"{time.time() - start_time:.2f}s"

    return ExtractionResponse(
        document_type=doc_type,
        confidence=confidence,
        entities=entities,
        processing_time=processing_time
    )
