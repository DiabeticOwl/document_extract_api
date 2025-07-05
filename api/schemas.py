from pydantic import BaseModel, Field
from typing import Dict, Any, Optional


class ConfidenceValue(BaseModel):
    # Optional to gracefully handle cases where the Language Model cannot find
    # a specific piece of information in the document.
    value: Optional[Any] = Field(
        ...,
        example="INV-12345",
        description="The extracted value for the entity. Can be null if not found."
    )
    confidence: float = Field(
        ...,
        ge=0,
        le=1,
        example=0.99,
        description="The LLM's confidence score for this specific extraction."
    )


class ExtractionResponse(BaseModel):
    document_type: str = Field(
        ...,
        example="Invoice",
        description="The classified type of the document."
    )
    confidence: float = Field(
        ...,
        ge=0,
        le=1,
        example=0.92,
        description="The confidence score of the classification (0.0 to 1.0)."
    )
    entities: Dict[str, Optional[ConfidenceValue]] = Field(
        ...,
        example={"invoice_number": {"value": "INV-12345", "confidence": 0.99}},
        description="Key-value pairs of extracted data, each with a value and a confidence score."
    )
    processing_time: str = Field(
        ...,
        example="1.25s",
        description="Total time taken to process the request."
    )

