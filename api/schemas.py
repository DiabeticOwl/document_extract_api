from pydantic import BaseModel, Field
from typing import Dict, Any


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
    # The 'entities' field will be populated by the LLM in a future step.
    # For now, it can be an empty dictionary.
    entities: Dict[str, Any] = Field(
        ...,
        example={"invoice_number": "INV-12345", "total_amount": "$450.00"},
        description="Key-value pairs of extracted data."
    )
    processing_time: str = Field(
        ...,
        example="1.25s",
        description="Total time taken to process the request."
    )

