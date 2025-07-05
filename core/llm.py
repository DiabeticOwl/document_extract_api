import requests
import json

# --- Configuration ---
# The URL of the local Ollama API. This assumes Ollama is running on the same
# machine.
OLLAMA_API_URL = "http://localhost:11434/api/generate"
LLM_MODEL = "phi3:mini"

PROMPT_TEMPLATE = """
Given the following text extracted from a document of type '{document_type}', extract the following fields: {field_list}.
Return your response as a valid JSON object.
For each field, provide a nested JSON object with two keys: "value" which is the extracted information (or null if not found), and "confidence" which is your estimated confidence score from 0.0 to 1.0 that the value is correct based on the text.
Provide no additional text, commentary, or explanation outside of the JSON object.

Document Text:
---
{document_text}
---
"""

# This dictionary defines the specific fields we want to extract for each
# document type.
DOCUMENT_SCHEMAS = {
    "advertisement": ["product_or_service_name", "brand_name", "slogan_or_headline", "call_to_action", "contact_info", "offer_or_discount"],
    "budget": ["budget_title", "time_period", "total_income", "total_expenses", "net_balance"],
    "email": ["sender_name", "sender_email", "recipient_name", "recipient_email", "subject", "date_sent", "summary"],
    "file_folder": ["folder_label"],
    "form": ["form_title", "full_name", "date_of_birth", "address"],
    "handwritten": ["title", "author", "date", "summary_of_content"],
    "invoice": ["invoice_number", "vendor_name", "customer_name", "date", "total_amount"],
    "letter": ["sender_name", "sender_address", "recipient_name", "recipient_address", "date", "salutation", "closing"],
    "memo": ["to", "from", "date", "subject", "cc"],
    "news_article": ["headline", "author", "publication_name", "publication_date", "summary"],
    "presentation": ["presentation_title", "presenter_name", "event_or_conference_name", "date", "key_topics"],
    "questionnaire": ["questionnaire_title", "issuing_organization", "list_of_questions"],
    "receipt": ["store_name", "date", "total_amount", "items"],
    "resume": ["candidate_name", "contact_email", "contact_phone", "summary_or_objective", "skills"],
    "scientific_publication": ["title", "authors", "journal_name", "publication_date", "doi", "abstract"],
    "scientific_report": ["report_title", "authors", "reporting_organization", "report_date", "report_number", "summary"],
    "specification": ["document_title", "document_id_or_version", "authoring_organization", "effective_date", "product_or_system_name"]
}


def extract_entities_with_llm(document_text: str, document_type: str) -> dict:
    """
    Extracts structured entities from document text using a local LLM.

    Args:
        document_text (str): The raw text extracted from the document via OCR.
        document_type (str): The classified type of the document (e.g., 'invoice').

    Returns:
        dict: A dictionary containing the extracted key-value pairs, or an error dictionary.
    """
    if document_type.lower() not in DOCUMENT_SCHEMAS:
        # If the document type is unknown or unsupported, we can't extract entities.
        return {}

    field_list = DOCUMENT_SCHEMAS[document_type.lower()]

    # Format the prompt with the specific details of this document.
    prompt = PROMPT_TEMPLATE.format(
        document_type=document_type,
        field_list=", ".join(field_list),
        document_text=document_text
    )

    # This is the payload that will be sent to the Ollama API.
    payload = {
        "model": LLM_MODEL,
        "prompt": prompt,
        "stream": False,
        "format": "json"
    }

    try:
        # Make the API call to the local LLM.
        response = requests.post(OLLAMA_API_URL, json=payload, timeout=180)
        response.raise_for_status() # Raise an exception for bad status codes (4xx or 5xx)

        # The response from Ollama with format=json is already a JSON object.
        # The actual JSON string is in the 'response' key.
        response_text = response.json().get('response', '{}')
        return json.loads(response_text)

    except requests.exceptions.RequestException as e:
        print(f"Error calling LLM API: {e}")
        return {"error": "Failed to connect to the local LLM service."}
    except json.JSONDecodeError:
        print(f"Error: LLM returned a non-JSON response: {response_text}")
        return {"error": "LLM returned a malformed response."}
