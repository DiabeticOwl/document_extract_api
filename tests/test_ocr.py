import pytest
from pathlib import Path
from core.ocr import extract_text_from_document

# Using `Path(__file__).parent` makes the path relative to this test file,
# ensuring that tests will run correctly regardless of where they are executed
# from.
FIXTURES_DIR = Path(__file__).parent / "fixtures"

def test_extract_text_from_pdf():
    """
    Tests successful text extraction from a standard, multi-line PDF document.

    Purpose:
        Verify that the OCR function can correctly process a valid PDF file
        and extract meaningful text content from it.

    Setup:
        - Loads a sample invoice pdf from the fixtures directory.

    Action:
        - Reads the PDF file into bytes.
        - Calls `extract_text_from_document` with the bytes and filename.

    Assertions:
        - The returned output is a non-empty string.
        - The extracted text contains specific, expected keywords "INVOICE" and
          "Total" to confirm the content is correct.
    """
    pdf_path = FIXTURES_DIR / "invoice-template.pdf"
    with open(pdf_path, "rb") as f:
        file_bytes = f.read()

    extracted_text = extract_text_from_document(file_bytes, pdf_path.name)

    assert isinstance(extracted_text, str)
    assert len(extracted_text) > 100, "Extracted text should not be empty for a valid invoice."
    assert "Invoice" in extracted_text, "The keyword 'Invoice' should be present."
    assert "Total" in extracted_text, "The keyword 'Total' should be present."


def test_extract_text_from_image():
    """
    Tests successful text extraction from a standard PNG image file.

    Purpose:
        Verify that the OCR function can handle common image formats and
        extract text content accurately.

    Setup:
        - Loads a sample image from the fixtures directory.

    Action:
        - Reads the image file into bytes.
        - Calls `extract_text_from_document` with the bytes and filename.

    Assertions:
        - The output is a non-empty string.
        - The extracted text contains expected keywords "Receipt" and "Amount".
    """
    image_path = FIXTURES_DIR / "receipt.png"
    with open(image_path, "rb") as f:
        file_bytes = f.read()

    extracted_text = extract_text_from_document(file_bytes, image_path.name)

    assert isinstance(extracted_text, str)
    assert len(extracted_text) > 50, "Extracted text should not be empty for a valid receipt."
    # Note: These keywords should exist in your test image.
    assert "RECEIPT" in extracted_text, "The keyword 'RECEIPT' should be present."
    assert "Amount" in extracted_text, "The keyword 'Amount' should be present."


def test_extract_text_from_blank_document():
    """
    Tests that a blank or empty document results in an empty string.

    Purpose:
        Ensure the function handles edge cases like empty documents gracefully
        without raising an error.

    Setup:
        - Loads a 'blank.pdf' file that contains no text or images.

    Action:
        - Calls `extract_text_from_document`.

    Assertions:
        - The function should return a string that is empty after stripping whitespace.
    """
    blank_pdf_path = FIXTURES_DIR / "blank.pdf"
    with open(blank_pdf_path, "rb") as f:
        file_bytes = f.read()

    extracted_text = extract_text_from_document(file_bytes, blank_pdf_path.name)

    assert isinstance(extracted_text, str)
    assert extracted_text.strip() == "", "A blank document should result in an empty string."


def test_unsupported_file_type():
    """
    Tests that the function returns an empty string for unsupported file types.

    Purpose:
        Verify that the function's file type validation is working correctly
        and that it doesn't attempt to process files it shouldn't.

    Setup:
        - Creates some dummy bytes and a filename with a '.txt' extension.

    Action:
        - Calls `extract_text_from_document` with the unsupported file.

    Assertions:
        - The function should immediately return an empty string.
    """
    dummy_bytes = b"This is a text file, not an image or PDF."
    filename = "document.txt"

    extracted_text = extract_text_from_document(dummy_bytes, filename)

    assert extracted_text == "", "Unsupported file types should return an empty string."
