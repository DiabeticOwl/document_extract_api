import easyocr
import pymupdf

from pathlib import Path

reader = easyocr.Reader(['en'])

SUPPORTED_PDF_FORMATS = (".pdf",)
SUPPORTED_IMAGE_FORMATS = (".png", ".jpg", ".jpeg", ".bmp", ".tiff")


def extract_text_from_document(
        file_bytes: bytes,
        filename: str,
        reader: easyocr.Reader = None
) -> str:
    """
    Extracts text from a given document file (PDF or image).

    This function acts as a versatile OCR processor. It determines the file type
    based on the filename extension and applies the appropriate processing logic.
    For PDFs, it iterates through each page, converts it to a high-quality image,
    and then performs OCR. For images, it performs OCR directly.

    This function accepts an optional, pre-initialized EasyOCR reader. If no
    reader is provided, it will create its own for single-use cases
    (like the API endpoint). This makes the function flexible for both
    single-threaded and multi-threaded environments.

    Args:
        file_bytes (bytes): The raw byte content of the file uploaded by the user.
        filename (str): The original name of the file (e.g., "my_invoice.pdf").
                        Used to determine the file type.
        reader (easyocr.Reader, optional): A pre-initialized EasyOCR reader instance.
                                           If None, a new one is created. Defaults to None.

    Returns:
        str: A single string containing all the extracted text, with different
             text blocks separated by newline characters. Returns an empty
             string if an error occurs or no text is found.
    """
    if reader is None:
        reader = easyocr.Reader(['en'])
    doc_path = Path(filename)
    file_suffix = doc_path.suffix.lower()

    # A list to aggregate all recognized text blocks from the document.
    text_blocks = []

    if file_suffix in SUPPORTED_PDF_FORMATS:
        try:
            pdf_document = pymupdf.open(stream=file_bytes, filetype="pdf")
        except Exception as e:
            # This can fail if the PDF is password-protected, corrupted,
            # or not a valid PDF.
            print(f"Failed to open PDF '{filename}'. It may be corrupted or invalid. Error: {e}")
            return ""

        # Iterate over each page in the PDF document.
        for page_num in range(len(pdf_document)):
            try:
                page = pdf_document.load_page(page_num)

                # Convert the page into a pixmap (a raster image representation).
                pix = page.get_pixmap(dpi=400)
                img_bytes = pix.tobytes("png")

                result = reader.readtext(img_bytes, detail=0, paragraph=True)
                text_blocks.extend(result)
            except Exception as e:
                # This might fail if a single page is corrupted. We log the error
                # and continue to the next pages instead of failing the whole document.
                print(f"Could not process page {page_num} of '{filename}'. Error: {e}")
                continue

        pdf_document.close()

    elif file_suffix in SUPPORTED_IMAGE_FORMATS:
        try:
            # For image files, we can pass the bytes directly.
            result = reader.readtext(file_bytes, detail=0, paragraph=True)
            text_blocks.extend(result)
        except Exception as e:
            # This can fail if the image data is malformed or unsupported by
            # the underlying library.
            print(f"EasyOCR failed to process image '{filename}'. Error: {e}")
            return ""

    else:
        print(f"Unsupported file type '{file_suffix}' for file '{filename}'.")
        return ""

    # --- Final Output ---
    # Join all the extracted text blocks into a single string.
    return "\n".join(text_blocks)
