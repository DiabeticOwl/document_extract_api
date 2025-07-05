import io
import easyocr
import pymupdf

from core.utils import noise_reduction, adaptive_thresholding, deskew
from pathlib import Path
from PIL import Image

# This module is the heart of the Optical Character Recognition (OCR) pipeline.
# Its primary responsibility is to take a raw document file (image or PDF)
# and extract all readable text from it. It is designed to be flexible,
# supporting various file types and optional image preprocessing steps to
# improve accuracy on low-quality documents.

reader = easyocr.Reader(['en'])

# Define supported file formats by their extensions.
SUPPORTED_PDF_FORMATS = (".pdf",)
SUPPORTED_IMAGE_FORMATS = (".png", ".jpg", ".jpeg", ".bmp", ".tiff")
SUPPORTED_FORMATS = SUPPORTED_PDF_FORMATS + SUPPORTED_IMAGE_FORMATS

# Define supported file formats by their MIME types for API validation.
# This creates a single source of truth for what the application can handle.
SUPPORTED_MIME_TYPES = (
    "application/pdf",
    "image/png",
    "image/jpeg",
    "image/bmp",
    "image/tiff",
)


def extract_text_from_document(
    file_bytes: bytes,
    filename: str,
    reader: easyocr.Reader = None,
    preprocessing: str = None
) -> str:
    """
    Extracts text from a document, with optional preprocessing.

    This is the main function of the module. It orchestrates the process of
    reading a document from its byte representation, applying an optional
    image enhancement technique, and then using the EasyOCR engine to
    extract the text.

    Args:
        file_bytes (bytes): The raw byte content of the file.
        filename (str): The original name of the file, used to determine its type (PDF vs. image).
        reader (easyocr.Reader, optional): A pre-initialized EasyOCR reader instance.
            This is a key optimization for parallel processing, allowing worker
            processes to reuse a single loaded model instead of re-initializing
            it for every task. If None, a new reader is created. Defaults to None.
        preprocessing (str, optional): A string specifying which preprocessing
            step to apply. Can be 'deskew', 'noise', 'threshold', or None.
            Defaults to None.

    Returns:
        str: A single string containing all the extracted text, with different
             text blocks separated by newline characters.
    """
    # If no reader is provided, create a temporary one. This is useful for
    # single-threaded use cases, like the final API endpoint.
    if reader is None:
        reader = easyocr.Reader(['en'])

    doc_path = Path(filename)
    file_suffix = doc_path.suffix.lower()
    # A list to aggregate all recognized text blocks from the document.
    text_blocks = []

    # --- PDF Processing Logic ---
    if file_suffix in SUPPORTED_PDF_FORMATS:
        try:
            # This can fail if the PDF is password-protected, corrupted, or not a valid PDF.
            pdf_document = pymupdf.open(stream=file_bytes, filetype="pdf")
        except Exception as e:
            print(f"Failed to open PDF '{filename}'. It may be corrupted or invalid. Error: {e}")
            return ""

        # Iterate through each page of the PDF.
        for page_num, page in enumerate(pdf_document):
            try:
                # This block handles errors on a per-page basis. If one page fails,
                # we can log the error and continue to the next, making the process more resilient.
                pix = page.get_pixmap(dpi=400)
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

                if preprocessing == 'deskew':
                    img = deskew(img)
                elif preprocessing == 'noise':
                    img = noise_reduction(img)
                elif preprocessing == 'threshold':
                    img = adaptive_thresholding(img)

                buf = io.BytesIO()
                img.save(buf, format='PNG')
                img_bytes = buf.getvalue()

                # This call to readtext can also fail if the image of the page is unreadable.
                result = reader.readtext(img_bytes, detail=0)
                text_blocks.extend(result)
            except Exception as e:
                print(f"Could not process page {page_num + 1} of '{filename}'. Error: {e}. Skipping to next page.")
                continue # Continue to the next page

        pdf_document.close()

    # --- Image Processing Logic ---
    elif file_suffix in SUPPORTED_IMAGE_FORMATS:
        try:
            img = Image.open(io.BytesIO(file_bytes))

            if preprocessing == 'deskew':
                img = deskew(img)
            elif preprocessing == 'noise':
                img = noise_reduction(img)
            elif preprocessing == 'threshold':
                img = adaptive_thresholding(img)

            buf = io.BytesIO()
            img.save(buf, format='PNG')
            img_bytes = buf.getvalue()

            result = reader.readtext(img_bytes, detail=0)
            text_blocks.extend(result)
        except Exception as e:
            # This can fail if the image data is malformed or unsupported by the underlying library.
            print(f"Failed to process image '{filename}'. Error: {e}")
            return ""

    # Join all the extracted text blocks into a single string.
    return "\n".join(text_blocks)
