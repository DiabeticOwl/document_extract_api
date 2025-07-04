import easyocr
import pymupdf
import io

from PIL import Image

reader = easyocr.Reader(['en'])


def extract_text_from_document(file_bytes: bytes, filename: str) -> str:
    """Extracts text from a document (image or PDF)."""
    text = ""
    # TODO: Change this to use pathlib.
    # TODO: Enhance the file type management, using the suffix should not be enough.
    # TODO: Double check that the tokens are read properly, currently the `text`
    # variable is showing a character per line, instead of whole words.
    if filename.lower().endswith(".pdf"):
        pdf_document = pymupdf.open(stream=file_bytes, filetype="pdf")
        for page_num in range(len(pdf_document)):
            page = pdf_document.load_page(page_num)
            pix = page.get_pixmap()
            img_bytes = pix.tobytes("png")
            result = reader.readtext(img_bytes, detail=0, paragraph=True)
            text += "\n".join(result)
        pdf_document.close()
    else:
        result = reader.readtext(file_bytes, detail=0, paragraph=True)
        text = "\n".join(result)

    return text
