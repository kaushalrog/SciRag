import fitz  # PyMuPDF
import os
from typing import List

class PDFExtractor:
    """
    Extracts text from scientific PDFs using PyMuPDF.
    """
    def __init__(self):
        pass

    def extract_text(self, pdf_path: str) -> str:
        """
        Extracts raw text from a PDF file.
        """
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF not found: {pdf_path}")
            
        try:
            doc = fitz.open(pdf_path)
            text = ""
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                text += page.get_text("text") + "\n"
            return text
        except Exception as e:
            print(f"Error reading {pdf_path}: {e}")
            return ""
