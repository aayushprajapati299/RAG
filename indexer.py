"""
indexer.py

This script takes the uploaded files and rips the text out, storing it in our trusty PAGE_INDEX.
It handles normal PDFs, plain text, and will even try to OCR scanned PDFs if things get tricky.
"""

import os
import re
import fitz  # pymupdf
import pytesseract
from PIL import Image
import io

# Gotta tell pytesseract where the exe is since we're on Windows
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Quick and dirty in-memory index to hold our pages: {page_number: "text"}
PAGE_INDEX = {}

def extract_text_with_ocr(page) -> str:
    """
    When normal text extraction fails, we turn the page into an image and let OCR take a crack at it.
    """
    pix = page.get_pixmap(dpi=200)
    img_bytes = pix.tobytes("png")
    img = Image.open(io.BytesIO(img_bytes))
    text = pytesseract.image_to_string(img)
    return text.strip()

def index_document(filepath: str) -> int:
    """
    Chews through a file and dumps the text into PAGE_INDEX.
    If it's a PDF, we try normal text extraction or fallback to OCR.
    If it's just a text file, we chunk it up every 500 words and pretend they're pages.
    """
    global PAGE_INDEX
    PAGE_INDEX.clear()

    ext = os.path.splitext(filepath)[1].lower()

    if ext == ".pdf":
        doc = fitz.open(filepath)
        for i, page in enumerate(doc):
            # Let's hope it's not a scanned document
            text = page.get_text().strip()

            # Looks like a scanned image if there's barely any text, time to OCR
            if len(text) < 20:
                text = extract_text_with_ocr(page)

            if text:
                PAGE_INDEX[i + 1] = text

        doc.close()

    elif ext == ".txt":
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()

        words = content.split()
        chunk_size = 500
        chunks = [words[i:i + chunk_size] for i in range(0, len(words), chunk_size)]

        for i, chunk in enumerate(chunks):
            PAGE_INDEX[i + 1] = " ".join(chunk)

    return len(PAGE_INDEX)

def get_page(page_number: int) -> str:
    """Just a quick helper to pull a page's text, or give back nothing if it's missing."""
    return PAGE_INDEX.get(page_number, "")