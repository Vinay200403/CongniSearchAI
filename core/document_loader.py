"""
document_loader.py
-------------------
Extracts plain text from an uploaded file, regardless of format.
Each loader is intentionally isolated so adding a new file type later
(e.g. .pptx, .html) only means adding one function + one dispatch entry.
"""
import os
from pypdf import PdfReader
import docx


def _load_pdf(path: str) -> str:
    reader = PdfReader(path)
    pages = []
    for page in reader.pages:
        text = page.extract_text() or ""
        pages.append(text)
    return "\n".join(pages)


def _load_docx(path: str) -> str:
    document = docx.Document(path)
    return "\n".join(p.text for p in document.paragraphs)


def _load_text(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


_LOADERS = {
    "pdf": _load_pdf,
    "docx": _load_docx,
    "txt": _load_text,
    "md": _load_text,
}


def load_document(path: str) -> str:
    """
    Dispatch to the correct loader based on file extension.
    Raises ValueError for unsupported types (should be caught earlier by
    the upload route's extension check, this is a defensive second layer).
    """
    ext = path.rsplit(".", 1)[-1].lower()
    loader = _LOADERS.get(ext)
    if loader is None:
        raise ValueError(f"Unsupported file type: .{ext}")

    text = loader(path)
    if not text or not text.strip():
        raise ValueError(
            "No extractable text found — the file may be a scanned "
            "image PDF (needs OCR, not supported yet) or empty."
        )
    return text
