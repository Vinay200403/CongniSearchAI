"""
chunker.py
----------
Splits long document text into overlapping chunks for embedding.

Why overlap matters: without it, an answer that spans a chunk boundary
(e.g. a sentence cut in half between chunk 3 and chunk 4) becomes
unretrievable. A ~15% overlap fixes this at negligible storage cost.

This is a character-based splitter that respects sentence/paragraph
boundaries where possible, rather than cutting mid-word.
"""
import re
from config import CHUNK_SIZE_CHARS, CHUNK_OVERLAP_CHARS


def _split_sentences(text: str):
    # Simple sentence splitter — good enough for chunking, not meant to
    # be a full NLP sentence tokenizer.
    return re.split(r"(?<=[.!?])\s+", text.strip())


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE_CHARS,
               overlap: int = CHUNK_OVERLAP_CHARS):
    """
    Returns a list of text chunks, each roughly `chunk_size` characters,
    with `overlap` characters repeated between consecutive chunks.
    """
    sentences = _split_sentences(text)
    chunks = []
    current = ""

    for sentence in sentences:
        if len(current) + len(sentence) + 1 <= chunk_size:
            current = f"{current} {sentence}".strip()
        else:
            if current:
                chunks.append(current.strip())
            # start new chunk, carrying over the tail of the previous
            # chunk as overlap for context continuity
            tail = current[-overlap:] if overlap and current else ""
            current = f"{tail} {sentence}".strip()

    if current.strip():
        chunks.append(current.strip())

    # Fallback: if a single "sentence" is itself longer than chunk_size
    # (e.g. no punctuation in the doc at all), hard-split it.
    final_chunks = []
    for c in chunks:
        if len(c) <= chunk_size * 1.5:
            final_chunks.append(c)
        else:
            for i in range(0, len(c), chunk_size - overlap):
                final_chunks.append(c[i:i + chunk_size])

    return [c for c in final_chunks if c.strip()]
