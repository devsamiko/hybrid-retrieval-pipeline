"""
chunking.py — Structure-aware semantic chunking
================================================
Splits documents into retrieval-sized chunks using structure (headings, pages,
paragraphs, sentences) instead of naive fixed-size windows. Adapted from a
production RAG system; markdown/PDF/generic strategies, size bounds, and
overlap are unchanged from the original.

Usage:
    from chunking import semantic_chunk

    chunks = semantic_chunk(text, file_type=".md")
"""

import re

TARGET_SIZE = 800
MAX_SIZE = 1500
MIN_SIZE = 150
OVERLAP = 80

_SENTENCE_RE = re.compile(r'(?<=[.!?])\s+')
_HEADING_RE = re.compile(r'^(#{1,6}\s)', re.MULTILINE)


def _split_sentences(text: str) -> list[str]:
    parts = _SENTENCE_RE.split(text)
    return [p for p in parts if p.strip()]


def _merge_small(chunks: list[str], min_size: int) -> list[str]:
    if not chunks:
        return chunks
    merged = [chunks[0]]
    for c in chunks[1:]:
        if len(merged[-1]) < min_size:
            merged[-1] += "\n\n" + c
        else:
            merged.append(c)
    if len(merged) > 1 and len(merged[-1]) < min_size:
        merged[-2] += "\n\n" + merged[-1]
        merged.pop()
    return merged


def _split_large(chunks: list[str], max_size: int) -> list[str]:
    result = []
    for chunk in chunks:
        if len(chunk) <= max_size:
            result.append(chunk)
            continue
        sentences = _split_sentences(chunk)
        buf = ""
        for sent in sentences:
            if buf and len(buf) + len(sent) + 1 > max_size:
                result.append(buf)
                buf = sent
            else:
                buf = (buf + " " + sent).strip() if buf else sent
        if buf:
            if len(buf) > max_size:
                words = buf.split()
                sub = ""
                for w in words:
                    if sub and len(sub) + len(w) + 1 > max_size:
                        result.append(sub)
                        sub = w
                    else:
                        sub = (sub + " " + w) if sub else w
                if sub:
                    result.append(sub)
            else:
                result.append(buf)
    return result


def _add_overlap(chunks: list[str], overlap: int) -> list[str]:
    if overlap <= 0 or len(chunks) <= 1:
        return chunks
    result = [chunks[0]]
    for i in range(1, len(chunks)):
        prev_tail = chunks[i - 1][-overlap:]
        space_idx = prev_tail.find(" ")
        if space_idx >= 0:
            prev_tail = prev_tail[space_idx + 1:]
        result.append(prev_tail + " " + chunks[i] if prev_tail else chunks[i])
    return result


def _chunk_markdown(text: str) -> list[str]:
    sections = _HEADING_RE.split(text)
    chunks = []
    i = 0
    while i < len(sections):
        part = sections[i]
        if _HEADING_RE.match(part) and i + 1 < len(sections):
            chunks.append(part + sections[i + 1])
            i += 2
        else:
            if part.strip():
                chunks.append(part)
            i += 1

    result = []
    for section in chunks:
        if len(section) <= TARGET_SIZE:
            result.append(section.strip())
        else:
            paragraphs = [p.strip() for p in section.split("\n\n") if p.strip()]
            buf = ""
            for para in paragraphs:
                if buf and len(buf) + len(para) + 2 > TARGET_SIZE:
                    result.append(buf)
                    buf = para
                else:
                    buf = (buf + "\n\n" + para) if buf else para
            if buf:
                result.append(buf)
    return result


def _chunk_pdf(text: str) -> list[str]:
    pages = text.split("\f")
    result = []
    for page in pages:
        page = page.strip()
        if not page:
            continue
        if len(page) <= TARGET_SIZE:
            result.append(page)
        else:
            paragraphs = [p.strip() for p in page.split("\n\n") if p.strip()]
            buf = ""
            for para in paragraphs:
                if buf and len(buf) + len(para) + 2 > TARGET_SIZE:
                    result.append(buf)
                    buf = para
                else:
                    buf = (buf + "\n\n" + para) if buf else para
            if buf:
                result.append(buf)
    return result


def _chunk_generic(text: str) -> list[str]:
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    if not paragraphs:
        paragraphs = [text.strip()]
    result = []
    buf = ""
    for para in paragraphs:
        if buf and len(buf) + len(para) + 2 > TARGET_SIZE:
            result.append(buf)
            buf = para
        else:
            buf = (buf + "\n\n" + para) if buf else para
    if buf:
        result.append(buf)
    return result


def semantic_chunk(text: str, file_type: str = "") -> list[str]:
    """Split text into semantically meaningful, retrieval-sized chunks.

    Args:
        text: Full document text.
        file_type: Extension (".md", ".pdf", or anything else) selects strategy.

    Returns:
        List of chunk strings between MIN_SIZE and MAX_SIZE chars, with
        OVERLAP chars carried over from the previous chunk.
    """
    if not text or not text.strip():
        return []

    text = text.strip()
    if len(text) <= TARGET_SIZE:
        return [text]

    ft = file_type.lower().lstrip(".")
    if ft == "md":
        raw = _chunk_markdown(text)
    elif ft == "pdf":
        raw = _chunk_pdf(text)
    else:
        raw = _chunk_generic(text)

    chunks = _split_large(raw, MAX_SIZE)
    chunks = _merge_small(chunks, MIN_SIZE)
    chunks = _add_overlap(chunks, OVERLAP)
    return [c for c in chunks if c.strip()]
