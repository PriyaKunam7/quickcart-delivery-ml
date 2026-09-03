"""
Splits loaded documents into chunks for embedding.

Chunking is heading-aware: the document is first split on Markdown
headings so each chunk knows which section it came from (a "heading
path" breadcrumb like "Runbooks > Database > Restart Steps"). Any
section still longer than chunk_size is then split further using a
recursive character-based splitter with configurable overlap.
"""

import hashlib
import re
from dataclasses import dataclass

from opsassist.rag.loader import Document

DEFAULT_CHUNK_SIZE = 800
DEFAULT_CHUNK_OVERLAP = 100

HEADING_PATTERN = re.compile(r"^(#{1,6})\s+(.+)$", flags=re.MULTILINE)


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    source_path: str
    heading_path: str
    text: str
    chunk_index: int


def _split_by_headings(content: str) -> list[tuple[str, str]]:
    """
    Split content into (heading_path, section_text) pairs.

    heading_path is a breadcrumb of nested headings, e.g.
    "Runbooks > Database > Restart Steps". Content before the first
    heading gets an empty heading_path.
    """
    matches = list(HEADING_PATTERN.finditer(content))
    if not matches:
        return [("", content.strip())] if content.strip() else []

    sections = []
    stack: list[tuple[int, str]] = []  # (heading_level, heading_text)

    # Leading content before the first heading, if any.
    if matches[0].start() > 0:
        leading = content[: matches[0].start()].strip()
        if leading:
            sections.append(("", leading))

    for i, match in enumerate(matches):
        level = len(match.group(1))
        heading_text = match.group(2).strip()

        # Pop deeper/equal headings off the stack, then push this one.
        while stack and stack[-1][0] >= level:
            stack.pop()
        stack.append((level, heading_text))

        heading_path = " > ".join(h for _, h in stack)

        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(content)
        section_text = content[start:end].strip()

        if section_text:
            sections.append((heading_path, section_text))

    return sections


def _recursive_split(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Split text into overlapping windows of roughly chunk_size characters."""
    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        if end == len(text):
            break
        start = end - overlap  # step forward, keeping `overlap` chars of context
    return chunks


def _make_chunk_id(doc_id: str, chunk_index: int, text: str) -> str:
    """
    Deterministic chunk ID. Same doc_id + index + text always produces
    the same ID, which is what lets the indexer detect unchanged chunks
    and skip re-embedding them.
    """
    payload = f"{doc_id}:{chunk_index}:{text}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def chunk_document(
    doc: Document,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[Chunk]:
    sections = _split_by_headings(doc.content)
    chunks: list[Chunk] = []
    chunk_index = 0

    for heading_path, section_text in sections:
        for piece in _recursive_split(section_text, chunk_size, overlap):
            chunks.append(
                Chunk(
                    chunk_id=_make_chunk_id(doc.doc_id, chunk_index, piece),
                    doc_id=doc.doc_id,
                    source_path=doc.source_path,
                    heading_path=heading_path,
                    text=piece,
                    chunk_index=chunk_index,
                )
            )
            chunk_index += 1

    return chunks


def chunk_documents(
    documents: list[Document],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[Chunk]:
    all_chunks: list[Chunk] = []
    for doc in documents:
        all_chunks.extend(chunk_document(doc, chunk_size, overlap))
    return all_chunks
