"""
Loads Markdown documents from a directory, preserving source path and
heading metadata, and computing a content checksum used later for
change detection during re-ingestion.
"""

import hashlib
import os
import re
from dataclasses import dataclass, field


@dataclass
class Document:
    """A single loaded Markdown document."""

    doc_id: str
    source_path: str
    content: str
    checksum: str
    headings: list[str] = field(default_factory=list)


def compute_checksum(content: str) -> str:
    """Stable content hash used to detect whether a document changed."""
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def compute_doc_id(source_path: str) -> str:
    """Deterministic ID derived from the file's relative path."""
    return hashlib.sha256(source_path.encode("utf-8")).hexdigest()[:16]


def extract_headings(content: str) -> list[str]:
    """Pull every Markdown heading line (# ... through ###### ...) in order."""
    return re.findall(r"^(#{1,6}\s+.+)$", content, flags=re.MULTILINE)


def load_markdown_documents(root_dir: str) -> list[Document]:
    """
    Walk root_dir and load every .md file into a Document.

    Uses the path relative to root_dir as the stable source identifier,
    so moving the whole knowledge base to a new machine doesn't change
    document IDs.
    """
    documents: list[Document] = []

    for dirpath, _dirnames, filenames in os.walk(root_dir):
        for filename in sorted(filenames):
            if not filename.endswith(".md"):
                continue

            full_path = os.path.join(dirpath, filename)
            rel_path = os.path.relpath(full_path, root_dir).replace(os.sep, "/")

            with open(full_path, "r", encoding="utf-8") as f:
                content = f.read()

            doc = Document(
                doc_id=compute_doc_id(rel_path),
                source_path=rel_path,
                content=content,
                checksum=compute_checksum(content),
                headings=extract_headings(content),
            )
            documents.append(doc)

    return documents
