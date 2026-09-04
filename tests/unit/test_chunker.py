from opsassist.rag.chunker import chunk_document, chunk_documents
from opsassist.rag.loader import Document, compute_checksum


def make_doc(content, doc_id="doc1", source_path="a.md"):
    return Document(
        doc_id=doc_id,
        source_path=source_path,
        content=content,
        checksum=compute_checksum(content),
    )


def test_chunk_document_basic():
    doc = make_doc("# Title\n\nSome short content.")
    chunks = chunk_document(doc)
    assert len(chunks) == 1
    assert chunks[0].heading_path == "Title"
    assert "short content" in chunks[0].text


def test_heading_path_breadcrumb():
    content = (
        "# Runbooks\n\nintro\n\n## Database\n\ndb text\n\n### Restart\n\nrestart text"
    )
    doc = make_doc(content)
    chunks = chunk_document(doc)
    heading_paths = [c.heading_path for c in chunks]
    assert "Runbooks" in heading_paths
    assert "Runbooks > Database" in heading_paths
    assert "Runbooks > Database > Restart" in heading_paths


def test_long_section_is_split_with_overlap():
    long_text = "word " * 500  # ~2500 characters
    content = f"# Long Section\n\n{long_text}"
    doc = make_doc(content)
    chunks = chunk_document(doc, chunk_size=500, overlap=50)
    assert len(chunks) > 1
    # All chunks from a long section share the same heading path.
    assert all(c.heading_path == "Long Section" for c in chunks)


def test_chunk_ids_are_deterministic():
    doc = make_doc("# Title\n\nSame content every time.")
    chunks_a = chunk_document(doc)
    chunks_b = chunk_document(doc)
    assert [c.chunk_id for c in chunks_a] == [c.chunk_id for c in chunks_b]


def test_chunk_documents_handles_multiple_docs():
    doc1 = make_doc("# One\n\ntext one", doc_id="d1", source_path="one.md")
    doc2 = make_doc("# Two\n\ntext two", doc_id="d2", source_path="two.md")
    chunks = chunk_documents([doc1, doc2])
    doc_ids = {c.doc_id for c in chunks}
    assert doc_ids == {"d1", "d2"}


def test_no_heading_still_produces_chunk():
    doc = make_doc("Just plain text with no headings at all.")
    chunks = chunk_document(doc)
    assert len(chunks) == 1
    assert chunks[0].heading_path == ""
