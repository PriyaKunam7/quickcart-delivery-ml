import os

from opsassist.rag.loader import (
    compute_checksum,
    compute_doc_id,
    extract_headings,
    load_markdown_documents,
)


def write_md(path, content):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


def test_load_markdown_documents(tmp_path):
    write_md(str(tmp_path / "a.md"), "# Title\n\nSome content.")
    write_md(str(tmp_path / "sub" / "b.md"), "# Other\n\nMore content.")
    write_md(str(tmp_path / "not_markdown.txt"), "ignore me")

    docs = load_markdown_documents(str(tmp_path))

    assert len(docs) == 2
    paths = {d.source_path for d in docs}
    assert paths == {"a.md", "sub/b.md"}


def test_checksum_changes_with_content():
    c1 = compute_checksum("hello")
    c2 = compute_checksum("hello world")
    c3 = compute_checksum("hello")
    assert c1 != c2
    assert c1 == c3


def test_doc_id_is_stable_for_same_path():
    id1 = compute_doc_id("runbooks/db.md")
    id2 = compute_doc_id("runbooks/db.md")
    assert id1 == id2


def test_extract_headings():
    content = "# Top\n\ntext\n\n## Sub\n\nmore text\n\n### Deep\n"
    headings = extract_headings(content)
    assert headings == ["# Top", "## Sub", "### Deep"]


def test_load_preserves_headings_metadata(tmp_path):
    write_md(str(tmp_path / "doc.md"), "# Heading One\n\nbody\n\n## Heading Two\n")
    docs = load_markdown_documents(str(tmp_path))
    assert docs[0].headings == ["# Heading One", "## Heading Two"]
