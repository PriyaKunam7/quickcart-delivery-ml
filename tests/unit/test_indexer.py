import numpy as np

from opsassist.rag.chunker import chunk_documents
from opsassist.rag.indexer import (
    FaissVectorStore,
    LocalHashEmbeddingProvider,
    RagIndexer,
)
from opsassist.rag.loader import Document, compute_checksum, compute_doc_id


def make_doc(content, source_path="a.md"):
    return Document(
        doc_id=compute_doc_id(source_path),
        source_path=source_path,
        content=content,
        checksum=compute_checksum(content),
    )


def build_indexer(tmp_path, dim=32):
    embedding_provider = LocalHashEmbeddingProvider(dimension=dim)
    vector_store = FaissVectorStore(dimension=dim)
    metadata_path = str(tmp_path / "metadata.json")
    indexer = RagIndexer(embedding_provider, vector_store, metadata_path)
    return indexer, vector_store, embedding_provider


def test_embedding_provider_is_deterministic():
    provider = LocalHashEmbeddingProvider(dimension=16)
    v1 = provider.embed(["hello world"])
    v2 = provider.embed(["hello world"])
    np.testing.assert_array_equal(v1, v2)


def test_embedding_provider_different_text_different_vector():
    provider = LocalHashEmbeddingProvider(dimension=16)
    v1 = provider.embed(["hello"])
    v2 = provider.embed(["goodbye"])
    assert not np.allclose(v1, v2)


def test_fresh_build_adds_all_chunks(tmp_path):
    indexer, vector_store, _ = build_indexer(tmp_path)
    doc = make_doc("# Title\n\nSome content here.")
    chunks_by_doc = {doc.doc_id: chunk_documents([doc])}

    summary = indexer.build([doc], chunks_by_doc)

    assert summary["documents_skipped_unchanged"] == 0
    assert summary["chunks_added"] > 0
    assert vector_store.size == summary["chunks_added"]


def test_rerun_unchanged_document_skips_and_no_duplicates(tmp_path):
    indexer, vector_store, _ = build_indexer(tmp_path)
    doc = make_doc("# Title\n\nSome content here.")
    chunks_by_doc = {doc.doc_id: chunk_documents([doc])}

    first = indexer.build([doc], chunks_by_doc)
    second = indexer.build([doc], chunks_by_doc)

    assert second["documents_skipped_unchanged"] == 1
    assert second["chunks_added"] == 0
    assert second["total_chunks_in_index"] == first["total_chunks_in_index"]


def test_changed_document_replaces_its_chunks_only(tmp_path):
    indexer, vector_store, _ = build_indexer(tmp_path)

    doc_a = make_doc("# A\n\noriginal content", source_path="a.md")
    doc_b = make_doc("# B\n\nunrelated content", source_path="b.md")
    chunks_by_doc = {
        doc_a.doc_id: chunk_documents([doc_a]),
        doc_b.doc_id: chunk_documents([doc_b]),
    }
    indexer.build([doc_a, doc_b], chunks_by_doc)
    b_chunk_ids_before = set(indexer.doc_chunk_ids[doc_b.doc_id])

    doc_a_changed = make_doc(
        "# A\n\ncompletely different content now", source_path="a.md"
    )
    chunks_by_doc_2 = {
        doc_a_changed.doc_id: chunk_documents([doc_a_changed]),
        doc_b.doc_id: chunk_documents([doc_b]),
    }
    summary = indexer.build([doc_a_changed, doc_b], chunks_by_doc_2)

    assert summary["documents_updated"] == 1
    assert summary["documents_skipped_unchanged"] == 1
    b_chunk_ids_after = set(indexer.doc_chunk_ids[doc_b.doc_id])
    assert b_chunk_ids_before == b_chunk_ids_after  # untouched


def test_metadata_persists_across_indexer_instances(tmp_path):
    indexer1, vector_store1, _ = build_indexer(tmp_path)
    doc = make_doc("# Title\n\nContent.")
    chunks_by_doc = {doc.doc_id: chunk_documents([doc])}
    indexer1.build([doc], chunks_by_doc)
    vector_store1.persist(str(tmp_path / "store"))

    # Simulate a fresh process: new indexer instance loading from disk.
    embedding_provider = LocalHashEmbeddingProvider(dimension=32)
    vector_store2 = FaissVectorStore(dimension=32)
    vector_store2.load(str(tmp_path / "store"))
    indexer2 = RagIndexer(
        embedding_provider, vector_store2, str(tmp_path / "metadata.json")
    )

    summary = indexer2.build([doc], {doc.doc_id: chunks_by_doc[doc.doc_id]})
    assert summary["documents_skipped_unchanged"] == 1
    assert summary["chunks_added"] == 0


def test_every_chunk_has_required_metadata(tmp_path):
    indexer, _, _ = build_indexer(tmp_path)
    doc = make_doc("# Title\n\nSome content here.")
    chunks_by_doc = {doc.doc_id: chunk_documents([doc])}
    indexer.build([doc], chunks_by_doc)

    for record in indexer.chunk_records.values():
        assert record.doc_id
        assert record.source_path
        assert record.doc_checksum
        assert record.ingested_at
        assert record.embedding_model_version
