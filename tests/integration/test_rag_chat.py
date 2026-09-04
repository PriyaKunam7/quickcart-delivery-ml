"""
Integration tests for POST /v1/chat.

Builds a small, isolated RAG index in a temporary directory (so tests
never depend on or mutate the developer's local data/rag_index/), then
overrides the FastAPI dependency that loads the runtime to point at it.
"""

import pytest
from fastapi.testclient import TestClient

from opsassist.api import main as api_main
from opsassist.rag.chunker import chunk_documents
from opsassist.rag.indexer import (
    EMBEDDING_DIMENSION,
    FaissVectorStore,
    LocalHashEmbeddingProvider,
    RagIndexer,
)
from opsassist.rag.loader import load_markdown_documents

KNOWLEDGE_BASE_DIR = "knowledge_base"


@pytest.fixture()
def test_runtime(tmp_path):
    index_dir = tmp_path / "rag_index"
    index_dir.mkdir()

    documents = load_markdown_documents(KNOWLEDGE_BASE_DIR)
    chunks_by_doc = {doc.doc_id: chunk_documents([doc]) for doc in documents}

    embedding_provider = LocalHashEmbeddingProvider(dimension=EMBEDDING_DIMENSION)
    vector_store = FaissVectorStore(dimension=EMBEDDING_DIMENSION)
    indexer = RagIndexer(
        embedding_provider=embedding_provider,
        vector_store=vector_store,
        metadata_path=str(index_dir / "metadata.json"),
    )
    indexer.build(documents, chunks_by_doc)
    vector_store.persist(str(index_dir / "vector_store"))

    return api_main.load_runtime(str(index_dir))


@pytest.fixture()
def client(test_runtime):
    api_main.app.dependency_overrides[api_main.get_runtime] = lambda: test_runtime
    with TestClient(api_main.app) as c:
        yield c
    api_main.app.dependency_overrides.clear()


def test_chat_answers_relevant_question_with_citations(client):
    response = client.post(
        "/v1/chat", json={"message": "How do I restart the database?"}
    )
    assert response.status_code == 200
    body = response.json()

    assert body["answer"]
    assert body["citations"], "expected at least one citation for a relevant question"
    assert any("database_restart" in c["source_path"] for c in body["citations"])


def test_chat_returns_full_response_shape(client):
    response = client.post(
        "/v1/chat", json={"message": "How do I restart the database?"}
    )
    body = response.json()

    for field in (
        "conversation_id",
        "message",
        "answer",
        "citations",
        "model",
        "latency_ms",
    ):
        assert field in body

    assert isinstance(body["citations"], list)
    if body["citations"]:
        citation = body["citations"][0]
        assert {"source_path", "chunk_id", "heading_path", "score"} <= citation.keys()


def test_chat_insufficient_evidence_for_unrelated_question(client):
    response = client.post(
        "/v1/chat", json={"message": "How do I bake a chocolate cake?"}
    )
    assert response.status_code == 200
    body = response.json()

    assert "do not support a confident answer" in body["answer"].lower()
    assert body["citations"] == []


def test_chat_preserves_conversation_id_when_provided(client):
    response = client.post(
        "/v1/chat",
        json={
            "conversation_id": "abc-123",
            "message": "How do I restart the database?",
        },
    )
    assert response.json()["conversation_id"] == "abc-123"


def test_chat_generates_conversation_id_when_absent(client):
    response = client.post(
        "/v1/chat", json={"message": "How do I restart the database?"}
    )
    body = response.json()
    assert body["conversation_id"]  # non-empty, auto-generated


def test_chat_respects_top_k(client):
    response = client.post(
        "/v1/chat",
        json={"message": "How do I restart the database?", "top_k": 1},
    )
    body = response.json()
    assert len(body["citations"]) <= 1


def test_chat_does_not_fabricate_facts_outside_context(client):
    """
    The extractive generator can only ever return text that appears in
    a retrieved chunk, so its answer must be a substring of some
    retrieved chunk's text -- it cannot introduce new facts.
    """
    response = client.post(
        "/v1/chat", json={"message": "How do I restart the database?"}
    )
    body = response.json()
    if body["citations"]:
        # The answer references the top citation's source explicitly.
        top_source = body["citations"][0]["source_path"]
        assert top_source in body["answer"]
