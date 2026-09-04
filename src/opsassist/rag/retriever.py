"""
Top-k similarity retriever over an already-built RAG index.

Loads the persisted vector store and chunk metadata (including chunk
text) written by scripts/build_index.py, and answers queries with the
most similar chunks plus their similarity scores.
"""

from dataclasses import dataclass

from opsassist.rag.indexer import (
    ChunkRecord,
    EmbeddingProvider,
    VectorStore,
)


@dataclass
class RetrievedChunk:
    chunk_id: str
    doc_id: str
    source_path: str
    heading_path: str
    text: str
    score: float


class Retriever:
    """
    Wraps an embedding provider and a vector store to answer top-k
    similarity queries, resolving vector store hits back to full chunk
    records (including text) via the metadata loaded from disk.
    """

    def __init__(
        self,
        embedding_provider: EmbeddingProvider,
        vector_store: VectorStore,
        chunk_records: dict[str, ChunkRecord],
    ):
        self.embedding_provider = embedding_provider
        self.vector_store = vector_store
        self.chunk_records = chunk_records

    def search(self, query: str, k: int = 5) -> list[RetrievedChunk]:
        """Return up to k most similar chunks to the query, best first."""
        query_vector = self.embedding_provider.embed([query])[0]
        hits = self.vector_store.search(query_vector, top_k=k)

        results: list[RetrievedChunk] = []
        for chunk_id, score in hits:
            record = self.chunk_records.get(chunk_id)
            if record is None:
                # Vector store and metadata store are out of sync; skip
                # rather than return a chunk we have no text/source for.
                continue
            results.append(
                RetrievedChunk(
                    chunk_id=record.chunk_id,
                    doc_id=record.doc_id,
                    source_path=record.source_path,
                    heading_path=record.heading_path,
                    text=record.text,
                    score=score,
                )
            )
        return results
