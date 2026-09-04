"""
Embedding provider abstraction, vector store abstraction, and the
indexing logic that ties them together with persisted metadata.

Design notes:
- EmbeddingProvider is an interface. LocalHashEmbeddingProvider is a
  fully offline, deterministic implementation used for local/sandbox
  development and tests. A production deployment would swap in a real
  provider (OpenAI embeddings API, a local sentence-transformers model,
  etc.) behind the same interface without touching the indexer logic.
- VectorStore is likewise an interface. FaissVectorStore is the
  concrete implementation for this sandbox, backed by a flat FAISS
  index (cosine similarity via normalized inner product).
- RagIndexer is responsible for idempotent ingestion: re-running the
  build on unchanged documents must not create duplicate chunks. It
  does this by tracking a checksum per document in a metadata store
  and skipping re-embedding when the checksum hasn't changed.
"""

import hashlib
import json
import os
import re
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from datetime import datetime, timezone

import numpy as np

from opsassist.rag.chunker import Chunk
from opsassist.rag.loader import Document

EMBEDDING_DIMENSION = 512


# --------------------------------------------------------------------
# Embedding provider abstraction
# --------------------------------------------------------------------
class EmbeddingProvider(ABC):
    """Interface every embedding backend must implement."""

    model_version: str
    dimension: int

    @abstractmethod
    def embed(self, texts: list[str]) -> np.ndarray:
        """Return an (n_texts, dimension) float32 array of embeddings."""
        raise NotImplementedError


class LocalHashEmbeddingProvider(EmbeddingProvider):
    """
    Deterministic, fully offline embedding provider using a bag-of-words
    hashing trick (feature hashing) with stopword filtering: each
    significant token in the text is hashed into a fixed-size vector
    position with a random sign, weighted by log-scaled term frequency,
    and the resulting vector is L2-normalized.

    This is NOT a real semantic embedding model — it has no notion of
    word meaning, synonyms, or grammar. What it *does* capture is
    lexical overlap: two texts that share more of the same significant
    (non-stopword) words will have a higher cosine similarity than two
    texts that don't. That's enough to make retrieval and the
    "insufficient evidence" threshold meaningfully testable offline,
    without any external API calls or downloaded models. Swap this out
    for a real embedding provider (e.g. an OpenAI embeddings endpoint or
    a local sentence-transformers model) before using retrieval results
    for anything user-facing.
    """

    _STOPWORDS = {
        "a",
        "an",
        "the",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "do",
        "does",
        "did",
        "i",
        "you",
        "he",
        "she",
        "it",
        "we",
        "they",
        "what",
        "which",
        "who",
        "whom",
        "this",
        "that",
        "these",
        "those",
        "for",
        "of",
        "in",
        "on",
        "at",
        "to",
        "and",
        "or",
        "as",
        "with",
        "if",
        "how",
        "when",
        "where",
        "why",
        "can",
        "could",
        "should",
        "would",
        "will",
        "shall",
        "may",
        "might",
        "must",
        "not",
        "no",
        "so",
        "than",
        "then",
        "there",
        "here",
        "up",
        "down",
        "out",
        "about",
    }

    def __init__(self, dimension: int = EMBEDDING_DIMENSION):
        self.dimension = dimension
        self.model_version = f"local-hash-embed-v2-dim{dimension}"

    @classmethod
    def _tokenize(cls, text: str) -> list[str]:
        tokens = re.findall(r"[a-z0-9]+", text.lower())
        return [t for t in tokens if t not in cls._STOPWORDS and len(t) > 1]

    def embed(self, texts: list[str]) -> np.ndarray:
        vectors = np.zeros((len(texts), self.dimension), dtype="float32")
        for i, text in enumerate(texts):
            tokens = self._tokenize(text)
            counts: dict[str, int] = {}
            for token in tokens:
                counts[token] = counts.get(token, 0) + 1

            vec = np.zeros(self.dimension, dtype="float32")
            for token, count in counts.items():
                digest = hashlib.sha256(token.encode("utf-8")).digest()
                idx = int.from_bytes(digest[:4], "big") % self.dimension
                sign = 1.0 if digest[4] % 2 == 0 else -1.0
                weight = 1.0 + np.log(count)  # log-scaled term frequency
                vec[idx] += sign * weight

            norm = np.linalg.norm(vec)
            vectors[i] = vec / norm if norm > 0 else vec
        return vectors


# --------------------------------------------------------------------
# Vector store abstraction
# --------------------------------------------------------------------
class VectorStore(ABC):
    @abstractmethod
    def add(self, ids: list[str], vectors: np.ndarray) -> None:
        raise NotImplementedError

    @abstractmethod
    def delete(self, ids: list[str]) -> None:
        raise NotImplementedError

    @abstractmethod
    def search(
        self, query_vector: np.ndarray, top_k: int = 5
    ) -> list[tuple[str, float]]:
        raise NotImplementedError

    @abstractmethod
    def persist(self, path: str) -> None:
        raise NotImplementedError

    @abstractmethod
    def load(self, path: str) -> None:
        raise NotImplementedError

    @property
    @abstractmethod
    def size(self) -> int:
        raise NotImplementedError


class FaissVectorStore(VectorStore):
    """
    Flat FAISS index using inner product over normalized vectors,
    which is equivalent to cosine similarity.

    FAISS itself has no concept of "delete by ID" for a flat index, so
    deletion is implemented by rebuilding the index from the retained
    vectors. Fine at this scale (sandbox / small knowledge bases);
    a production deployment with frequent updates would use FAISS's
    IndexIDMap2 with remove_ids, or a database-backed store like
    pgvector instead.
    """

    def __init__(self, dimension: int = EMBEDDING_DIMENSION):
        import faiss  # imported lazily so the rest of the module works without faiss installed

        self._faiss = faiss
        self.dimension = dimension
        self._index = faiss.IndexFlatIP(dimension)
        self._ids: list[str] = []
        self._vectors: dict[str, np.ndarray] = {}

    @property
    def size(self) -> int:
        return len(self._ids)

    def add(self, ids: list[str], vectors: np.ndarray) -> None:
        if len(ids) != len(vectors):
            raise ValueError("ids and vectors must be the same length")
        self._index.add(vectors)
        self._ids.extend(ids)
        for _id, vec in zip(ids, vectors):
            self._vectors[_id] = vec

    def delete(self, ids: list[str]) -> None:
        ids_to_remove = set(ids)
        remaining_ids = [i for i in self._ids if i not in ids_to_remove]
        remaining_vectors = (
            np.stack([self._vectors[i] for i in remaining_ids])
            if remaining_ids
            else np.zeros((0, self.dimension), dtype="float32")
        )
        self._index = self._faiss.IndexFlatIP(self.dimension)
        self._index.add(remaining_vectors)
        self._ids = remaining_ids
        for i in ids_to_remove:
            self._vectors.pop(i, None)

    def search(
        self, query_vector: np.ndarray, top_k: int = 5
    ) -> list[tuple[str, float]]:
        if self.size == 0:
            return []
        query = query_vector.reshape(1, -1).astype("float32")
        scores, indices = self._index.search(query, min(top_k, self.size))
        results = []
        for idx, score in zip(indices[0], scores[0]):
            if idx == -1:
                continue
            results.append((self._ids[idx], float(score)))
        return results

    def persist(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self._faiss.write_index(self._index, f"{path}.faiss")
        with open(f"{path}.ids.json", "w") as f:
            json.dump(self._ids, f)

    def load(self, path: str) -> None:
        self._index = self._faiss.read_index(f"{path}.faiss")
        with open(f"{path}.ids.json", "r") as f:
            self._ids = json.load(f)
        # Reconstruct vectors from the index so delete() (which rebuilds
        # the index from retained vectors) still works after a reload.
        self._vectors = {
            _id: self._index.reconstruct(i) for i, _id in enumerate(self._ids)
        }


# --------------------------------------------------------------------
# Index metadata (per-chunk record persisted alongside the vector store)
# --------------------------------------------------------------------
@dataclass
class ChunkRecord:
    chunk_id: str
    doc_id: str
    source_path: str
    heading_path: str
    text: str
    doc_checksum: str
    ingested_at: str
    embedding_model_version: str


class RagIndexer:
    """
    Orchestrates chunking, embedding, and storage, with idempotent
    re-ingestion: running the same unchanged documents through build()
    again does not create duplicate chunks or re-call the embedding
    provider for content that hasn't changed.
    """

    def __init__(
        self,
        embedding_provider: EmbeddingProvider,
        vector_store: VectorStore,
        metadata_path: str,
    ):
        self.embedding_provider = embedding_provider
        self.vector_store = vector_store
        self.metadata_path = metadata_path
        self.doc_checksums: dict[str, str] = {}  # doc_id -> checksum
        self.doc_chunk_ids: dict[str, list[str]] = {}  # doc_id -> [chunk_id, ...]
        self.chunk_records: dict[str, ChunkRecord] = {}  # chunk_id -> record

        if os.path.exists(metadata_path):
            self._load_metadata()

    def _load_metadata(self) -> None:
        with open(self.metadata_path, "r") as f:
            data = json.load(f)
        self.doc_checksums = data.get("doc_checksums", {})
        self.doc_chunk_ids = data.get("doc_chunk_ids", {})
        self.chunk_records = {
            chunk_id: ChunkRecord(**record)
            for chunk_id, record in data.get("chunk_records", {}).items()
        }

    def _save_metadata(self) -> None:
        os.makedirs(os.path.dirname(self.metadata_path) or ".", exist_ok=True)
        data = {
            "doc_checksums": self.doc_checksums,
            "doc_chunk_ids": self.doc_chunk_ids,
            "chunk_records": {
                chunk_id: asdict(record)
                for chunk_id, record in self.chunk_records.items()
            },
        }
        with open(self.metadata_path, "w") as f:
            json.dump(data, f, indent=2)

    def build(
        self, documents: list[Document], chunks_by_doc: dict[str, list[Chunk]]
    ) -> dict:
        """
        Ingest documents. Skips any document whose checksum matches what
        was recorded on a previous run. Returns a summary dict.
        """
        added_chunks = 0
        skipped_docs = 0
        updated_docs = 0

        for doc in documents:
            previous_checksum = self.doc_checksums.get(doc.doc_id)

            if previous_checksum == doc.checksum:
                skipped_docs += 1
                continue

            # Document is new or changed. Remove any previously indexed
            # chunks for this doc before adding the fresh set.
            old_chunk_ids = self.doc_chunk_ids.get(doc.doc_id, [])
            if old_chunk_ids:
                self.vector_store.delete(old_chunk_ids)
                for cid in old_chunk_ids:
                    self.chunk_records.pop(cid, None)
                updated_docs += 1

            new_chunks = chunks_by_doc.get(doc.doc_id, [])
            if not new_chunks:
                self.doc_checksums[doc.doc_id] = doc.checksum
                self.doc_chunk_ids[doc.doc_id] = []
                continue

            texts = [c.text for c in new_chunks]
            vectors = self.embedding_provider.embed(texts)
            chunk_ids = [c.chunk_id for c in new_chunks]
            self.vector_store.add(chunk_ids, vectors)

            ingested_at = datetime.now(timezone.utc).isoformat()
            for c in new_chunks:
                self.chunk_records[c.chunk_id] = ChunkRecord(
                    chunk_id=c.chunk_id,
                    doc_id=c.doc_id,
                    source_path=c.source_path,
                    heading_path=c.heading_path,
                    text=c.text,
                    doc_checksum=doc.checksum,
                    ingested_at=ingested_at,
                    embedding_model_version=self.embedding_provider.model_version,
                )
            added_chunks += len(new_chunks)

            self.doc_checksums[doc.doc_id] = doc.checksum
            self.doc_chunk_ids[doc.doc_id] = chunk_ids

        self._save_metadata()

        return {
            "documents_processed": len(documents),
            "documents_skipped_unchanged": skipped_docs,
            "documents_updated": updated_docs,
            "chunks_added": added_chunks,
            "total_chunks_in_index": self.vector_store.size,
        }
