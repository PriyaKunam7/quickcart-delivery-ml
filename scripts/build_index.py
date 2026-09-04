"""
End-to-end pipeline: load Markdown documents, chunk them, embed the
chunks, and persist a searchable vector index plus index metadata.

Usage:
    python scripts/build_index.py
    python scripts/build_index.py --knowledge-base path/to/docs
"""

import argparse
import json
import os

from opsassist.rag.chunker import chunk_documents
from opsassist.rag.indexer import (
    EMBEDDING_DIMENSION,
    FaissVectorStore,
    LocalHashEmbeddingProvider,
    RagIndexer,
)
from opsassist.rag.loader import load_markdown_documents

DEFAULT_KNOWLEDGE_BASE = "knowledge_base"
INDEX_DIR = "data/rag_index"
VECTOR_STORE_PATH = os.path.join(INDEX_DIR, "vector_store")
METADATA_PATH = os.path.join(INDEX_DIR, "metadata.json")
SUMMARY_REPORT_PATH = "reports/rag_index_summary.json"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--knowledge-base", default=DEFAULT_KNOWLEDGE_BASE)
    args = parser.parse_args()

    documents = load_markdown_documents(args.knowledge_base)
    print(f"Loaded {len(documents)} document(s) from '{args.knowledge_base}'.")

    chunks_by_doc = {}
    total_chunks = 0
    for doc in documents:
        doc_chunks = chunk_documents([doc])
        chunks_by_doc[doc.doc_id] = doc_chunks
        total_chunks += len(doc_chunks)
    print(f"Produced {total_chunks} chunk(s) before dedup.")

    embedding_provider = LocalHashEmbeddingProvider(dimension=EMBEDDING_DIMENSION)

    vector_store = FaissVectorStore(dimension=EMBEDDING_DIMENSION)
    if os.path.exists(f"{VECTOR_STORE_PATH}.faiss"):
        vector_store.load(VECTOR_STORE_PATH)

    indexer = RagIndexer(
        embedding_provider=embedding_provider,
        vector_store=vector_store,
        metadata_path=METADATA_PATH,
    )

    summary = indexer.build(documents, chunks_by_doc)
    vector_store.persist(VECTOR_STORE_PATH)

    os.makedirs(os.path.dirname(SUMMARY_REPORT_PATH), exist_ok=True)
    with open(SUMMARY_REPORT_PATH, "w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))
    print(f"Summary written to {SUMMARY_REPORT_PATH}")


if __name__ == "__main__":
    main()
