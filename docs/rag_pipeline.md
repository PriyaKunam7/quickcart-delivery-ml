# RAG Document Ingestion Pipeline

Describes the retrieval pipeline that indexes Markdown knowledge base documents (runbooks, incident postmortems, CMDB reference docs) for retrieval-augmented generation.

## Architecture

\`\`\`
knowledge_base/*.md
        │
        ▼
   loader.py          -- reads files, computes checksums, extracts headings
        │
        ▼
   chunker.py          -- heading-aware + recursive splitting into Chunks
        │
        ▼
   indexer.py
     ├── EmbeddingProvider  -- turns chunk text into vectors
     ├── VectorStore        -- stores vectors, supports similarity search
     └── RagIndexer         -- orchestrates ingestion, tracks metadata,
                               skips unchanged documents
        │
        ▼
  data/rag_index/       -- persisted vector store + metadata.json
  reports/rag_index_summary.json  -- last build's summary stats
\`\`\`

## Components

### Loader (\`loader.py\`)
Walks a directory of Markdown files. For each file, records:
- \`source_path\` — path relative to the knowledge base root (stable across machines)
- \`doc_id\` — deterministic hash of \`source_path\`
- \`checksum\` — SHA-256 of file content, used to detect changes on re-ingestion
- \`headings\` — every Markdown heading found in the file, for reference

### Chunker (\`chunker.py\`)
Splits each document into \`Chunk\` objects using two passes:
1. **Heading-aware split** — the document is divided at every Markdown heading, and each resulting section is tagged with a \`heading_path\` breadcrumb (e.g. \`"Runbooks > Database > Restart Steps"\`), preserving where in the document the content came from.
2. **Recursive character split** — any section still longer than \`chunk_size\` (default 800 characters) is further split into overlapping windows (\`overlap\`, default 100 characters), so context isn't lost at chunk boundaries.

Each chunk gets a deterministic \`chunk_id\`, derived from \`doc_id + chunk_index + chunk text\`. This determinism is what makes idempotent re-ingestion possible.

### Indexer (\`indexer.py\`)

**Embedding provider abstraction** — \`EmbeddingProvider\` is an interface with one method, \`embed(texts) -> vectors\`. This project ships \`LocalHashEmbeddingProvider\`, a fully offline, deterministic implementation used for local development and CI (no external API calls, no downloaded models). It is **not semantically meaningful** — it doesn't understand language, it just produces a stable vector for a given piece of text. A production deployment would implement the same interface with a real embedding model (e.g. an OpenAI embeddings endpoint or a local sentence-transformers model) and swap it in without changing any other code.

**Vector store abstraction** — \`VectorStore\` is an interface with \`add\`, \`delete\`, \`search\`, \`persist\`, \`load\`. \`FaissVectorStore\` is the concrete sandbox implementation, using a flat FAISS index with cosine similarity (inner product over normalized vectors).

**\`RagIndexer\`** ties it together and is responsible for the idempotency guarantee in the acceptance criteria:
- Tracks a SHA-256 checksum per document across runs (\`metadata.json\`).
- On each build, if a document's checksum matches what was recorded last time, it's **skipped entirely** — no re-chunking, no re-embedding, no vector store writes.
- If a document is new or its checksum changed, its **old chunks are deleted first**, then the new chunks are embedded and added. This prevents both duplicate chunks and stale chunks from an old version of the document lingering in the index.

### Persisted metadata

Every chunk in the index has a \`ChunkRecord\` with:
- \`chunk_id\`, \`doc_id\`, \`source_path\`, \`heading_path\`
- \`doc_checksum\` — the checksum of the parent document at ingestion time
- \`ingested_at\` — UTC timestamp
- \`embedding_model_version\` — which embedding provider/version produced this chunk's vector

This is stored in \`data/rag_index/metadata.json\` and is what lets a future re-index (or a different embedding model version) be applied safely and traceably.

## Running the pipeline

\`\`\`bash
python scripts/build_index.py
# or, to point at a different knowledge base directory:
python scripts/build_index.py --knowledge-base path/to/docs
\`\`\`

Output: a summary of documents processed/skipped/updated and chunks added, printed to the console and written to \`reports/rag_index_summary.json\`.

## Known limitations (sandbox scope)

- \`LocalHashEmbeddingProvider\` produces vectors with no real semantic meaning — similarity search results are not meaningful for actual retrieval quality yet. Swapping in a real embedding provider is the next step before this pipeline is used for actual question-answering.
- \`FaissVectorStore.delete()\` rebuilds the index from retained vectors rather than removing IDs in place. This is fine at small/sandbox scale; a production system with frequent updates or a large corpus should use \`IndexIDMap2\` with \`remove_ids\`, or a database-backed store such as pgvector.