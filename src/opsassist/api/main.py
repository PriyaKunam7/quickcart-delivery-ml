"""
POST /v1/chat — question answering over the runbook knowledge base.

Retrieves the top-k most similar chunks for the incoming message, and
only answers from that retrieved content. If the best retrieval score
is below RETRIEVAL_SCORE_THRESHOLD, the endpoint explicitly reports
that available sources don't support a confident answer, rather than
letting an answer generator fabricate something.

Run locally:
    uvicorn opsassist.api.main:app --reload --port 8100
"""

import os
import time
import uuid
from abc import ABC, abstractmethod

from fastapi import Depends, FastAPI
from pydantic import BaseModel, Field

from opsassist.rag.indexer import (
    EMBEDDING_DIMENSION,
    FaissVectorStore,
    LocalHashEmbeddingProvider,
    RagIndexer,
)
from opsassist.rag.retriever import RetrievedChunk, Retriever

DEFAULT_TOP_K = 5
RETRIEVAL_SCORE_THRESHOLD = 0.15
INSUFFICIENT_EVIDENCE_MESSAGE = (
    "The available sources do not support a confident answer to this question."
)

PROMPT_TEMPLATE_PATH = os.path.join(
    os.path.dirname(__file__), "..", "prompts", "rag_answer.txt"
)


def load_prompt_template(path: str = PROMPT_TEMPLATE_PATH) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def build_prompt(template: str, question: str, chunks: list[RetrievedChunk]) -> str:
    context_blocks = []
    for i, chunk in enumerate(chunks, start=1):
        context_blocks.append(
            f"[{i}] (source: {chunk.source_path} \u00a7 {chunk.heading_path})\n{chunk.text}"
        )
    context = "\n\n".join(context_blocks)
    return template.format(context=context, question=question)


# --------------------------------------------------------------------
# Answer generator abstraction
# --------------------------------------------------------------------
class AnswerGenerator(ABC):
    model_version: str

    @abstractmethod
    def generate(self, prompt: str, chunks: list[RetrievedChunk]) -> str:
        raise NotImplementedError


class ExtractiveAnswerGenerator(AnswerGenerator):
    """
    Offline placeholder generator: does not call any LLM. It builds the
    same grounded prompt a real model would receive (via build_prompt),
    but "answers" by returning the most relevant retrieved passage
    directly, clearly labeled as extractive.

    A production deployment would implement this same interface with a
    real LLM call (e.g. the Anthropic API) that receives `prompt` as
    its input and returns a synthesized, grounded answer — the rest of
    the pipeline (retrieval, threshold check, citations) is unchanged
    either way.
    """

    model_version = "extractive-answer-v1"

    def generate(self, prompt: str, chunks: list[RetrievedChunk]) -> str:
        top = chunks[0]
        snippet = top.text.strip()
        if len(snippet) > 400:
            snippet = snippet[:400].rsplit(" ", 1)[0] + "..."
        return f"Based on {top.source_path} (\u00a7 {top.heading_path}) [1]: {snippet}"


# --------------------------------------------------------------------
# Runtime: wires together the retriever + generator + prompt template
# --------------------------------------------------------------------
class RagRuntime:
    def __init__(
        self, retriever: Retriever, generator: AnswerGenerator, prompt_template: str
    ):
        self.retriever = retriever
        self.generator = generator
        self.prompt_template = prompt_template


def load_runtime(index_dir: str) -> RagRuntime:
    vector_store_path = os.path.join(index_dir, "vector_store")
    metadata_path = os.path.join(index_dir, "metadata.json")

    embedding_provider = LocalHashEmbeddingProvider(dimension=EMBEDDING_DIMENSION)
    vector_store = FaissVectorStore(dimension=EMBEDDING_DIMENSION)
    if os.path.exists(f"{vector_store_path}.faiss"):
        vector_store.load(vector_store_path)

    # RagIndexer is reused purely to load the persisted chunk metadata
    # (including chunk text) — build() is never called here.
    indexer = RagIndexer(embedding_provider, vector_store, metadata_path)

    retriever = Retriever(embedding_provider, vector_store, indexer.chunk_records)
    generator = ExtractiveAnswerGenerator()
    prompt_template = load_prompt_template()

    return RagRuntime(retriever, generator, prompt_template)


def get_index_dir() -> str:
    return os.environ.get("OPSASSIST_INDEX_DIR", "data/rag_index")


_default_runtime: RagRuntime | None = None


def get_runtime() -> RagRuntime:
    """
    FastAPI dependency. Loads the runtime once from the default index
    location and caches it at module scope. Tests override this
    dependency (see tests/integration/test_rag_chat.py) to point at an
    isolated temporary index instead of touching this process-wide cache.
    """
    global _default_runtime
    if _default_runtime is None:
        _default_runtime = load_runtime(get_index_dir())
    return _default_runtime


# --------------------------------------------------------------------
# API
# --------------------------------------------------------------------
class ChatRequest(BaseModel):
    conversation_id: str | None = None
    message: str
    top_k: int = Field(default=DEFAULT_TOP_K, ge=1, le=20)


class Citation(BaseModel):
    source_path: str
    chunk_id: str
    heading_path: str
    score: float


class ChatResponse(BaseModel):
    conversation_id: str
    message: str
    answer: str
    citations: list[Citation]
    model: str
    latency_ms: float


app = FastAPI(title="OpsAssist RAG Chat API")


@app.post("/v1/chat", response_model=ChatResponse)
def chat(
    request: ChatRequest, runtime: RagRuntime = Depends(get_runtime)
) -> ChatResponse:
    start = time.perf_counter()
    conversation_id = request.conversation_id or str(uuid.uuid4())

    retrieved = runtime.retriever.search(request.message, k=request.top_k)

    if not retrieved or retrieved[0].score < RETRIEVAL_SCORE_THRESHOLD:
        answer = INSUFFICIENT_EVIDENCE_MESSAGE
        citations: list[Citation] = []
        model = "none (insufficient evidence)"
    else:
        prompt = build_prompt(runtime.prompt_template, request.message, retrieved)
        answer = runtime.generator.generate(prompt, retrieved)
        # Only cite chunks that carried genuine signal -- padding the
        # citation list with zero-similarity chunks (which top_k can
        # include once relevant matches run out) would misrepresent
        # what the answer is actually grounded in.
        citations = [
            Citation(
                source_path=c.source_path,
                chunk_id=c.chunk_id,
                heading_path=c.heading_path,
                score=c.score,
            )
            for c in retrieved
            if c.score > 0.0
        ]
        model = runtime.generator.model_version

    latency_ms = (time.perf_counter() - start) * 1000

    return ChatResponse(
        conversation_id=conversation_id,
        message=request.message,
        answer=answer,
        citations=citations,
        model=model,
        latency_ms=round(latency_ms, 2),
    )
