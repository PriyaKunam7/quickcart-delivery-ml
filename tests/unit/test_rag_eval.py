from opsassist.evals.run_rag_eval import (
    HeuristicAnswerJudge,
    hit_at_k,
    reciprocal_rank,
    run_evaluation,
)
from opsassist.rag.retriever import RetrievedChunk


def make_chunk(source_path, score, heading_path="Section", text="some text"):
    return RetrievedChunk(
        chunk_id=f"chunk-{source_path}-{heading_path}",
        doc_id="doc",
        source_path=source_path,
        heading_path=heading_path,
        text=text,
        score=score,
    )


# --------------------------------------------------------------------
# Retrieval metrics
# --------------------------------------------------------------------
def test_hit_at_k_true_when_expected_source_present():
    retrieved = [make_chunk("a.md", 0.9), make_chunk("b.md", 0.5)]
    assert hit_at_k(retrieved, "b.md") is True


def test_hit_at_k_false_when_expected_source_absent():
    retrieved = [make_chunk("a.md", 0.9), make_chunk("b.md", 0.5)]
    assert hit_at_k(retrieved, "c.md") is False


def test_hit_at_k_false_on_empty_retrieval():
    assert hit_at_k([], "a.md") is False


def test_reciprocal_rank_first_position():
    retrieved = [make_chunk("a.md", 0.9), make_chunk("b.md", 0.5)]
    assert reciprocal_rank(retrieved, "a.md") == 1.0


def test_reciprocal_rank_second_position():
    retrieved = [make_chunk("a.md", 0.9), make_chunk("b.md", 0.5)]
    assert reciprocal_rank(retrieved, "b.md") == 0.5


def test_reciprocal_rank_zero_when_absent():
    retrieved = [make_chunk("a.md", 0.9)]
    assert reciprocal_rank(retrieved, "z.md") == 0.0


# --------------------------------------------------------------------
# Heuristic answer judge
# --------------------------------------------------------------------
def test_judge_groundedness_true_when_citation_matches_expected_source():
    judge = HeuristicAnswerJudge()
    citations = [make_chunk("a.md", 0.5)]
    score = judge.judge("q", "some answer", "some reference", "a.md", citations)
    assert score.groundedness == 1.0


def test_judge_groundedness_false_when_citation_is_wrong_source():
    judge = HeuristicAnswerJudge()
    citations = [make_chunk("b.md", 0.5)]
    score = judge.judge("q", "some answer", "some reference", "a.md", citations)
    assert score.groundedness == 0.0


def test_judge_groundedness_true_for_correct_refusal():
    judge = HeuristicAnswerJudge()
    score = judge.judge("q", "insufficient evidence", None, None, citations=[])
    assert score.groundedness == 1.0


def test_judge_groundedness_false_for_fabricated_answer_on_unanswerable_item():
    judge = HeuristicAnswerJudge()
    citations = [make_chunk("a.md", 0.5)]
    score = judge.judge("q", "an answer that should not exist", None, None, citations)
    assert score.groundedness == 0.0


def test_judge_relevance_and_completeness_full_overlap():
    judge = HeuristicAnswerJudge()
    score = judge.judge(
        "q",
        "notify the on-call channel",
        "notify the on-call channel",
        "a.md",
        citations=[make_chunk("a.md", 0.5)],
    )
    assert score.relevance == 1.0
    assert score.completeness == 1.0


def test_judge_relevance_and_completeness_no_overlap():
    judge = HeuristicAnswerJudge()
    score = judge.judge(
        "q",
        "completely unrelated words here",
        "notify the on-call channel",
        "a.md",
        citations=[make_chunk("a.md", 0.5)],
    )
    assert score.relevance == 0.0
    assert score.completeness == 0.0


# --------------------------------------------------------------------
# End-to-end evaluation run (small isolated index, mirrors integration style)
# --------------------------------------------------------------------
def test_run_evaluation_produces_expected_summary_shape(tmp_path):
    import json

    from opsassist.rag.chunker import chunk_documents
    from opsassist.rag.indexer import (
        EMBEDDING_DIMENSION,
        FaissVectorStore,
        LocalHashEmbeddingProvider,
        RagIndexer,
    )
    from opsassist.rag.loader import load_markdown_documents

    index_dir = tmp_path / "rag_index"
    index_dir.mkdir()
    documents = load_markdown_documents("knowledge_base")
    chunks_by_doc = {doc.doc_id: chunk_documents([doc]) for doc in documents}

    embedding_provider = LocalHashEmbeddingProvider(dimension=EMBEDDING_DIMENSION)
    vector_store = FaissVectorStore(dimension=EMBEDDING_DIMENSION)
    indexer = RagIndexer(
        embedding_provider, vector_store, str(index_dir / "metadata.json")
    )
    indexer.build(documents, chunks_by_doc)
    vector_store.persist(str(index_dir / "vector_store"))

    tiny_dataset = {
        "version": "test",
        "items": [
            {
                "id": "t1",
                "question": "How do I restart the database?",
                "expected_source": "runbooks/database_restart.md",
                "reference_answer": "safe restart procedures",
                "category": "answerable",
            },
            {
                "id": "t2",
                "question": "How do I bake a chocolate cake?",
                "expected_source": None,
                "reference_answer": None,
                "category": "unanswerable",
            },
        ],
    }
    dataset_path = tmp_path / "eval.json"
    with open(dataset_path, "w") as f:
        json.dump(tiny_dataset, f)

    summary = run_evaluation(str(dataset_path), str(index_dir))

    assert summary["total_items"] == 2
    assert summary["n_answerable"] == 1
    assert summary["n_unanswerable"] == 1
    assert summary["retrieval_metrics"]["hit_at_k"] == 1.0
    assert summary["refusal_metrics"]["correct_refusal_rate"] == 1.0
    assert len(summary["items"]) == 2
