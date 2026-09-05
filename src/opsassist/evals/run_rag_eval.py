"""
Offline evaluation harness for the RAG pipeline.

Runs every question in an eval dataset through the real retriever and
answer pipeline (the same code path as the /v1/chat API), and reports:

- Retrieval metrics: hit@k and mean reciprocal rank (MRR), computed
  only over "answerable" eval items (ones with a known expected source
  document).
- Answer quality metrics: groundedness, relevance, and completeness,
  via a pluggable AnswerJudge interface. The default HeuristicAnswerJudge
  is fully offline; a real deployment could implement the same
  interface with an LLM-as-judge call instead.
- Correct-refusal rate: for "unanswerable" eval items, whether the
  system correctly declined to answer rather than fabricating something.

Usage:
    python -m opsassist.evals.run_rag_eval
"""

import json
import os
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timezone

from opsassist.api.main import (
    RETRIEVAL_SCORE_THRESHOLD,
    build_prompt,
    load_runtime,
)
from opsassist.rag.retriever import RetrievedChunk

EVAL_DATASET_PATH = "evals/datasets/rag_eval_v1.json"
REPORT_PATH = "reports/rag_eval_v1.json"
INDEX_DIR = "data/rag_index"
RETRIEVAL_K = 5

_WORD_PATTERN = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> set[str]:
    return set(_WORD_PATTERN.findall(text.lower()))


# --------------------------------------------------------------------
# Retrieval metrics
# --------------------------------------------------------------------
def hit_at_k(retrieved: list[RetrievedChunk], expected_source: str) -> bool:
    """True if the expected source document appears anywhere in the retrieved chunks."""
    return any(c.source_path == expected_source for c in retrieved)


def reciprocal_rank(retrieved: list[RetrievedChunk], expected_source: str) -> float:
    """1 / rank of the first chunk matching the expected source, else 0."""
    for rank, chunk in enumerate(retrieved, start=1):
        if chunk.source_path == expected_source:
            return 1.0 / rank
    return 0.0


# --------------------------------------------------------------------
# Answer quality judge abstraction
# --------------------------------------------------------------------
@dataclass
class JudgeScore:
    groundedness: (
        float  # 0-1: is the answer's cited source correct / is it not fabricating
    )
    relevance: float  # 0-1: token overlap between answer and reference answer
    completeness: (
        float  # 0-1: fraction of reference answer's key terms present in the answer
    )


class AnswerJudge(ABC):
    @abstractmethod
    def judge(
        self,
        question: str,
        answer: str,
        reference_answer: str | None,
        expected_source: str | None,
        citations: list[RetrievedChunk],
    ) -> JudgeScore:
        raise NotImplementedError


class HeuristicAnswerJudge(AnswerJudge):
    """
    Fully offline judge using simple token-overlap heuristics rather
    than a real LLM call. Good enough to catch regressions in an
    automated pipeline; a production evaluation would likely also run
    an LLM-as-judge (e.g. asking Claude to rate groundedness and
    completeness) behind this same interface for a second opinion.
    """

    def judge(
        self,
        question: str,
        answer: str,
        reference_answer: str | None,
        expected_source: str | None,
        citations: list[RetrievedChunk],
    ) -> JudgeScore:
        # Groundedness: for answerable items, did the citations include
        # the expected source? For unanswerable items, groundedness is
        # about *not fabricating* -- scored 1.0 if there are no citations
        # (i.e. the system correctly abstained) and 0.0 if it invented
        # citations for a question it shouldn't have answered.
        if expected_source is None:
            groundedness = 1.0 if not citations else 0.0
        else:
            groundedness = (
                1.0 if any(c.source_path == expected_source for c in citations) else 0.0
            )

        if reference_answer is None:
            # No reference to compare against (unanswerable item) --
            # relevance/completeness aren't meaningful concepts here.
            return JudgeScore(
                groundedness=groundedness, relevance=0.0, completeness=0.0
            )

        answer_tokens = _tokenize(answer)
        reference_tokens = _tokenize(reference_answer)

        if not reference_tokens:
            relevance = 0.0
            completeness = 0.0
        else:
            overlap = answer_tokens & reference_tokens
            # Relevance: Jaccard-style overlap between answer and reference.
            union = answer_tokens | reference_tokens
            relevance = len(overlap) / len(union) if union else 0.0
            # Completeness: recall -- how much of the reference's content
            # made it into the answer.
            completeness = len(overlap) / len(reference_tokens)

        return JudgeScore(
            groundedness=groundedness, relevance=relevance, completeness=completeness
        )


# --------------------------------------------------------------------
# Evaluation runner
# --------------------------------------------------------------------
def run_evaluation(
    eval_dataset_path: str = EVAL_DATASET_PATH, index_dir: str = INDEX_DIR
) -> dict:
    with open(eval_dataset_path, "r") as f:
        dataset = json.load(f)

    runtime = load_runtime(index_dir)
    judge = HeuristicAnswerJudge()

    per_item_results = []
    hits, reciprocal_ranks = [], []
    groundedness_scores, relevance_scores, completeness_scores = [], [], []
    correct_refusals, incorrect_refusals = 0, 0

    for item in dataset["items"]:
        question = item["question"]
        expected_source = item["expected_source"]
        reference_answer = item["reference_answer"]
        category = item["category"]

        retrieved = runtime.retriever.search(question, k=RETRIEVAL_K)

        if not retrieved or retrieved[0].score < RETRIEVAL_SCORE_THRESHOLD:
            answer = "The available sources do not support a confident answer to this question."
            citations: list[RetrievedChunk] = []
        else:
            prompt = build_prompt(runtime.prompt_template, question, retrieved)
            answer = runtime.generator.generate(prompt, retrieved)
            citations = [c for c in retrieved if c.score > 0.0]

        score = judge.judge(
            question, answer, reference_answer, expected_source, citations
        )

        result = {
            "id": item["id"],
            "question": question,
            "category": category,
            "expected_source": expected_source,
            "answered": bool(citations),
            "groundedness": score.groundedness,
            "relevance": score.relevance,
            "completeness": score.completeness,
        }

        if category == "answerable":
            h = hit_at_k(retrieved, expected_source)
            rr = reciprocal_rank(retrieved, expected_source)
            hits.append(h)
            reciprocal_ranks.append(rr)
            result["hit"] = h
            result["reciprocal_rank"] = rr
            groundedness_scores.append(score.groundedness)
            relevance_scores.append(score.relevance)
            completeness_scores.append(score.completeness)
        else:  # unanswerable
            correctly_refused = not citations
            if correctly_refused:
                correct_refusals += 1
            else:
                incorrect_refusals += 1
            result["correctly_refused"] = correctly_refused

        per_item_results.append(result)

    n_answerable = len(hits)
    n_unanswerable = correct_refusals + incorrect_refusals

    summary = {
        "version": dataset["version"],
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset_path": eval_dataset_path,
        "total_items": len(dataset["items"]),
        "n_answerable": n_answerable,
        "n_unanswerable": n_unanswerable,
        "retrieval_metrics": {
            "hit_at_k": sum(hits) / n_answerable if n_answerable else None,
            "mrr": sum(reciprocal_ranks) / n_answerable if n_answerable else None,
            "k": RETRIEVAL_K,
        },
        "answer_quality_metrics": {
            "mean_groundedness": (
                sum(groundedness_scores) / len(groundedness_scores)
                if groundedness_scores
                else None
            ),
            "mean_relevance": (
                sum(relevance_scores) / len(relevance_scores)
                if relevance_scores
                else None
            ),
            "mean_completeness": (
                sum(completeness_scores) / len(completeness_scores)
                if completeness_scores
                else None
            ),
        },
        "refusal_metrics": {
            "correct_refusal_rate": (
                correct_refusals / n_unanswerable if n_unanswerable else None
            ),
            "correct_refusals": correct_refusals,
            "incorrect_refusals": incorrect_refusals,
        },
        "items": per_item_results,
    }
    return summary


def main() -> None:
    summary = run_evaluation()

    os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)
    with open(REPORT_PATH, "w") as f:
        json.dump(summary, f, indent=2)

    print(
        f"Evaluated {summary['total_items']} questions "
        f"({summary['n_answerable']} answerable, {summary['n_unanswerable']} unanswerable)"
    )
    print(
        f"Retrieval  -> hit@{RETRIEVAL_K}: {summary['retrieval_metrics']['hit_at_k']:.3f}  "
        f"MRR: {summary['retrieval_metrics']['mrr']:.3f}"
    )
    print(
        f"Answer     -> groundedness: {summary['answer_quality_metrics']['mean_groundedness']:.3f}  "
        f"relevance: {summary['answer_quality_metrics']['mean_relevance']:.3f}  "
        f"completeness: {summary['answer_quality_metrics']['mean_completeness']:.3f}"
    )
    print(
        f"Refusal    -> correct refusal rate: {summary['refusal_metrics']['correct_refusal_rate']:.3f}"
    )
    print(f"Report written to {REPORT_PATH}")


if __name__ == "__main__":
    main()
