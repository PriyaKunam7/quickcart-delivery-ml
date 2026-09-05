# RAG Evaluation Policy & Quality Gates

Defines how retrieval and answer quality are measured for OpsAssist, and the minimum thresholds a change must meet before merge/deployment.

## Evaluation Dataset

`evals/datasets/rag_eval_v1.json` — 50 hand-written questions against the current `knowledge_base/` runbooks:
- **39 answerable questions** (3 per document section, across all 13 sections), each with an `expected_source` document and a short `reference_answer` drawn from that section's real text.
- **11 unanswerable questions**, deliberately unrelated to any runbook content (cooking, hobbies, entertainment), used to measure whether the system correctly refuses rather than fabricating an answer.

Running `python -m opsassist.evals.run_rag_eval` executes every question through the same retrieval + answer pipeline used by `POST /v1/chat`, and writes a versioned report to `reports/rag_eval_v1.json`.

## Metrics

### Retrieval metrics (answerable questions only)
- **hit@k** -- fraction of questions where the expected source document appears anywhere in the top-k retrieved chunks (k=5).
- **MRR (Mean Reciprocal Rank)** -- average of `1/rank` of the first correctly-sourced chunk across all questions; rewards the correct document appearing *early* in the results, not just somewhere in them.

### Answer quality metrics
Computed by a pluggable `AnswerJudge` interface. The current implementation, `HeuristicAnswerJudge`, is fully offline (no LLM call):
- **Groundedness** -- for answerable questions: 1.0 if the answer's citations include the expected source, else 0.0. For unanswerable questions: 1.0 if the system correctly returned zero citations (didn't fabricate), else 0.0. This is the most important metric -- it's the direct measurement of "does the system hallucinate."
- **Relevance** -- token-overlap (Jaccard similarity) between the generated answer and the reference answer.
- **Completeness** -- recall of the reference answer's key terms within the generated answer.

### Refusal metrics (unanswerable questions only)
- **Correct refusal rate** -- fraction of clearly out-of-scope questions where the system explicitly declined to answer instead of generating something.

A production evaluation would likely add an **LLM-as-judge** implementation of the same `AnswerJudge` interface (e.g. asking Claude to rate groundedness and completeness against the retrieved context) as a second signal alongside the heuristic judge -- the interface is designed so that swap doesn't require changing the evaluation runner.

## Measured Baseline (this dataset, current pipeline)

| Metric | Result |
|---|---|
| hit@5 | 0.949 |
| MRR | 0.919 |
| Mean groundedness | 0.923 |
| Mean relevance | 0.224 |
| Mean completeness | 0.644 |
| Correct refusal rate | 0.636 |

## Quality Gates (minimum to merge / deploy)

| Metric | Gate | Rationale |
|---|---|---|
| hit@5 | **>= 0.90** | Retrieval is the foundation everything else depends on; the measured 0.949 has healthy margin above this. |
| MRR | **>= 0.85** | Not just "found somewhere" but "found near the top" -- measured 0.919 clears this comfortably. |
| Mean groundedness | **>= 0.90** | The most safety-critical metric: this is the direct measurement of hallucination/fabrication. Measured 0.923 clears the gate, but any regression below 0.90 should block a merge outright. |
| Mean completeness | **>= 0.55** | Directional quality signal for how much of the expected content actually surfaces in the answer. |
| Correct refusal rate | **>= 0.60** *(known baseline limit -- see below)* | Set at the current measured floor (0.636), not an aspirational target, pending the fix described below. |
| Mean relevance | *(monitored, not gated)* | Naturally low under strict token-overlap scoring across varied phrasings; tracked for regressions rather than gated on an absolute threshold. |

## Known Limitation: Refusal Rate Ceiling

The eval run surfaced a real, reproducible weakness: **4 of the 11 unanswerable questions incorrectly cleared the confidence threshold** (`q042`, `q045`, `q048`, `q049` in the current dataset -- questions about a birthday party, houseplants, learning guitar, and movie showtimes). Their retrieval scores (0.155-0.218) are pure hash-collision noise from `LocalHashEmbeddingProvider` -- these questions share no real vocabulary with the runbook corpus, but a handful of unrelated tokens happened to hash into overlapping buckets.

**This cannot be fixed by simply raising `RETRIEVAL_SCORE_THRESHOLD`.** One of Day 4's legitimately answerable questions ("What are the severity levels for incidents?") scores 0.192 -- inside the same range as these false positives. There is no single threshold value that cleanly separates genuine matches from collision noise with the current embedding approach; the score distributions overlap.

**Root cause:** `LocalHashEmbeddingProvider` is a deliberate offline placeholder (see `docs/rag_pipeline.md`) -- a bag-of-words hashing trick with no real semantic understanding. This is exactly the kind of gap a placeholder is expected to have.

**Action item, not a workaround:** replacing the embedding provider with a real semantic model (e.g. a hosted embeddings API or a local sentence-transformers model) is expected to resolve this class of false positive, since genuine semantic embeddings don't rely on incidental token-hash collisions. The refusal-rate gate is intentionally set at the current measured baseline (0.60) rather than a higher aspirational number, so that this is tracked honestly as a known gap rather than hidden behind an artificially lenient gate -- and the gate should be raised as soon as the embedding provider is upgraded.

## Process

1. Any change to the retriever, embedding provider, prompt template, or answer generator must be followed by `python -m opsassist.evals.run_rag_eval` before merge.
2. If any gated metric falls below its threshold, the PR should not merge until investigated -- either the regression is fixed, or the gate itself is revisited with an explicit, documented rationale (as done above for refusal rate).
3. `reports/rag_eval_v1.json` (or the current dataset version's report) should be attached as evidence in the PR, alongside the eval dataset version used.
4. When the knowledge base or eval dataset changes meaningfully, bump the dataset version (`rag_eval_v2.json`, etc.) rather than silently mutating `v1` -- this keeps historical evaluation results comparable across versions.