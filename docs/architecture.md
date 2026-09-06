# OpsAssist Architecture

OpsAssist is an enterprise support assistant that answers questions about infrastructure assets, service ownership, incidents, and operational telemetry, using CMDB, incident, runbook, and observability data. This document describes its intended architecture and which pieces currently exist versus are planned for future work.

## Use Case

A support engineer or on-call responder asks a natural-language question ("Who owns payment-gateway?", "What's the standard restart procedure for the database?", "How many SEV1 incidents has order-service had this month?") and OpsAssist answers using only real, retrievable enterprise data -- CMDB records, incident history, and runbook documentation -- with citations, and an explicit refusal when it doesn't have enough evidence to answer confidently.

## Request Flow

\`\`\`
Client
  |
  v
FastAPI (src/opsassist/api/)
  |
  v
Orchestrator (src/opsassist/agents/)          <-- routes a request to the right
  |                                                capability: pure retrieval,
  |                                                a tool call, or both
  +--> Retrieval (src/opsassist/rag/)         <-- embeds the query, searches
  |                                                the vector index, returns
  |                                                cited chunks
  |
  +--> Tools / Connectors (src/opsassist/tools/,  <-- structured lookups against
       src/opsassist/connectors/)                  CMDB/incident data (e.g.
                                                     "list open SEV1s for X")
  |
  v
LLM (provider-agnostic, configs/opsassist.dev.yaml selects openai/azure/aws)
  |
  v
Guardrails (src/opsassist/guardrails/)        <-- enforces citation presence,
  |                                                response length limits,
  |                                                blocks ungrounded answers
  v
Response (with citations, model, latency_ms)
\`\`\`

**Tracing and evaluation span every call in this chain** (src/opsassist/observability/, src/opsassist/evals/) -- every request should be traceable end-to-end (which retrieval results were used, which tool calls were made, what the guardrail layer decided), and the same request/response shape used in production is what the offline evaluation harness (\`evals/run_rag_eval.py\`) replays against a fixed test set.

## What's Already Implemented (Days 3-5)

- **Retrieval** (\`src/opsassist/rag/loader.py\`, \`chunker.py\`, \`indexer.py\`, \`retriever.py\`) -- document ingestion, heading-aware chunking, embedding, and top-k similarity search over a FAISS index.
- **The LLM + Guardrails step, in placeholder form** (\`src/opsassist/api/main.py\`) -- \`ExtractiveAnswerGenerator\` stands in for a real LLM call, and the retrieval-confidence threshold check acts as an early, simple guardrail (refuse rather than answer when evidence is weak). Both are built behind interfaces (\`AnswerGenerator\`, \`EmbeddingProvider\`) so a real LLM provider and a dedicated guardrails module can be substituted later without changing the surrounding pipeline.
- **Evaluation** (\`src/opsassist/evals/run_rag_eval.py\`) -- offline retrieval and answer-quality metrics against a fixed question set, with documented quality gates (\`docs/evaluation_policy.md\`).
- **Data sources** (\`scripts/generate_cmdb.py\`, \`generate_incidents.py\`) -- synthetic CMDB assets and incidents with a genuine foreign-key relationship, plus generated knowledge docs.

## What's Planned (Future Days)

- **Orchestrator / Agents** (\`src/opsassist/agents/\`) -- currently an empty skeleton. Will route between pure-retrieval questions and questions that need a structured tool call (e.g. "how many open incidents does X have" is a CMDB/incident query, not a document-retrieval question).
- **Tools / Connectors** (\`src/opsassist/tools/\`, \`src/opsassist/connectors/\`) -- structured query functions against \`data/cmdb.csv\` / \`data/incidents.csv\` (or a real database in production), exposed as callable tools the orchestrator can invoke.
- **Dedicated Guardrails module** (\`src/opsassist/guardrails/\`) -- today's threshold check in \`api/main.py\` is a simple, inline version of this. A dedicated module will centralize citation enforcement, response-length limits, and content policy checks so they apply uniformly regardless of which code path produced an answer.
- **Observability** (\`src/opsassist/observability/\`) -- structured tracing across the full request chain (retrieval hits, tool calls, guardrail decisions, latency per stage), not just the current single \`latency_ms\` field.
- **Infra** (\`infra/\`) -- deployment configuration (containerization, IaC) for running OpsAssist outside a local dev environment.

## Provider-Agnostic LLM Configuration

\`configs/opsassist.dev.yaml\` selects the LLM backend by name (\`openai\`, \`azure\`, or \`aws\`) and model identifier -- no provider-specific code path is hardcoded elsewhere. Credentials for whichever provider is active are read from environment variables only (see \`.env.example\`); they are never stored in a config file or committed to the repository.

## Data & Naming Constraints

No production data, real infrastructure names, internal URLs, or company-specific identifiers appear anywhere in this codebase. All CMDB assets, incidents, applications, and knowledge documents are synthetically generated (see \`docs/data_dictionary.md\`).