# Adsparkx AI — Persona-Adaptive Customer Support Agent

An AI support agent that detects the customer's persona, retrieves grounded
answers from a knowledge base via RAG, adapts tone/style per persona, and
escalates to a human agent (with a structured handoff summary) when needed.

Built for the Adsparkx AI Assignment — Persona-Aware Customer Support Agent
using LLMs, RAG, and Human Escalation.

## 1. Project Overview

The agent acts as tier-1 support for a fictional SaaS product, **CloudSuite**.
On every message it:

1. Detects whether the user is a **Technical Expert**, **Frustrated User**,
   or **Business Executive**.
2. Retrieves the most relevant chunks from a 12-document knowledge base
   (password resets, API errors, billing, SLAs, security, SSO, etc.) using a
   FAISS vector index.
3. Generates a response **grounded only in retrieved content**, styled to
   match the detected persona.
4. Checks configurable escalation rules (low retrieval confidence, sensitive
   keywords, repeated frustration, long unresolved conversations) and, if
   triggered, escalates with a structured JSON handoff summary for a human
   agent.

It runs as a **FastAPI** backend, with an interactive **CLI** chatbot and a
bonus **Streamlit** chat UI, all calling the same orchestration pipeline in
`app/agent.py`.

A deliberate design choice: **the whole pipeline runs with zero API keys**.
If no `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` is set, retrieval falls back to
a local TF-IDF embedding space and response generation falls back to an
extractive responder that builds its answer directly from retrieved KB text
(see "RAG Pipeline Design" and "Known Limitations" below for the tradeoffs).
This made the system far easier to test and demo end-to-end without
depending on external network access.

## 2. Tech Stack

| Component | Choice | Version |
|---|---|---|
| Language | Python | 3.11+ |
| API framework | FastAPI + Uvicorn | 0.115 / 0.30 |
| Agent orchestration | Custom pipeline (`app/agent.py`) | — |
| LLM | OpenAI (`gpt-4o-mini`) or Anthropic (`claude-sonnet-4-6`) or Ollama, pluggable | — |
| Embeddings | OpenAI `text-embedding-3-small` (if key present) **or** local TF-IDF fallback (`scikit-learn`) | 1.51 / 1.5.2 |
| Vector database | FAISS (`faiss-cpu`, `IndexFlatIP` on normalized vectors = cosine similarity) | 1.8.0 |
| PDF parsing | `pypdf` | 4.3.1 |
| DOCX parsing | `python-docx` | 1.1.2 |
| UI | CLI (`cli.py`) + bonus Streamlit app (`streamlit_app.py`) | streamlit 1.38 |
| Testing | `pytest` | 8.3.3 |

## 3. Architecture

```
                ┌─────────────────┐
   User Query → │ Persona Detection│  (rule-based keyword scoring +
                └────────┬─────────┘   punctuation heuristics, LLM tie-break)
                         │
                         ▼
                ┌─────────────────┐
                │   Retrieval (RAG)│  (chunk → embed → FAISS top-k search)
                └────────┬─────────┘
                         │  retrieved chunks + similarity scores
                         ▼
                ┌─────────────────┐
                │Response Generation│ (persona-specific prompt, grounded
                └────────┬─────────┘   only in retrieved content)
                         │
                         ▼
                ┌─────────────────┐
                │ Escalation Check │ (sensitive keywords, low confidence,
                └────────┬─────────┘   repeated frustration, long convo)
                         │
              escalate?  │
            ┌────yes─────┴─────no────┐
            ▼                        ▼
   ┌─────────────────┐      Return response to user
   │  Human Handoff    │
   │  Summary (JSON)   │
   └─────────────────┘
```

This maps directly onto `app/agent.py::run_turn`, which every interface
(FastAPI, CLI, Streamlit) calls.

### Module map

```
app/
  config.py      persona keywords + escalation thresholds (all tunable here)
  persona.py     persona detection (rules, with optional LLM tie-break)
  rag.py         document loading, chunking, embeddings, FAISS vector store
  llm.py         persona-specific prompts + multi-provider LLM calls
  escalation.py  escalation rule checks
  session.py     in-memory conversation/session state
  handoff.py     structured human handoff summary builder
  agent.py       orchestrates the full pipeline (the "brain")
  main.py        FastAPI app (/chat, /session/{id}, /health)
cli.py           interactive CLI chatbot
streamlit_app.py bonus chat UI
data/            12 knowledge base documents (11 markdown + 1 PDF)
tests/           pytest unit tests for persona + escalation logic
```

## 4. Persona Detection Strategy

**Classification method:** rule-based keyword/phrase scoring (see
`app/config.py::PERSONA_KEYWORDS`), not a black-box classifier — this keeps
the decision auditable and free of API calls in the common case.

- Each persona has a curated list of signal phrases (e.g. Technical Expert:
  `api`, `stack trace`, `oauth`, `root cause`; Frustrated User: `nothing
  works`, `fed up`, `asap`; Business Executive: `business impact`, `sla`,
  `timeline`).
- The **Frustrated User** score also gets heuristic bonuses for repeated
  `!` and ALL-CAPS runs, since emotional tone is often punctuation-driven.
- If the top two persona scores tie (including a 0-0-0 tie, i.e. no
  keywords matched at all), the agent falls back to an **LLM classification
  call** (`app/llm.py::classify_persona_llm`) as a tie-breaker — this only
  fires for ambiguous messages, keeping cost and latency low in the common
  case.
- **Prompt design** for the LLM tie-breaker is a single constrained
  classification prompt: "Reply with only the persona name, nothing else."

**Rules used:** see `PERSONA_KEYWORDS` and `FRUSTRATION_*` constants in
`app/config.py` — fully configurable without touching code logic.

## 5. RAG Pipeline Design

**Chunking strategy:** Markdown documents are first split on `## ` headings
so each chunk inherits a meaningful section name; PDFs are split per page;
each resulting block is then word-chunked (180 words, 30-word overlap) to
keep chunks small enough for precise retrieval while preserving local
context (`app/rag.py::_chunk_text`). Every chunk carries `source` (filename)
and `section` (heading or page label) metadata, which is surfaced to the
user and included in the handoff summary.

**Embedding model:**
- **Primary:** OpenAI `text-embedding-3-small`, used automatically when
  `OPENAI_API_KEY` is set.
- **Fallback:** a local TF-IDF vectorizer (`scikit-learn`), fit once over
  the full chunk corpus. This was chosen over downloading a sentence-
  transformers model because the assignment needed to run in a sandboxed
  environment without access to model-hosting domains — TF-IDF needs no
  network access at all and still gives reasonable lexical retrieval over a
  small, well-curated KB like this one. Both backends implement the same
  `embed_texts()` interface so swapping is a one-line change
  (`app/rag.py::get_embedding_backend`).

**Vector database:** FAISS `IndexFlatIP` over L2-normalized vectors (inner
product on normalized vectors = cosine similarity). Chosen for zero
external infra (pure in-process index) appropriate for a ~150-chunk corpus;
swapping to ChromaDB/Qdrant/Pinecone would only require changing
`VectorStore.build`/`search` in `app/rag.py`.

**Retrieval strategy:** top-`k=4` nearest chunks per query
(`app/agent.py::TOP_K`). The top similarity score is also reused directly as
the **escalation confidence signal** (see below) so retrieval and
escalation share one source of truth instead of two separate heuristics.

## 6. Adaptive Response Generation

Persona-specific system instructions live in `app/llm.py::PERSONA_PROMPTS`:

- **Technical Expert** → root cause + numbered troubleshooting steps +
  technical detail, no filler/apologies.
- **Frustrated User** → one-sentence empathetic acknowledgement, then
  simple, action-oriented steps, no jargon.
- **Business Executive** → concise, impact/timeline-first, jargon-free.

Every prompt includes a hard grounding instruction: *"Only use facts
present in the provided CONTEXT... do not speculate."* When no LLM key is
configured, `app/llm.py::_extractive_fallback` builds the answer directly
from the retrieved chunk text (bulleted, persona-styled header) — this is
hallucination-proof by construction since no generation model is involved.

## 7. Escalation Logic

Defined in `app/escalation.py`, configured in `app/config.py::ESCALATION_CONFIG`.
Checks run in this order, first match wins:

| Trigger | Threshold (configurable) |
|---|---|
| Sensitive keyword (refund, legal, GDPR, breach, hacked, cancel account...) | exact list in `ESCALATION_CONFIG["sensitive_keywords"]` |
| No relevant document found | top similarity score < `min_relevance_score` (0.18) |
| Low retrieval confidence | top similarity score < `low_confidence_score` (0.30) |
| Repeated frustration | `Frustrated User` persona for `frustration_escalation_turns` (2) consecutive turns |
| Long unresolved conversation | turn count ≥ `max_unresolved_turns` (4) |

On escalation, `app/handoff.py::build_handoff_summary` produces:

```json
{
  "persona": "Frustrated User",
  "issue": "Unable to reset password",
  "conversation_history": "User: ...\nAgent: ...",
  "documents_used": ["password_reset_guide.md"],
  "attempted_steps": ["Provided guidance from: password_reset_guide.md"],
  "escalation_reason": "No sufficiently relevant information found (top score 0.13).",
  "recommendation": "Investigate manually and consider adding a new KB article if recurring."
}
```

## 8. Setup Instructions

```bash
# 1. Clone and enter the repo
git clone <your-repo-url>
cd adsparkx-agent

# 2. Create a virtual environment
python3 -m venv .venv
source .venv/bin/activate     # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment variables (optional — works with none set)
cp .env.example .env
# edit .env and add OPENAI_API_KEY or ANTHROPIC_API_KEY if you want a real LLM

# 5a. Run the CLI chatbot
python cli.py

# 5b. OR run the FastAPI server
uvicorn app.main:app --reload
# then POST to http://127.0.0.1:8000/chat  {"message": "..."}
# interactive docs at http://127.0.0.1:8000/docs

# 5c. OR run the bonus Streamlit UI
streamlit run streamlit_app.py
```

Run the test suite:
```bash
pytest -q
```

## 9. Environment Variables

| Variable | Required? | Purpose |
|---|---|---|
| `LLM_PROVIDER` | No (default `auto`) | `auto`\|`openai`\|`anthropic`\|`ollama`\|`none` |
| `OPENAI_API_KEY` | No | Enables OpenAI chat + embeddings |
| `OPENAI_MODEL` | No | Default `gpt-4o-mini` |
| `ANTHROPIC_API_KEY` | No | Enables Claude responses |
| `ANTHROPIC_MODEL` | No | Default `claude-sonnet-4-6` |
| `OLLAMA_HOST` / `OLLAMA_MODEL` | No | For local LLM via Ollama |

**No key is required** — see Section 1 / Section 11 for what changes when
none are set.

## 10. Example Queries

```
1. "Can you explain the API authentication failure and provide error details?"
   → Technical Expert persona, retrieves api_authentication_errors.md

2. "I've tried everything and nothing works! My password reset is broken!!!"
   → Frustrated User persona, retrieves password_reset_guide.md

3. "How does this outage impact operations and when will it be resolved?"
   → Business Executive persona, retrieves sla_uptime_policy.md

4. "I want a refund, this is a billing dispute"
   → escalates immediately (sensitive keyword: "refund")

5. "Someone got into my account and changed my email, I think I've been hacked"
   → escalates immediately (sensitive keyword: "hacked";
     account_security_suspicious_activity.md is still surfaced as context)
```

## 11. Known Limitations & Future Improvements

- **TF-IDF fallback is lexical, not semantic.** Without an OpenAI key,
  retrieval can miss paraphrased queries that don't share vocabulary with
  the KB. Swapping in a local `sentence-transformers` model (when network
  access to a model hub is available) would close this gap with minimal
  code change (`EmbeddingBackend` interface already supports it).
- **Extractive fallback responses are bullet-style, not fluent prose.**
  This is intentional (zero hallucination risk without an LLM key) but is
  noticeably less polished than the LLM-backed path.
- **In-memory session store** — conversation history is lost on process
  restart. Swapping in Redis/Postgres (`app/session.py`) would add
  durability for production use.
- **No multi-turn memory passed into the LLM prompt** beyond persona
  history; the response generator only sees the current message + KB
  context. Adding rolling conversation summary into the prompt is a
  natural next step.
- **Single embedding fit per process.** The TF-IDF vectorizer is fit once
  at startup over the static KB; it doesn't support incremental document
  ingestion without a rebuild (`get_vector_store` caches a singleton).
- **Persona detection is keyword-based by design** for transparency and
  cost, with an LLM tie-break only on ambiguous input — a fully ML-based
  classifier (e.g. fine-tuned on labeled support tickets) would generalize
  better to oblique phrasing.
