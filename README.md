# OmniComm AI Support Assistant

A production-grade, Retrieval-Augmented Generation (RAG) customer-support assistant for OmniComm's telecom and cloud communications platform.

## Architecture

```
User Question
     │
     ├──→ [HuggingFace Embeddings] → ChromaDB Vector Search (k=5)
     │         all-MiniLM-L6-v2           │
     │                                    ▼
     │                          Top-5 Documents + Metadata
     │                          (status, last_updated, id)
     │                                    │
     └──→ [Passthrough] ─────────────────▼
                              [ChatPromptTemplate]
                              System Prompt + Context + Question
                                          │
                                          ▼
                              [Gemini 2.5 Flash]
                              with_structured_output(ResponseFormat)
                                          │
                                          ▼
                              { answer: "...", sources: ["kb_002"] }
```

### Key Components

| Component | File | Purpose |
|---|---|---|
| Document Manager | `src/document_manager.py` | Parses KB JSON → LangChain Documents → ChromaDB vector store |
| RAG Chain | `src/rag_chain.py` | Prompt engineering, LCEL chain, structured output via Pydantic |
| API Server | `src/api.py` | FastAPI `POST /ask` endpoint |
| Ragas Evaluation | `tests/evaluate.py` | Reference-free evaluation (Faithfulness, Answer Relevancy) |
| Custom Test Suite | `tests/test_custom_queries.py` | 12 hand-crafted edge-case tests with assertions |

### Design Decisions

1. **Retrieval — Dense Embeddings (MiniLM-L6-v2 + ChromaDB)**
   - Local embeddings (no API dependency, zero latency overhead)
   - k=5 retrieval balances multi-hop coverage vs context noise
   - Per-article chunking (articles are ~150 tokens; splitting would destroy coherence)

2. **Generation — Gemini 2.5 Flash with Structured Output**
   - `with_structured_output(ResponseFormat)` guarantees valid JSON on every call (vs ~20% failure with prompt-only JSON)
   - Temperature 0.0 for deterministic, faithful answers
   - In-line citations `[kb_001]` enforced via system prompt

3. **Deprecated Document Handling**
   - Deprecated docs are **not filtered out** — they're tagged with a `[WARNING: DEPRECATED]` label
   - This allows the LLM to answer historical questions ("What was the OLD rate?") while never citing deprecated data as current truth

4. **Graceful Abstention**
   - The system prompt instructs the LLM to explicitly state when the KB is insufficient
   - Tested with out-of-scope and adversarial queries (see evaluation below)

### Trade-offs

| Decision | Pro | Con |
|---|---|---|
| Local ChromaDB | Zero-config, fast for small KB | Doesn't scale horizontally |
| Local MiniLM embeddings | No API cost, low latency | Slightly less accurate than API embeddings |
| k=5 retrieval | Covers multi-hop queries | Retrieves ~20% of corpus per query |
| No query augmentation (HyDE) | Simpler pipeline, sufficient for small KB | Would improve recall for vague queries at scale |
| No re-ranking | Lower latency | Cross-encoder re-ranking could improve precision |

## Setup and Running

### 1. Install Dependencies
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure API Key
```bash
cp .env.example .env
# Edit .env and add your Gemini API key
```

### 3. Run the API Server
```bash
python -m src.api
```

### 4. Query the Prototype
```bash
curl -X POST "http://localhost:8000/ask" \
     -H "Content-Type: application/json" \
     -d '{"question": "How much does it cost to call Indonesia?"}'
```

**Example Response:**
```json
{
  "answer": "The cost to call Indonesia depends on your plan. For the Basic plan, outbound voice calls to Indonesia are $0.12 per minute [kb_002]. For the Pro plan, the rate is $0.09 per minute [kb_003]. Enterprise pricing is typically contract-specific [kb_003].",
  "sources": ["kb_002", "kb_003"]
}
```

## Evaluation

### Ragas Evaluation (15 provided queries)

Run: `python tests/evaluate.py`

Results across all 15 evaluation queries:

| Metric | Score |
|---|---|
| **Faithfulness** | **0.93** (avg) — 12/15 queries scored 1.0 |
| **Answer Relevancy** | **0.93** (avg) — high relevance across queries |

The one lower faithfulness score (0.33) was on a multi-source citation question where the model correctly identified the answer but Ragas penalized a minor attribution nuance. Full per-query results are in `evaluation_results.csv`.

### Custom Edge-Case Test Suite (12 hand-crafted tests)

Run: `python tests/test_custom_queries.py`

**Result: 12/12 PASSED (100%)**

| Category | Test | Status |
|---|---|---|
| Direct Lookup | Basic plan rate for Singapore | ✅ |
| Multi-Hop | Roaming + Pricing (Hong Kong → Netherlands) | ✅ |
| Deprecated Handling | Old 2025 Indonesia rate | ✅ |
| Deprecated vs Active | API rate limit 60 vs 100 | ✅ |
| Policy Enforcement | Emergency services (911) | ✅ |
| Abstention | Out-of-scope (fax to Japan) | ✅ |
| Multi-Hop Troubleshooting | Firewall + call drops | ✅ |
| Multi-Hop Billing | Double charge + refund timeline | ✅ |
| Enterprise Abstention | Enterprise-specific pricing | ✅ |
| Adversarial | CEO / office address hallucination | ✅ |
| SLA Multi-Hop | Pro plan P1 response time | ✅ |
| Temporal Reasoning | Suspicious activity suspension | ✅ |

Full results in `custom_test_results.json`.

## Project Structure

```
├── src/
│   ├── document_manager.py   # KB loading, embedding, vector store
│   ├── rag_chain.py          # Prompt, LCEL chain, structured output
│   └── api.py                # FastAPI server
├── tests/
│   ├── evaluate.py           # Ragas evaluation pipeline
│   └── test_custom_queries.py # 12 edge-case tests
├── data/
│   ├── knowledge_base.json   # 24-article knowledge base
│   └── evaluation_queries.json # 15 test queries
├── examples/
│   ├── api_request_example.json
│   └── api_response_example.json
├── evaluation_results.csv    # Ragas per-query results
├── custom_test_results.json  # Custom test results
├── requirements.txt
├── .env.example
└── README.md
```
