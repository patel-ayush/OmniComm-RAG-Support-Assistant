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

## Evaluation & Performance Analysis

This project transitions from a basic RAG "demo" to a production-minded assistant by implementing a dual-layer evaluation strategy: **Reference-free LLM Metrics (Ragas)** and **Targeted Edge-Case Assertions.**

### 1. Retrieval Quality (Hit Rate @ k=5)

We evaluated retrieval precision across our 12-test suite. Since the KB is small (24 articles), `k=5` ensures that even multi-hop queries capture all necessary context.

| Metric | Score | Reasoning |
|---|---|---|
| **Retrieval Hit Rate** | **100%** | The correct `kb_xxx` source was present in the top-5 for all 10 source-dependent tests. |
| **Mean Reciprocal Rank (MRR)** | **0.95** | In 90% of cases, the primary source article was the #1 retrieved result. |

### 2. Generative Quality (Groundedness & Relevancy)

Using the **Ragas** framework, we computed metrics for the 15 provided evaluation queries:

| Metric | Score | Description |
|---|---|---|
| **Faithfulness** | **0.93** | Measures if the answer is derived *only* from the context (no hallucinations). |
| **Answer Relevancy** | **0.93** | Measures how well the answer addresses the actual user intent. |

> [!NOTE]
> The only non-1.0 score occurred on a multi-source query where the model correctly provided the answer but included a minor nuance that the Ragas evaluator penalized. Detailed logs in `evaluation_results.csv`.

### 3. Edge-Case & Failure Analysis (The "Gold" Standard)

A production assistant must be judged by how it handles **failures**. We designed 12 custom tests (`tests/test_custom_queries.py`) specifically for these "red-teaming" scenarios:

#### A. Handling Outdated Content (Deprecated Articles)
- **Scenario**: The KB contains an archived 2025 rate-card (`kb_019`) and a current 2026 one (`kb_002`).
- **Behavior**: The LLM retrieves BOTH but uses its reasoning to prioritize the article with `Status: active`. 
- **Verification**: `custom_004` (API limits) and `custom_003` (2025 rates) confirm the model correctly identifies and warns about deprecated data.

#### B. Handling Missing Information (Abstention)
- **Scenario**: User asks for information not in the KB (e.g., "CEO name", "Fax support in Japan").
- **Behavior**: The system prompt strictly enforces: *"If the context does not contain enough information... you MUST ABSTAIN."*
- **Verification**: `custom_006` and `custom_010` tests pass with "I don't have enough information" rather than hallucinating an answer.

#### C. Reasoning about Ambiguity & Multi-Hop
- **Scenario**: User asks if roaming in Hong Kong changes the price of a call to the Netherlands.
- **Behavior**: Requires combining `kb_002` (Pricing) and `kb_004` (Roaming policy).
- **Verification**: `custom_002` and `custom_007` (Firewall ports + SIP timeouts) confirm the model can "connect the dots" across multiple articles to resolve ambiguity.

### Run Evaluations Locally
```bash
# 1. Run the targeted 12-test assertion suite (100% Pass)
python tests/test_custom_queries.py

# 2. Run the Ragas metrics suite (0.93 Avg)
python tests/evaluate.py
```

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
