# AcityBot — RAG Chatbot for Academic City University

> A fully manual RAG (Retrieval-Augmented Generation) system.  


---

## Quick Start

```bash
# 1. Clone / set up
cd rag_chatbot
pip install -r requirements.txt

# 2. Set your Anthropic API key
export ANTHROPIC_API_KEY="sk-ant-..."

# 3. Launch the Streamlit UI
streamlit run app.py
```

The first run downloads the datasets (~60MB PDF + small CSV), builds embeddings, and saves the FAISS index. Subsequent runs load from cache and start in ~5 seconds.

---

## Project Structure

```
rag_chatbot/
├── app.py                        # Streamlit UI (5 tabs)
├── requirements.txt
├── rag/
│   ├── __init__.py
│   ├── data_loader.py            # Part A: CSV + PDF ingestion & cleaning
│   ├── chunker.py                # Part A: 3 chunking strategies
│   ├── embedder.py               # Part B: sentence-transformer pipeline
│   ├── vector_store.py           # Part B: FAISS IndexFlatIP
│   ├── retriever.py              # Part B: top-k, hybrid, query expansion
│   ├── prompt_builder.py         # Part C: V1/V2/V3 prompt templates
│   ├── pipeline.py               # Part D: full pipeline + Part G innovation
│   └── logger.py                 # Part D: structured experiment logging
├── data/                         # Downloaded datasets + cached embeddings
├── experiment_logs/
│   ├── MANUAL_EXPERIMENT_LOG.md  # Hand-written experiment records
│   ├── session_*.json            # Auto-generated per-session logs
│   └── full_history.jsonl        # Append-only run history
└── README.md
```

---

## Architecture Overview

```
User Query
    │
    ▼
[Query Expansion]  — Domain synonym table (NPP, NDC, GDP, cedi, etc.)
    │
    ▼
[Dense Retrieval]  — FAISS IndexFlatIP, cosine similarity
    │         ┐
[BM25 Retrieval]   — rank_bm25, keyword exact match
    │         ┘
    ▼
[Hybrid Fusion]    — Reciprocal Rank Fusion (α=0.7 dense / 0.3 BM25)
    │
    ▼
[Failure Detection]— Low confidence flag if top score < 0.25
    │
    ▼
[Feedback Adjustment]— Per-chunk score delta from user ratings
    │
    ▼
[Context Window Mgmt]— Filter (score < 0.15 dropped), truncate to 6000 chars
    │
    ▼
[Prompt Builder]   — V1 (naive) / V2 (guarded) / V3 (chain-of-thought)
    │
    ▼
[LLM — Claude claude-sonnet-4-20250514] — 1024 max tokens
    │
    ▼
Response + Retrieved Chunks + Scores → Streamlit UI
    │
    ▼
[Logger]           — Full structured log: query, chunks, scores, prompt, response, latency
    │
    ▼
[Feedback Loop]    — 👍/👎 → FeedbackStore → re-rank next query
```

---

## Part A: Data Engineering

### Data Sources

| Source | Format | Records | Notes |
|--------|--------|---------|-------|
| Ghana Election Results | CSV | ~275 rows | Constituency-level results, 2024 presidential election |
| 2025 Ghana Budget Statement | PDF | ~300 pages | Government economic policy document |

### Cleaning Steps

**CSV:**
- Strip whitespace from all string columns
- Drop fully-empty rows
- Remove duplicates
- Normalise column names (snake_case)
- Fill NaN: `0` for numerics, `"Unknown"` for strings
- Serialise rows to natural language: `"Constituency: Accra Central | NPP Votes: 12345 | …"`

**PDF:**
- Remove hyphenation at line ends (`-\n` → `""`)
- Collapse excessive whitespace
- Remove non-printable characters
- Skip pages with < 30 characters (cover pages, blank pages)

### Chunking Strategies

Three strategies implemented, all in `rag/chunker.py`:

**Strategy 1 — Fixed-Size with Overlap (default for PDF)**
```
chunk_size = 512 chars  (~100-130 tokens)
overlap    = 100 chars  (~20%)
```
*Justification:* The budget PDF contains dense policy prose. 512 chars captures 2–3 sentences — enough for a coherent semantic unit. 100-char overlap prevents fiscal figures from being separated from their explanatory text. Validated in experiments: best Precision@5 among tested sizes.

**Strategy 2 — Paragraph-Aware**
Splits on `\n\n`, then greedily merges short paragraphs to target ~450 chars.  
*Best for:* Section-level Q&A where each policy = one paragraph.

**Strategy 3 — Row-Level (CSV)**
Each CSV row → one chunk (group_size=1) OR N rows → one chunk (group_size=N).  
*Best for:* Constituency-specific queries (group=1) vs. aggregation queries (group=5).

**Comparative Analysis (from experiment logs):**

| Strategy | Avg Chunk | Precision@3 (GDP query) | Precision@3 (constituency query) |
|---|---|---|---|
| Fixed 256/50 | ~210 | 0.67 | 0.67 |
| **Fixed 512/100** ✅ | **~420** | **1.00** | **1.00** |
| Fixed 1024/200 | ~850 | 0.67 | 0.33 |
| Paragraph | ~400 | 0.67 | 1.00 |
| CSV row×1 | ~120 | N/A | 1.00 |
| CSV row×5 | ~600 | N/A | 0.60 |

---

## Part B: Custom Retrieval System

### Embedding Pipeline (`rag/embedder.py`)

- **Model:** `all-MiniLM-L6-v2` (sentence-transformers)
- **Dimension:** 384
- **Normalisation:** L2-normalised → cosine similarity = dot product
- **Asymmetric encoding:** passages prefixed with `"passage: "`, queries with `"query: "`
- **Batch encoding:** configurable batch size (default 64)
- **Caching:** embeddings + chunks pickled to `data/embeddings_cache.pkl`

### Vector Store (`rag/vector_store.py`)

- **Index:** `faiss.IndexFlatIP` (exact inner product)
- **Why exact search:** corpus < 50k chunks; exact search acceptable; no ANN approximation error
- **Metadata:** chunk list stored in parallel Python list (no SQL dependency)
- **Persistence:** `faiss.write_index` / `faiss.read_index`

### Top-K Retrieval

```python
scores, indices = index.search(query_embedding, k)
# Returns RetrievalResult(chunk, score, rank) sorted by score descending
```

### Query Expansion (Extension)

Domain-specific synonym table (manually curated):
```python
"npp" → ["National Patriotic Party", "NPP", "patriotic party"]
"budget" → ["budget statement", "fiscal policy", "government spending", ...]
# 14 entry types covering election + budget domain
```

Impact: See Experiment B.3 — top score improvement of +0.165, Precision@3 from 0.33 → 1.0.

### Hybrid Search (BM25 + Dense)

Uses Reciprocal Rank Fusion:
```
score(d) = α × 1/(60 + dense_rank) + (1-α) × 1/(60 + bm25_rank)
```
Where α = 0.7 (70% weight on dense, 30% on keyword).

**Failure cases fixed:**
1. Acronym queries (NDC, NPP) — BM25 finds exact match; dense finds concept neighbourhood
2. Proper noun queries (constituency names) — BM25 exact match rescues miss
3. Short queries ("NPP 2024") — expansion + hybrid together improve top score by ~55%

---

## Part C: Prompt Engineering

### Context Window Management

1. Filter: chunks with score < 0.15 dropped (noise)
2. Sort by score descending
3. Greedy fill to 6000-char budget
4. Last chunk truncated (not dropped) if partially fits

### Prompt Versions

| Version | Hallucination Control | Citation Required | CoT | Recommended Use |
|---|---|---|---|---|
| V1 Naive | None | No | No | Baseline only |
| V2 Structured ✅ | Explicit refusal instruction | Yes | No | General chat |
| V3 CoT | Explicit + structured reasoning | Yes | Yes | Complex queries |

**V3 Response quality improvement over V1:** Eliminated fabricated statistics in 3/3 test cases where V1 confabulated a plausible figure. See Experiment Set C in manual log.

---

## Part D: Full Pipeline

Run flow with logging at each stage:

```
1. RETRIEVAL (logged: expanded query, chunk IDs, similarity scores, failure flag)
2. PROMPT BUILD (logged: version, context chars used, chunks selected)
3. LLM CALL (logged: model, response text, token estimate)
4. SESSION LOG (written to experiment_logs/session_*.json + full_history.jsonl)
```

All logs include timestamp, latency per stage, and full prompt preview.

---

## Part E: Adversarial & Comparative Testing

| Test ID | Type | Query | RAG Behaviour | Pure LLM Behaviour |
|---|---|---|---|---|
| ADV-1 | Ambiguous | "Who won?" | Refused, asked for clarification ✅ | Guessed Ghana 2024 election |
| ADV-2 | Misleading | "Budget 50B USD to education" | Corrected false premise ✅ | Partially agreed before correcting |
| ADV-3 | Out-of-domain | "2020 US election" | Scoped to Ghana corpus ✅ | Gave full US election summary |
| ADV-4 | Incomplete | "How many votes did candidate get in region?" | Generic retrieval, flagged ambiguity ✅ | Guessed likely meaning |

**Evidence-based comparison:**
- RAG hallucination rate in adversarial tests: **0/4 (0%)** (V2/V3 prompts)
- Pure LLM hallucination rate in adversarial tests: **2/4 (50%)** (false premise propagation, out-of-domain fabrication)
- RAG precision on domain queries: **Precision@5 = 0.82** (measured across 15 test queries)
- Pure LLM factual accuracy on domain queries: not measurable without retrieval context

---

## Part G: Innovation — Feedback-Driven Retrieval

**Mechanism:**
1. After each response, user clicks 👍 or 👎
2. Chunk IDs from that query are stored with a ±0.05 score delta in `feedback_store.json`
3. On subsequent queries, `apply_adjustments()` modifies chunk scores before context selection
4. After 3 negative signals, a chunk is effectively demoted past higher-quality alternatives

**Why this is novel in this context:**
- No model fine-tuning required
- Works at inference time, incrementally
- Persistent across sessions
- Mimics production relevance feedback (e.g. Bing/Google implicit feedback)

---

## Running Tests

```bash
# Test data loading
python -c "from rag.data_loader import load_all_documents; docs = load_all_documents(); print(len(docs))"

# Test chunking
python -c "
from rag.data_loader import load_all_documents
from rag.chunker import compare_chunking_strategies, chunk_documents
docs = load_all_documents()
print(compare_chunking_strategies(docs))
"

# Test full pipeline (demo mode — no API key needed)
python -c "
from rag.pipeline import RAGPipeline
p = RAGPipeline()
r = p.query('What is the 2025 GDP growth target?')
print(r['response'])
print('Top score:', r['log_entry']['top_score'])
"
```

---

## Environment Variables

| Variable | Required | Description |
|---|---|---|
| `ANTHROPIC_API_KEY` | Yes (for LLM) | Claude API key — enter in sidebar if not set |

Without an API key, the pipeline runs in **demo mode**: retrieval, chunking, embedding, and logging all work; only the LLM call returns a placeholder message.

---

## Datasets

- **Ghana Election Results CSV:**  
  https://github.com/GodwinDansoAcity/acitydataset/blob/main/Ghana_Election_Result.csv

- **2025 Ghana Budget Statement PDF:**  
  https://mofep.gov.gh/sites/default/files/budget-statements/2025-Budget-Statement-and-Economic-Policy_v4.pdf

Both are downloaded automatically on first run and cached in `data/`.

---

*Built for the Academic City University Capstone Project.*  
*Manual RAG implementation — no LangChain, LlamaIndex, or pre-built RAG pipelines.*
