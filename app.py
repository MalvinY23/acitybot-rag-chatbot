"""
AcityBot — RAG Chatbot UI
Streamlit application with:
  • Query input + chat history
  • Retrieved chunks display (with scores)
  • Prompt version selector
  • Source filter (CSV / PDF / Both)
  • Feedback buttons (Part G)
  • Experiment log viewer
  • Adversarial test runner (Part E)
  • Architecture diagram tab (Part F)
"""

import os
import sys
import json
import time
import streamlit as st

# ── Page config (must be first Streamlit call) ─────────────────────────
st.set_page_config(
    page_title="AcityBot — Academic City RAG Chatbot",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

sys.path.insert(0, os.path.dirname(__file__))

# ── Custom CSS ─────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;600;700&family=IBM+Plex+Mono:wght@400;600&display=swap');

  html, body, [class*="css"] { font-family: 'Space Grotesk', sans-serif; }

  .main-header {
    background: linear-gradient(135deg, #0a2540 0%, #1e40af 100%);
    padding: 2rem 2.5rem; border-radius: 16px; margin-bottom: 1.5rem;
    color: white;
  }
  .main-header h1 { font-size: 2rem; font-weight: 700; margin: 0; }
  .main-header p  { margin: 0.3rem 0 0; opacity: 0.8; font-size: 0.95rem; }

  .chunk-card {
    background: #f0f4ff; border-left: 4px solid #2563eb;
    border-radius: 8px; padding: 0.85rem 1rem; margin-bottom: 0.6rem;
    font-size: 0.85rem;
  }
  .chunk-card .chunk-meta {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.75rem; color: #475569; margin-bottom: 0.3rem;
  }
  .score-badge {
    display: inline-block; background: #2563eb; color: white;
    padding: 1px 8px; border-radius: 99px; font-size: 0.72rem;
    font-weight: 600; margin-left: 6px;
  }
  .score-low { background: #dc2626; }
  .score-mid { background: #d97706; }
  .score-high{ background: #16a34a; }

  .response-box {
    background: #ffffff; border: 1px solid #e2e8f0;
    border-radius: 12px; padding: 1.2rem 1.4rem;
    box-shadow: 0 1px 6px rgba(0,0,0,0.05);
  }
  .failure-banner {
    background: #fef2f2; border: 1px solid #fca5a5;
    border-radius: 8px; padding: 0.7rem 1rem; color: #b91c1c;
    font-size: 0.85rem; margin-bottom: 0.7rem;
  }
  .feedback-row { margin-top: 0.7rem; }
  .stat-pill {
    background: #e0e7ff; color: #3730a3; padding: 3px 10px;
    border-radius: 99px; font-size: 0.78rem; font-weight: 600;
    display: inline-block; margin-right: 6px;
  }
  .log-entry {
    background: #f8fafc; border: 1px solid #e2e8f0;
    border-radius: 8px; padding: 0.8rem; margin-bottom: 0.5rem;
    font-size: 0.82rem; font-family: 'IBM Plex Mono', monospace;
  }
  .chat-bubble-user {
    background: #2563eb; color: white; padding: 0.8rem 1.1rem;
    border-radius: 16px 16px 4px 16px; margin-left: 20%; margin-bottom: 0.5rem;
    font-size: 0.92rem;
  }
  .chat-bubble-bot {
    background: #f1f5f9; color: #0f172a; padding: 0.8rem 1.1rem;
    border-radius: 16px 16px 16px 4px; margin-right: 20%; margin-bottom: 0.5rem;
    font-size: 0.92rem;
  }
</style>
""", unsafe_allow_html=True)


# ── Session state init ─────────────────────────────────────────────────
if "pipeline" not in st.session_state:
    st.session_state.pipeline = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "last_result" not in st.session_state:
    st.session_state.last_result = None
if "index_ready" not in st.session_state:
    st.session_state.index_ready = False


# ── SIDEBAR ────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Configuration")

    api_key = st.text_input(
        "Anthropic API Key",
        type="password",
        value=os.environ.get("ANTHROPIC_API_KEY", ""),
        help="Your API key — stored only in session memory",
    )

    st.markdown("---")

    prompt_ver = st.selectbox(
        "Prompt Version",
        options=["v2", "v1", "v3"],
        index=0,
        format_func=lambda v: {
            "v1": "V1 — Naive (baseline)",
            "v2": "V2 — Structured + Guard ✓",
            "v3": "V3 — Chain-of-Thought",
        }[v],
        help="V2 is recommended. V3 adds reasoning steps.",
    )

    top_k = st.slider("Top-K Chunks", 1, 10, 5)

    source_filter = st.selectbox(
        "Source Filter",
        ["Both", "Election CSV only", "Budget PDF only"],
    )
    source_map = {
        "Both": None,
        "Election CSV only": "csv",
        "Budget PDF only": "pdf",
    }

    use_expansion = st.toggle("Query Expansion", value=True)
    use_hybrid    = st.toggle("Hybrid Search (BM25 + Vector)", value=True)
    force_rebuild = st.toggle("Force Index Rebuild", value=False)

    st.markdown("---")

    if st.button("🚀 Initialise / Rebuild Index", use_container_width=True, type="primary"):
        with st.spinner("Loading data, chunking, embedding, indexing …"):
            from rag.pipeline import RAGPipeline
            st.session_state.pipeline = RAGPipeline(
                api_key=api_key or None,
                prompt_version=prompt_ver,
                k=top_k,
                use_query_expansion=use_expansion,
                use_hybrid=use_hybrid,
                force_rebuild=force_rebuild,
            )
            st.session_state.index_ready = True
        st.success("✅ Index ready!")

    if st.session_state.index_ready and st.session_state.pipeline:
        stats = st.session_state.pipeline.stats()
        vs    = stats["vector_store"]
        st.markdown(f"""
        **Index Stats**
        - Total vectors : `{vs['total_vectors']}`
        - CSV chunks    : `{vs['csv_chunks']}`
        - PDF chunks    : `{vs['pdf_chunks']}`
        - Embedding dim : `{vs['dimension']}`
        - Total runs    : `{stats['total_runs']}`
        """)

    st.markdown("---")
    st.markdown("**📁 Datasets**")
    st.markdown("- Ghana Election Results (CSV)")
    st.markdown("- 2025 Budget Statement (PDF)")
    st.markdown("---")
    st.caption("AcityBot v1.0 — Academic City University  \nManual RAG | No LangChain")


# ── MAIN TABS ──────────────────────────────────────────────────────────
tab_chat, tab_chunks, tab_logs, tab_adv, tab_arch = st.tabs([
    "💬 Chat",
    "📄 Retrieved Chunks",
    "📊 Experiment Logs",
    "⚔️ Adversarial Tests",
    "🏛 Architecture",
])


# ══════════════════════════════════════════════════════════════════════
#  TAB 1 — CHAT
# ══════════════════════════════════════════════════════════════════════
with tab_chat:
    st.markdown("""
    <div class="main-header">
      <h1>🎓 AcityBot — Academic City RAG Assistant</h1>
      <p>Ask about Ghana's 2024 Election Results or the 2025 Budget Statement</p>
    </div>
    """, unsafe_allow_html=True)

    # Example queries
    st.markdown("**Try an example:**")
    col1, col2, col3 = st.columns(3)
    example_q = None
    with col1:
        if st.button("🗳️ Who won the 2024 election?"):
            example_q = "Who won the 2024 Ghana presidential election?"
    with col2:
        if st.button("💰 Budget GDP growth target?"):
            example_q = "What is Ghana's GDP growth target in the 2025 budget?"
    with col3:
        if st.button("📍 NPP votes in Ashanti?"):
            example_q = "How many votes did NPP receive in the Ashanti region?"

    # Chat history display
    if st.session_state.chat_history:
        for turn in st.session_state.chat_history[-6:]:
            st.markdown(f'<div class="chat-bubble-user">👤 {turn["query"]}</div>',
                        unsafe_allow_html=True)
            st.markdown(f'<div class="chat-bubble-bot">🤖 {turn["response"]}</div>',
                        unsafe_allow_html=True)

    # Query input
    with st.form("chat_form", clear_on_submit=True):
        user_query = st.text_input(
            "Your question",
            value=example_q or "",
            placeholder="e.g. What is the 2025 inflation target?",
            label_visibility="collapsed",
        )
        submitted = st.form_submit_button("Send ▶", use_container_width=True, type="primary")

    if submitted and user_query.strip():
        if not st.session_state.index_ready or not st.session_state.pipeline:
            st.error("⚠️ Please initialise the index first using the sidebar button.")
        else:
            pipeline = st.session_state.pipeline
            # Update pipeline settings if changed
            pipeline.prompt_version = prompt_ver
            from rag.prompt_builder import PromptBuilder
            pipeline.prompt_builder = PromptBuilder(version=prompt_ver)
            pipeline.k              = top_k
            pipeline.retriever.use_query_expansion = use_expansion
            pipeline.retriever.use_hybrid          = use_hybrid

            with st.spinner("Retrieving … Generating …"):
                result = pipeline.query(
                    user_query.strip(),
                    source_filter=source_map[source_filter],
                )

            st.session_state.last_result = result
            st.session_state.chat_history.append({
                "query":    user_query.strip(),
                "response": result["response"],
                "result":   result,
            })
            st.rerun()

    # Show last response with feedback
    if st.session_state.last_result:
        r = st.session_state.last_result
        log = r["log_entry"]

        # Failure banner
        if log["failure_detected"]:
            st.markdown(
                f'<div class="failure-banner">⚠️ Low-confidence retrieval: '
                f'{log["failure_reason"]}</div>',
                unsafe_allow_html=True,
            )

        # Response
        st.markdown('<div class="response-box">', unsafe_allow_html=True)
        st.markdown(r["response"])
        st.markdown("</div>", unsafe_allow_html=True)

        # Stats row
        timers = r["timers"]
        st.markdown(
            f'<span class="stat-pill">🔍 Retrieval: {timers["retrieval_ms"]:.0f}ms</span>'
            f'<span class="stat-pill">✍️ Prompt: {timers["prompt_ms"]:.0f}ms</span>'
            f'<span class="stat-pill">🤖 LLM: {timers["llm_ms"]:.0f}ms</span>'
            f'<span class="stat-pill">📄 Chunks: {len(r["selected_chunks"])}</span>'
            f'<span class="stat-pill">Top score: {log["top_score"]:.3f}</span>',
            unsafe_allow_html=True,
        )

        # Feedback (Part G Innovation)
        st.markdown('<div class="feedback-row">', unsafe_allow_html=True)
        st.markdown("**Was this helpful?**")
        fc1, fc2, _ = st.columns([1, 1, 8])
        last_query = st.session_state.chat_history[-1]["query"] if st.session_state.chat_history else ""
        chunk_ids  = [c["chunk_id"] for c in r["retrieved_chunks"]]

        with fc1:
            if st.button("👍 Yes", key="fb_pos"):
                if st.session_state.pipeline:
                    st.session_state.pipeline.submit_feedback(last_query, chunk_ids, +1)
                st.toast("Thanks! Feedback recorded ✓", icon="✅")
        with fc2:
            if st.button("👎 No", key="fb_neg"):
                if st.session_state.pipeline:
                    st.session_state.pipeline.submit_feedback(last_query, chunk_ids, -1)
                st.toast("Noted! These chunks will be down-ranked.", icon="📉")
        st.markdown("</div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════
#  TAB 2 — RETRIEVED CHUNKS
# ══════════════════════════════════════════════════════════════════════
with tab_chunks:
    st.markdown("### 📄 Retrieved Chunks (Last Query)")

    if st.session_state.last_result:
        r = st.session_state.last_result
        debug = r["debug"]

        st.markdown(f"**Original Query:** `{debug['original_query']}`")
        st.markdown(f"**Expanded Query:** `{debug.get('expanded_query', debug['original_query'])}`")
        st.markdown(f"**Retrieval Mode:** `{debug.get('retrieval_mode', 'vector')}`")
        st.markdown("---")

        for chunk in r["retrieved_chunks"]:
            score = chunk["score"]
            badge_cls = "score-high" if score > 0.5 else ("score-mid" if score > 0.25 else "score-low")
            st.markdown(f"""
            <div class="chunk-card">
              <div class="chunk-meta">
                [{chunk.get('rank', '?')}]  {chunk['source_name']}  ·  ID: {chunk['chunk_id']}
                <span class="score-badge {badge_cls}">{score:.4f}</span>
              </div>
              {chunk['text'][:400].replace('<','&lt;').replace(chr(10),' ')} …
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("Run a query in the Chat tab to see retrieved chunks here.")


# ══════════════════════════════════════════════════════════════════════
#  TAB 3 — EXPERIMENT LOGS
# ══════════════════════════════════════════════════════════════════════
with tab_logs:
    st.markdown("### 📊 Experiment Logs")

    if st.session_state.pipeline:
        entries = st.session_state.pipeline.logger.get_all_entries()
        if entries:
            # Summary table
            import pandas as pd
            rows = []
            for e in entries:
                rows.append({
                    "Query":          e["query"][:50],
                    "Prompt Ver.":    e["prompt_version"],
                    "Top Score":      f"{e['top_score']:.3f}",
                    "Avg Score":      f"{e['avg_score']:.3f}",
                    "Failure":        "⚠️" if e["failure_detected"] else "✅",
                    "Total ms":       e["latency_total_ms"],
                    "Response (preview)": str(e["llm_response"])[:80],
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True)

            # Detail expander per entry
            for i, e in enumerate(entries):
                with st.expander(f"Run {i+1}: {e['query'][:60]}"):
                    col_a, col_b = st.columns(2)
                    with col_a:
                        st.markdown(f"**Prompt Version:** {e['prompt_version']}")
                        st.markdown(f"**Expanded Query:** {e['expanded_query']}")
                        st.markdown(f"**Similarity Scores:** {e['similarity_scores']}")
                        st.markdown(f"**Failure:** {e['failure_detected']} — {e['failure_reason']}")
                    with col_b:
                        st.markdown(f"**Retrieval:** {e['latency_retrieval_ms']} ms")
                        st.markdown(f"**Prompt build:** {e['latency_prompt_ms']} ms")
                        st.markdown(f"**LLM:** {e['latency_llm_ms']} ms")
                        st.markdown(f"**Total:** {e['latency_total_ms']} ms")

                    st.markdown("**Full Prompt Preview:**")
                    st.code(e["full_prompt_preview"], language="text")
                    st.markdown("**LLM Response:**")
                    st.markdown(e["llm_response"])
        else:
            st.info("No runs yet. Ask a question in the Chat tab.")

        # Download all logs
        if entries:
            st.download_button(
                "⬇️ Download Full Log (JSON)",
                data=json.dumps(entries, indent=2, default=str),
                file_name="experiment_logs.json",
                mime="application/json",
            )

        # Chunking comparison
        st.markdown("---")
        st.markdown("### Chunking Strategy Comparison")
        st.markdown("""
        | Strategy | Chunk Size | Overlap | Avg Length | Pros | Cons |
        |---|---|---|---|---|---|
        | Fixed 512/100 ✓ | 512 chars | 100 chars | ~420 | Predictable; handles cross-sentence context | May split mid-concept |
        | Fixed 256/50 | 256 chars | 50 chars | ~210 | More precise recall | Loses broader context |
        | Fixed 1024/200 | 1024 chars | 200 chars | ~850 | Very broad context | Dilutes embedding signal |
        | Paragraph-Aware | Variable | N/A | ~400 | Semantic integrity | Uneven lengths |
        | CSV Row (×1) ✓ | 1 row | N/A | ~120 | Maximum precision | No multi-row queries |
        | CSV Row (×5) | 5 rows | N/A | ~600 | Multi-candidate comparisons | Noisy retrieval |

        **Selected:** Fixed 512/100 for PDF; Row×1 for CSV — best Precision@5 in experiments.
        """)

    else:
        st.info("Initialise the index from the sidebar first.")


# ══════════════════════════════════════════════════════════════════════
#  TAB 4 — ADVERSARIAL TESTS
# ══════════════════════════════════════════════════════════════════════
with tab_adv:
    st.markdown("### ⚔️ Adversarial & Comparative Testing (Part E)")
    st.markdown("""
    Tests designed to expose RAG failure modes:
    - Ambiguous queries (missing subject)
    - Misleading premises (false facts in question)
    - Out-of-domain queries
    - Incomplete queries
    """)

    if st.session_state.index_ready and st.session_state.pipeline:
        if st.button("▶ Run All Adversarial Tests", type="primary"):
            with st.spinner("Running adversarial tests (this may take ~60s) …"):
                results = st.session_state.pipeline.run_adversarial_tests()

            for r in results:
                with st.expander(f"{r['id']} — {r['type']}: `{r['query']}`"):
                    st.markdown(f"**Description:** {r['description']}")
                    st.markdown(f"**Expected behaviour:** {r['expected_behaviour']}")
                    st.markdown(f"**RAG top score:** `{r['rag_top_score']:.4f}` | "
                                f"**Failure detected:** `{r['rag_failure_detected']}`")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("**🤖 RAG Response:**")
                        st.info(r["rag_response"])
                    with col2:
                        st.markdown("**🧠 Pure LLM (no retrieval):**")
                        st.warning(r["pure_llm_response"])

        st.markdown("---")
        st.markdown("### Prompt Version Comparison (Part C)")
        compare_q = st.text_input(
            "Query to compare across prompt versions:",
            value="What is Ghana's inflation target for 2025?",
        )
        if st.button("▶ Compare V1 / V2 / V3"):
            with st.spinner("Running 3 prompt versions …"):
                comparisons = st.session_state.pipeline.run_prompt_experiment(compare_q)
            for c in comparisons:
                with st.expander(f"Prompt {c['prompt_version'].upper()} "
                                 f"({c['prompt_chars']} chars | {c['latency_ms']} ms)"):
                    st.markdown(c["response"])
    else:
        st.info("Initialise the index from the sidebar first.")


# ══════════════════════════════════════════════════════════════════════
#  TAB 5 — ARCHITECTURE  (Part F)
# ══════════════════════════════════════════════════════════════════════
with tab_arch:
    st.markdown("### 🏛 System Architecture (Part F)")
    st.markdown("""
    ```
    ╔══════════════════════════════════════════════════════════════════╗
    ║                     AcityBot RAG System                         ║
    ╠══════════════════════════════════════════════════════════════════╣
    ║                                                                  ║
    ║  DATA SOURCES           DATA ENGINEERING           VECTOR STORE  ║
    ║  ┌──────────┐           ┌────────────────┐        ┌──────────┐  ║
    ║  │ CSV      │──clean──▶ │  Chunker       │──emb──▶│  FAISS   │  ║
    ║  │ Election │           │ Fixed 512/100  │        │ IndexIP  │  ║
    ║  └──────────┘           │ Paragraph-Aware│        │ 384-dim  │  ║
    ║  ┌──────────┐           │ Row-Level      │        └────┬─────┘  ║
    ║  │ PDF      │──clean──▶ └────────────────┘             │        ║
    ║  │ Budget   │                                           │ search ║
    ║  └──────────┘           ┌────────────────┐             │        ║
    ║                         │  Embedder      │◀────────────┘        ║
    ║  USER QUERY             │ MiniLM-L6-v2   │                      ║
    ║  ┌──────────┐           │ 384-dim, L2    │                      ║
    ║  │  Streamlit│          └────────────────┘                      ║
    ║  │  UI      │                                                    ║
    ║  └────┬─────┘  ┌──────────────────────────────────────────────┐ ║
    ║       │        │  RETRIEVER                                   │ ║
    ║       ▼        │  1. Query Expansion (domain synonym table)   │ ║
    ║  Raw Query ──▶ │  2. Dense FAISS search (cosine similarity)   │ ║
    ║                │  3. BM25 Sparse search (keyword)             │ ║
    ║                │  4. Reciprocal Rank Fusion (hybrid merge)    │ ║
    ║                │  5. Failure detection (threshold < 0.25)     │ ║
    ║                │  6. Feedback score adjustment (Part G)       │ ║
    ║                └────────────────────┬─────────────────────────┘ ║
    ║                                     │ Top-K chunks              ║
    ║                                     ▼                            ║
    ║                ┌──────────────────────────────────────────────┐ ║
    ║                │  PROMPT BUILDER                              │ ║
    ║                │  - Context window management (6000 chars)    │ ║
    ║                │  - Score threshold filter (< 0.15 dropped)   │ ║
    ║                │  - V1 / V2 / V3 prompt templates             │ ║
    ║                │  - Hallucination guard + citation rules       │ ║
    ║                └────────────────────┬─────────────────────────┘ ║
    ║                                     │ Final prompt               ║
    ║                                     ▼                            ║
    ║                ┌──────────────────────────────────────────────┐ ║
    ║                │  LLM (Claude claude-sonnet-4-20250514)              │ ║
    ║                │  Max tokens: 1024  |  Temperature: default   │ ║
    ║                └────────────────────┬─────────────────────────┘ ║
    ║                                     │ Response                   ║
    ║                                     ▼                            ║
    ║  ┌─────────────────────────────────────────────────────────────┐║
    ║  │  LOGGER: timestamps, scores, prompts, latencies, failures  │║
    ║  └─────────────────────────────────────────────────────────────┘║
    ║                                     │                            ║
    ║                              Streamlit UI                        ║
    ║                   Response + Chunks + Scores displayed           ║
    ║                   Feedback → FeedbackStore → re-rank             ║
    ╚══════════════════════════════════════════════════════════════════╝
    ```
    """)

    st.markdown("""
    #### Component Justifications

    **Why FAISS IndexFlatIP?**
    Exact search (no ANN approximation) is appropriate for our corpus size (~5k–30k chunks).
    IndexFlatIP + L2-normalised embeddings = exact cosine similarity. For production at scale,
    IndexIVFFlat or HNSW would be used.

    **Why all-MiniLM-L6-v2?**
    22M parameters, 384-dim embeddings, runs on CPU in <100ms per query. The asymmetric
    query/passage encoding (prefixed prompts) is well-suited to our open-domain Q&A task.

    **Why Hybrid Search (BM25 + Dense)?**
    Dense models struggle with rare proper nouns (constituency names, person names).
    BM25 handles exact keyword matches. Reciprocal Rank Fusion merges both ranked lists
    without needing score normalisation.

    **Why Feedback Loop (Part G)?**
    Continuously improves retrieval without retraining. Users signal relevance;
    chunk scores are adjusted persistently. After ~20 queries, noisy chunks are
    naturally down-ranked. This simulates production online learning.

    **Why is this design suitable for Academic City?**
    - Both datasets are authoritative primary sources (government data).
    - Students/faculty ask factual questions → hallucination control is critical.
    - V2/V3 prompts enforce citation, preventing the LLM from inventing policy details.
    - The feedback loop allows library or faculty staff to improve quality over time
      without needing ML expertise.
    """)

    st.markdown("---")
    st.markdown("#### Part G Innovation: Feedback-Driven Retrieval")
    st.markdown("""
    | Feature | Description |
    |---------|-------------|
    | **Signal** | User thumbs-up / thumbs-down per response |
    | **Storage** | `experiment_logs/feedback_store.json` — query → chunk_id → delta |
    | **Effect** | ±0.05 score adjustment per feedback event |
    | **Application** | Applied before context selection on every query |
    | **Novelty** | No model fine-tuning required; works at inference time |
    | **Limitation** | Score drift if feedback is biased; reset button recommended for production |
    """)
