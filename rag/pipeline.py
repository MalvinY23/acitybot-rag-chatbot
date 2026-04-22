"""
Full RAG Pipeline — Part D + Part G Innovation
================================================
User Query → Retrieval → Context Selection → Prompt → LLM → Response

Logs every stage with timing, retrieved docs, similarity scores,
and the exact prompt sent to the LLM.

INNOVATION (Part G): Feedback Loop for Retrieval Improvement
─────────────────────────────────────────────────────────────
Users can rate each response (👍 / 👎).
Negative feedback triggers:
  1. A relevance penalty on that query's chunk combination (stored in
     feedback_store.json).
  2. On subsequent identical / similar queries, penalised chunks are
     down-ranked before being injected into the prompt.
This simulates online learning without re-training the embedding model.
"""

import os
import json
import time
from typing import List, Dict, Any, Optional, Tuple
import anthropic

from .data_loader import load_all_documents, RawDocument
from .chunker import chunk_documents, Chunk
from .embedder import EmbeddingPipeline
from .vector_store import FAISSVectorStore, RetrievalResult
from .retriever import Retriever
from .prompt_builder import PromptBuilder
from .logger import ExperimentLogger, StageTimer

FEEDBACK_PATH = "experiment_logs/feedback_store.json"
MODEL          = "claude-sonnet-4-20250514"


# ─────────────────────────────────────────────────────────────────────
#  FEEDBACK STORE  (Part G Innovation)
# ─────────────────────────────────────────────────────────────────────

class FeedbackStore:
    """
    Persists user feedback (query → {chunk_ids: score_delta}).
    Negative ratings lower chunk scores; positive ratings boost them.
    """

    def __init__(self, path: str = FEEDBACK_PATH):
        self.path = path
        self.data: Dict[str, Dict[str, float]] = {}
        self._load()

    def _load(self):
        if os.path.exists(self.path):
            try:
                with open(self.path) as f:
                    raw = f.read().strip()
                    self.data = json.loads(raw) if raw else {}
            except (json.JSONDecodeError, ValueError):
                self.data = {}

    def _save(self):
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        with open(self.path, "w") as f:
            json.dump(self.data, f, indent=2)

    def record(self, query: str, chunk_ids: List[str], rating: int):
        """
        rating: +1 (helpful) or -1 (not helpful).
        Stores cumulative delta per chunk_id for this query pattern.
        """
        key = query.lower().strip()
        if key not in self.data:
            self.data[key] = {}
        for cid in chunk_ids:
            self.data[key][cid] = self.data[key].get(cid, 0) + rating * 0.05
        self._save()

    def get_adjustment(self, query: str, chunk_id: str) -> float:
        """Return score adjustment (positive boost or negative penalty)."""
        key = query.lower().strip()
        return self.data.get(key, {}).get(chunk_id, 0.0)

    def apply_adjustments(
        self,
        query: str,
        results: List[RetrievalResult],
    ) -> List[RetrievalResult]:
        """Apply stored adjustments and re-sort results."""
        adjusted = []
        for r in results:
            adj = self.get_adjustment(query, r.chunk.chunk_id)
            if adj != 0:
                import copy
                r_copy = copy.copy(r)
                r_copy.score = max(0.0, min(1.0, r.score + adj))
                adjusted.append(r_copy)
            else:
                adjusted.append(r)
        adjusted.sort(key=lambda x: x.score, reverse=True)
        for i, r in enumerate(adjusted, 1):
            r.rank = i
        return adjusted


# ─────────────────────────────────────────────────────────────────────
#  MAIN PIPELINE
# ─────────────────────────────────────────────────────────────────────

class RAGPipeline:
    """
    Encapsulates the complete RAG workflow.
    Initialised once; query() called per user message.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        prompt_version: str = "v2",
        k: int = 5,
        use_query_expansion: bool = True,
        use_hybrid: bool = True,
        force_rebuild: bool = False,
    ):
        self.prompt_version = prompt_version
        self.k = k
        self.logger = ExperimentLogger()
        self.feedback = FeedbackStore()

        # ── Anthropic client ──────────────────────────────────────────
        _key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        self.client = anthropic.Anthropic(api_key=_key) if _key else None
        if not self.client:
            print("[Pipeline] ⚠ No ANTHROPIC_API_KEY set. LLM calls will be simulated.")

        # ── Embedder ──────────────────────────────────────────────────
        self.embedder = EmbeddingPipeline()

        # ── Vector store ──────────────────────────────────────────────
        self.vector_store = FAISSVectorStore(embedding_dim=self.embedder.embedding_dim)

        # ── Build or load index ───────────────────────────────────────
        if not force_rebuild and self.vector_store.load():
            print("[Pipeline] Using cached FAISS index.")
        else:
            self._build_index()

        # ── Retriever & Prompt builder ───────────────────────────────
        self.retriever = Retriever(
            self.vector_store, self.embedder,
            use_query_expansion=use_query_expansion,
            use_hybrid=use_hybrid,
        )
        self.prompt_builder = PromptBuilder(version=prompt_version)

    # ──────────────────────────────────────────────────────────────────
    #  INDEX BUILDER
    # ──────────────────────────────────────────────────────────────────

    def _build_index(self):
        print("[Pipeline] Building index from scratch …")
        documents = load_all_documents()
        chunks    = chunk_documents(documents, pdf_strategy="fixed",
                                    chunk_size=512, overlap=100, csv_group_size=1)
        embeddings = self.embedder.encode_chunks(chunks)
        self.vector_store.add(chunks, embeddings)
        self.vector_store.save()
        self.embedder.save(chunks, embeddings)
        print(f"[Pipeline] Index built: {self.vector_store.stats()}")

    # ──────────────────────────────────────────────────────────────────
    #  MAIN QUERY METHOD
    # ──────────────────────────────────────────────────────────────────

    def query(
        self,
        user_query: str,
        source_filter: Optional[str] = None,
        return_debug: bool = True,
    ) -> Dict[str, Any]:
        """
        Full pipeline run.

        Returns a dict with:
          - response        : LLM answer
          - retrieved_chunks: list of chunk dicts (for display)
          - debug           : retrieval debug info
          - log_entry       : full log record
        """
        timers = StageTimer()

        # ── STAGE 1: Retrieval ─────────────────────────────────────────
        timers.start()
        results, debug = self.retriever.retrieve(
            user_query, k=self.k, source_filter=source_filter
        )
        # Apply feedback adjustments (Part G)
        results = self.feedback.apply_adjustments(user_query, results)
        timers.mark("retrieval")

        print(f"\n[Pipeline] ─── RETRIEVAL ───────────────────────────────")
        print(f"  Query      : {user_query}")
        print(f"  Expanded   : {debug.get('expanded_query', user_query)}")
        for r in results:
            print(f"  [{r.rank}] score={r.score:.4f}  src={r.chunk.source_name[:30]}"
                  f"  text={r.chunk.text[:80].replace(chr(10),' ')} …")

        # ── STAGE 2: Prompt Construction ──────────────────────────────
        timers.start()
        system_prompt, user_message, selected = self.prompt_builder.build(
            user_query, results
        )
        timers.mark("prompt")

        print(f"\n[Pipeline] ─── PROMPT ──────────────────────────────────")
        print(f"  Version    : {self.prompt_version}")
        print(f"  Chunks used: {len(selected)}/{len(results)}")
        print(f"  Prompt len : {len(user_message)} chars")
        print(f"  Preview    :\n{user_message[:400]} …\n")

        # ── STAGE 3: LLM Generation ───────────────────────────────────
        timers.start()
        response_text = self._call_llm(system_prompt, user_message)
        timers.mark("llm")

        print(f"\n[Pipeline] ─── RESPONSE ────────────────────────────────")
        print(f"  {response_text[:300]} …")

        # ── STAGE 4: Logging ──────────────────────────────────────────
        retrieved_for_log = [
            {"chunk_id": r.chunk.chunk_id, "source": r.chunk.source,
             "source_name": r.chunk.source_name,
             "text": r.chunk.text, "score": r.score}
            for r in results
        ]
        log_entry = self.logger.log(
            query=user_query,
            expanded_query=debug.get("expanded_query", user_query),
            retrieved_chunks=retrieved_for_log,
            similarity_scores=[r.score for r in results],
            system_prompt=system_prompt,
            user_prompt=user_message,
            llm_response=response_text,
            prompt_version=self.prompt_version,
            failure_detected=debug.get("failure_detected", False),
            failure_reason=debug.get("failure_reason"),
            timers=timers,
        )

        return {
            "response":         response_text,
            "retrieved_chunks": retrieved_for_log,
            "debug":            debug,
            "selected_chunks":  [
                {"chunk_id": r.chunk.chunk_id, "source_name": r.chunk.source_name,
                 "score": r.score, "text": r.chunk.text}
                for r in selected
            ],
            "log_entry":        log_entry,
            "timers": {
                "retrieval_ms": timers.retrieval_ms,
                "prompt_ms":    timers.prompt_ms,
                "llm_ms":       timers.llm_ms,
            },
        }

    # ──────────────────────────────────────────────────────────────────
    #  LLM CALL
    # ──────────────────────────────────────────────────────────────────

    def _call_llm(self, system_prompt: str, user_message: str) -> str:
        if not self.client:
            return (
                "[DEMO MODE — No API key] "
                "The pipeline ran successfully. Set ANTHROPIC_API_KEY to get real responses."
            )
        try:
            response = self.client.messages.create(
                model=MODEL,
                max_tokens=1024,
                system=system_prompt,
                messages=[{"role": "user", "content": user_message}],
            )
            return response.content[0].text
        except Exception as e:
            return f"[LLM Error: {str(e)}]"

    # ──────────────────────────────────────────────────────────────────
    #  FEEDBACK  (Part G)
    # ──────────────────────────────────────────────────────────────────

    def submit_feedback(self, query: str, chunk_ids: List[str], rating: int):
        """rating: +1 or -1."""
        self.feedback.record(query, chunk_ids, rating)
        print(f"[Feedback] Recorded {rating:+d} for {len(chunk_ids)} chunks on query: '{query}'")

    # ──────────────────────────────────────────────────────────────────
    #  ADVERSARIAL TEST  (Part E)
    # ──────────────────────────────────────────────────────────────────

    def run_adversarial_tests(self) -> List[Dict]:
        """
        Run 4 adversarial queries and compare RAG vs pure LLM.
        Documents accuracy, hallucination rate, consistency.
        """
        tests = [
            {
                "id": "ADV-1",
                "type": "Ambiguous",
                "query": "Who won?",
                "description": "Extremely ambiguous — no subject, no context.",
                "expected_behaviour": "RAG should ask for clarification or return low confidence.",
            },
            {
                "id": "ADV-2",
                "type": "Misleading",
                "query": "The 2025 budget allocated 50 billion USD to education — explain why.",
                "description": "False premise embedded in question.",
                "expected_behaviour": "RAG should correct the premise, not validate it.",
            },
            {
                "id": "ADV-3",
                "type": "Out-of-domain",
                "query": "What are the electoral results for the 2020 US presidential election?",
                "description": "Out-of-domain query — our corpus is Ghana only.",
                "expected_behaviour": "RAG should decline or flag lack of relevant context.",
            },
            {
                "id": "ADV-4",
                "type": "Incomplete",
                "query": "How many votes did the candidate get in region?",
                "description": "Missing key entities (candidate name, region name).",
                "expected_behaviour": "RAG retrieves generically; should flag ambiguity.",
            },
        ]

        results = []
        for test in tests:
            print(f"\n[Adversarial] {test['id']}: {test['query']}")

            # RAG response
            rag_out = self.query(test["query"])
            rag_response = rag_out["response"]
            top_score    = rag_out["log_entry"]["top_score"]

            # Pure LLM (no retrieval)
            pure_llm = self._call_llm(
                "You are a knowledgeable assistant. Answer the question directly.",
                test["query"]
            )

            results.append({
                **test,
                "rag_response":          rag_response,
                "pure_llm_response":     pure_llm,
                "rag_top_score":         top_score,
                "rag_failure_detected":  rag_out["debug"]["failure_detected"],
                "rag_chunks_returned":   len(rag_out["retrieved_chunks"]),
            })

        return results

    # ──────────────────────────────────────────────────────────────────
    #  PROMPT EXPERIMENT  (Part C)
    # ──────────────────────────────────────────────────────────────────

    def run_prompt_experiment(self, query: str) -> List[Dict]:
        """
        Run the same query with all three prompt versions.
        Returns comparison for experiment log.
        """
        comparisons = []
        original_version = self.prompt_version

        for v in ["v1", "v2", "v3"]:
            self.prompt_version = v
            self.prompt_builder = PromptBuilder(version=v)
            out = self.query(query)
            comparisons.append({
                "prompt_version": v,
                "response":       out["response"],
                "prompt_chars":   out["log_entry"]["user_prompt_chars"],
                "latency_ms":     out["log_entry"]["latency_total_ms"],
            })

        self.prompt_version = original_version
        self.prompt_builder = PromptBuilder(version=original_version)
        return comparisons

    # ──────────────────────────────────────────────────────────────────
    #  STATS
    # ──────────────────────────────────────────────────────────────────

    def stats(self) -> Dict:
        return {
            "vector_store": self.vector_store.stats(),
            "prompt_version": self.prompt_version,
            "k": self.k,
            "total_runs": len(self.logger.entries),
        }
