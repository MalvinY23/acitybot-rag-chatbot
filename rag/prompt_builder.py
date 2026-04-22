"""
RAG Prompt Builder — Part C Implementation
============================================
Implements three prompt iterations:
  V1 — Naive injection (baseline)
  V2 — Structured with hallucination guard and source citation instruction
  V3 — V2 + chain-of-thought scaffolding (innovation, Part G)

Context window management:
  • Max context chars : 6000  (leaves room for system prompt + response in
    Claude's 200k context window; conservative for cost efficiency)
  • Strategy: Rank chunks by similarity score, truncate from the bottom.
  • Filter: drop chunks below MIN_SCORE threshold (< 0.15).
"""

from typing import List, Tuple
from .vector_store import RetrievalResult

MAX_CONTEXT_CHARS = 6000
MIN_SCORE         = 0.15


# ─────────────────────────────────────────────────────────────────────
#  CONTEXT WINDOW MANAGEMENT
# ─────────────────────────────────────────────────────────────────────

def select_context(
    results: List[RetrievalResult],
    max_chars: int = MAX_CONTEXT_CHARS,
    min_score: float = MIN_SCORE,
) -> Tuple[List[RetrievalResult], str]:
    """
    Filter, rank, and truncate retrieved chunks to fit the context budget.

    Steps:
      1. Drop chunks below min_score  (noise filtering).
      2. Sort remaining by score descending (already sorted by retriever).
      3. Greedily add chunks until max_chars is reached.

    Returns:
      (selected_results, formatted_context_string)
    """
    filtered = [r for r in results if r.score >= min_score]

    selected: List[RetrievalResult] = []
    total_chars = 0

    for r in filtered:
        chunk_len = len(r.chunk.text)
        if total_chars + chunk_len > max_chars:
            # Truncate the last chunk rather than drop it entirely
            remaining = max_chars - total_chars
            if remaining > 100:
                from dataclasses import replace
                truncated_chunk = r.chunk
                # shallow copy with truncated text
                import copy
                truncated = copy.copy(r)
                # Create a copy of the chunk with truncated text
                chunk_copy = copy.copy(r.chunk)
                chunk_copy.text = r.chunk.text[:remaining] + " [TRUNCATED]"
                truncated.chunk = chunk_copy
                selected.append(truncated)
            break
        selected.append(r)
        total_chars += chunk_len

    # Build formatted context string
    parts: List[str] = []
    for i, r in enumerate(selected, start=1):
        source_label = f"{r.chunk.source_name}"
        if r.chunk.source == "pdf":
            page = r.chunk.metadata.get("page", "?")
            source_label += f" (page {page})"
        parts.append(
            f"[Context {i} | Source: {source_label} | Score: {r.score:.4f}]\n"
            f"{r.chunk.text}"
        )

    context_str = "\n\n---\n\n".join(parts)
    return selected, context_str


# ─────────────────────────────────────────────────────────────────────
#  SYSTEM PROMPTS
# ─────────────────────────────────────────────────────────────────────

SYSTEM_V1 = "You are a helpful assistant for Academic City University."

SYSTEM_V2 = """You are AcityBot, an AI assistant for Academic City University, Ghana.
You answer questions about Ghana's 2024 election results and the 2025 Budget Statement.

STRICT RULES:
1. Base your answer EXCLUSIVELY on the provided context passages.
2. If the context does not contain enough information, say:
   "I don't have sufficient information in my knowledge base to answer that."
3. Do NOT fabricate statistics, names, policies, or figures.
4. Cite the source of your information (e.g. "According to the 2025 Budget Statement…").
5. Keep answers concise and factual."""

SYSTEM_V3 = """You are AcityBot, an AI assistant for Academic City University, Ghana.
You are an expert on Ghana's 2024 election results and the 2025 Budget Statement.

ANSWER PROTOCOL (follow these steps explicitly):
Step 1 — Identify which context passages are directly relevant to the question.
Step 2 — Extract key facts, figures, or policy statements from those passages.
Step 3 — Synthesise a clear, structured answer citing the source.
Step 4 — If the context is insufficient, explicitly state what is missing
         and avoid speculation.

HALLUCINATION CONTROLS:
• Never invent numbers, names, or policy details not present in context.
• If asked about something outside the election / budget domain, politely decline.
• Use hedging language ("According to…", "The data shows…") to ground claims.
• End with: "Sources: [list context IDs used]"."""


# ─────────────────────────────────────────────────────────────────────
#  PROMPT TEMPLATE BUILDER
# ─────────────────────────────────────────────────────────────────────

class PromptBuilder:
    """
    Assembles the final prompt sent to the LLM.
    Supports three prompt versions (V1 / V2 / V3).
    """

    VERSIONS = {
        "v1": SYSTEM_V1,
        "v2": SYSTEM_V2,
        "v3": SYSTEM_V3,
    }

    def __init__(self, version: str = "v2"):
        assert version in self.VERSIONS, f"Unknown version: {version}"
        self.version = version
        self.system_prompt = self.VERSIONS[version]

    def build(
        self,
        query: str,
        results: List[RetrievalResult],
        max_chars: int = MAX_CONTEXT_CHARS,
    ) -> Tuple[str, str, List[RetrievalResult]]:
        """
        Build the full prompt for the LLM.

        Returns:
            (system_prompt, user_message, selected_results)
        """
        selected, context_str = select_context(results, max_chars=max_chars)

        if not selected:
            context_str = "[No relevant context found in the knowledge base.]"

        if self.version == "v1":
            user_message = (
                f"Context:\n{context_str}\n\n"
                f"Question: {query}\n\n"
                f"Answer:"
            )
        elif self.version == "v2":
            user_message = (
                f"CONTEXT PASSAGES:\n"
                f"{'='*60}\n"
                f"{context_str}\n"
                f"{'='*60}\n\n"
                f"USER QUESTION: {query}\n\n"
                f"Provide a grounded, factual answer based solely on the context above."
            )
        else:   # v3 — chain-of-thought
            user_message = (
                f"CONTEXT PASSAGES:\n"
                f"{'='*60}\n"
                f"{context_str}\n"
                f"{'='*60}\n\n"
                f"USER QUESTION: {query}\n\n"
                f"Now apply the Answer Protocol:\n"
                f"Step 1 — Identify relevant passages:\n"
                f"Step 2 — Extract key facts:\n"
                f"Step 3 — Synthesise answer:\n"
                f"Step 4 — Note gaps (if any):\n\n"
                f"Final Answer:"
            )

        return self.system_prompt, user_message, selected

    # ──────────────────────────────────────────────────────────────────
    #  PROMPT EXPERIMENT HELPER  (Part C: compare versions)
    # ──────────────────────────────────────────────────────────────────

    @staticmethod
    def describe_versions() -> dict:
        return {
            "v1": {
                "label": "Naive Injection",
                "description": "Minimal system prompt; raw context injected with no structure.",
                "hallucination_control": "None",
                "expected_weakness": "LLM may confabulate when context is sparse.",
            },
            "v2": {
                "label": "Structured + Guard",
                "description": "Explicit rules: cite sources, refuse if unsure, no fabrication.",
                "hallucination_control": "Explicit refusal instruction + citation requirement",
                "expected_weakness": "May be overly cautious; breaks multi-hop reasoning.",
            },
            "v3": {
                "label": "Chain-of-Thought + Full Guard",
                "description": "Scaffolded reasoning steps force the model to ground each claim.",
                "hallucination_control": "CoT + citation + explicit gap acknowledgement",
                "expected_weakness": "Longer outputs; slower for simple factual queries.",
            },
        }
