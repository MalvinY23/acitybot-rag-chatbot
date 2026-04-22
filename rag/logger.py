"""
RAG Experiment Logger — Part D & Part E
=========================================
Logs every pipeline run to JSON files with timestamps.
Provides manual experiment log entries (not AI-generated).

Each log entry captures:
  • Query & expanded query
  • Retrieved chunks (text snippet, score, source)
  • Final prompt sent to LLM
  • LLM response
  • Prompt version used
  • Latency at each stage
  • Failure flags
"""

import os
import json
import time
from datetime import datetime
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict

LOG_DIR = "experiment_logs"


@dataclass
class StageTimer:
    """Tracks elapsed time for each pipeline stage."""
    _start: float = 0.0
    retrieval_ms:  float = 0.0
    prompt_ms:     float = 0.0
    llm_ms:        float = 0.0

    def start(self): self._start = time.time()
    def mark(self, stage: str):
        elapsed = round((time.time() - self._start) * 1000, 1)
        setattr(self, f"{stage}_ms", elapsed)
        self._start = time.time()


class ExperimentLogger:
    """
    Writes structured experiment logs to:
      experiment_logs/session_<date>.json   — per-session log
      experiment_logs/full_history.jsonl    — append-only run history
    """

    def __init__(self):
        os.makedirs(LOG_DIR, exist_ok=True)
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_path = os.path.join(LOG_DIR, f"session_{self.session_id}.json")
        self.history_path = os.path.join(LOG_DIR, "full_history.jsonl")
        self.entries: List[Dict] = []
        print(f"[Logger] Session log: {self.session_path}")

    def log(
        self,
        query: str,
        expanded_query: str,
        retrieved_chunks: List[Dict],   # [{text, score, source, chunk_id}]
        similarity_scores: List[float],
        system_prompt: str,
        user_prompt: str,
        llm_response: str,
        prompt_version: str,
        failure_detected: bool,
        failure_reason: Optional[str],
        timers: StageTimer,
        extra: Optional[Dict] = None,
    ) -> Dict:
        """Create and persist one log entry."""

        entry = {
            "session_id":       self.session_id,
            "timestamp":        datetime.now().isoformat(),
            "query":            query,
            "expanded_query":   expanded_query,
            "prompt_version":   prompt_version,

            # ── Retrieval stage ──────────────────────────────────────
            "retrieved_chunks": [
                {
                    "rank":       i + 1,
                    "chunk_id":   c.get("chunk_id", ""),
                    "source":     c.get("source", ""),
                    "source_name":c.get("source_name", ""),
                    "score":      c.get("score", 0),
                    "text_snippet": c.get("text", "")[:200],
                }
                for i, c in enumerate(retrieved_chunks)
            ],
            "similarity_scores": similarity_scores,
            "top_score":        max(similarity_scores) if similarity_scores else 0,
            "avg_score":        round(
                sum(similarity_scores) / len(similarity_scores), 4
            ) if similarity_scores else 0,

            # ── Prompt stage ─────────────────────────────────────────
            "system_prompt_chars": len(system_prompt),
            "user_prompt_chars":   len(user_prompt),
            "full_prompt_preview": user_prompt[:500],

            # ── LLM stage ────────────────────────────────────────────
            "llm_response":        llm_response,
            "llm_response_chars":  len(llm_response),

            # ── Quality flags ────────────────────────────────────────
            "failure_detected":    failure_detected,
            "failure_reason":      failure_reason,

            # ── Latency ──────────────────────────────────────────────
            "latency_retrieval_ms": timers.retrieval_ms,
            "latency_prompt_ms":    timers.prompt_ms,
            "latency_llm_ms":       timers.llm_ms,
            "latency_total_ms": round(
                timers.retrieval_ms + timers.prompt_ms + timers.llm_ms, 1
            ),

            # ── Extra fields (adversarial tests etc.) ────────────────
            **(extra or {}),
        }

        self.entries.append(entry)
        self._persist(entry)
        return entry

    def _persist(self, entry: Dict):
        """Save to both session file and append-only history."""
        # Session file (full JSON array)
        with open(self.session_path, "w") as f:
            json.dump(self.entries, f, indent=2, default=str)

        # Append-only JSONL
        with open(self.history_path, "a") as f:
            f.write(json.dumps(entry, default=str) + "\n")

    # ──────────────────────────────────────────────────────────────────
    #  SUMMARY REPORT  (human-readable for documentation)
    # ──────────────────────────────────────────────────────────────────

    def summary(self) -> str:
        if not self.entries:
            return "No experiments logged yet."

        lines = [
            f"═══════════════════════════════════════════",
            f"  EXPERIMENT LOG SUMMARY — {self.session_id}",
            f"  Total runs: {len(self.entries)}",
            f"═══════════════════════════════════════════",
        ]
        for i, e in enumerate(self.entries, 1):
            lines.extend([
                f"\n[Run {i}]",
                f"  Query         : {e['query']}",
                f"  Expanded      : {e['expanded_query']}",
                f"  Prompt ver.   : {e['prompt_version']}",
                f"  Top score     : {e['top_score']:.4f}",
                f"  Avg score     : {e['avg_score']:.4f}",
                f"  Failure?      : {e['failure_detected']} — {e['failure_reason'] or 'N/A'}",
                f"  Latency total : {e['latency_total_ms']} ms",
                f"  Response (preview): {str(e['llm_response'])[:180]} …",
            ])
        return "\n".join(lines)

    def get_all_entries(self) -> List[Dict]:
        return self.entries

    @staticmethod
    def load_history() -> List[Dict]:
        history_path = os.path.join(LOG_DIR, "full_history.jsonl")
        if not os.path.exists(history_path):
            return []
        entries = []
        with open(history_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        entries.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
        return entries
