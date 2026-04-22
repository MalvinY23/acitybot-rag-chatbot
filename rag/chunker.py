"""
RAG Chunker — Part A Implementation
====================================
Three chunking strategies implemented manually:

STRATEGY 1 — Fixed-Size with Overlap  (default for PDF)
  Chunk size : 512 characters  (~100–130 tokens)
  Overlap    : 100 characters  (~20% overlap)
  Justification:
    • Budget PDF has dense prose. A 512-char window (~2–3 sentences) captures
      a coherent policy statement without exceeding typical embedding-model
      context limits (256–512 tokens for MiniLM).
    • 100-char overlap prevents context from being cut at sentence boundaries,
      which would otherwise split a fiscal figure from its explanation.
    • Tested against 256 / 512 / 1024 — 512 gave best Precision@3 in our
      experiment log (see experiment_logs/).

STRATEGY 2 — Sentence-Aware (paragraph-level) Chunking
  Splits on paragraph breaks (double newlines), then merges short paragraphs
  until the chunk reaches ~400–600 characters.
  Justification:
    • Preserves semantic units (one policy → one chunk).
    • Better for Q&A when the answer is a complete paragraph.

STRATEGY 3 — Row-Level (for CSV / structured data)
  Each CSV row → one chunk (already done in data_loader, kept separate here
  so the impact can be compared with grouping N rows into one chunk).
  We also provide N-row grouping (N=5) for comparison.
"""

import re
from dataclasses import dataclass, field
from typing import List, Tuple
from .data_loader import RawDocument


@dataclass
class Chunk:
    """A text chunk ready for embedding."""
    chunk_id: str
    doc_id: str
    source: str
    source_name: str
    text: str
    metadata: dict = field(default_factory=dict)
    char_start: int = 0
    char_end: int = 0
    strategy: str = ""


# ─────────────────────────────────────────────────────────────────────
#  STRATEGY 1 — Fixed-Size with Character Overlap
# ─────────────────────────────────────────────────────────────────────

def chunk_fixed_size(
    doc: RawDocument,
    chunk_size: int = 512,
    overlap: int = 100,
) -> List[Chunk]:
    """
    Slide a window of `chunk_size` characters over the document text,
    stepping by (chunk_size - overlap) each time.
    """
    text = doc.content
    step = chunk_size - overlap
    chunks: List[Chunk] = []
    idx = 0
    i = 0

    while idx < len(text):
        end = min(idx + chunk_size, len(text))
        chunk_text = text[idx:end].strip()

        if len(chunk_text) > 20:    # skip trivially short trailing chunks
            chunks.append(Chunk(
                chunk_id=f"{doc.doc_id}_fixed_{i}",
                doc_id=doc.doc_id,
                source=doc.source,
                source_name=doc.source_name,
                text=chunk_text,
                metadata={**doc.metadata, "chunk_index": i},
                char_start=idx,
                char_end=end,
                strategy="fixed_size",
            ))
        idx += step
        i += 1

    return chunks


# ─────────────────────────────────────────────────────────────────────
#  STRATEGY 2 — Sentence/Paragraph-Aware Chunking
# ─────────────────────────────────────────────────────────────────────

def _split_into_sentences(text: str) -> List[str]:
    """Naive sentence splitter using regex (no NLTK dependency)."""
    # Split after . ! ? followed by whitespace or end, but not abbreviations
    parts = re.split(r"(?<=[.!?])\s+", text)
    return [p.strip() for p in parts if p.strip()]


def chunk_paragraph_aware(
    doc: RawDocument,
    target_size: int = 450,
    min_size: int = 80,
) -> List[Chunk]:
    """
    Split on paragraph boundaries (\\n\\n), then greedily merge short
    paragraphs until the combined length is ≥ target_size.
    """
    paragraphs = re.split(r"\n{2,}", doc.content)
    paragraphs = [p.strip() for p in paragraphs if p.strip()]

    chunks: List[Chunk] = []
    buffer = ""
    chunk_idx = 0

    for para in paragraphs:
        if len(buffer) + len(para) < target_size:
            buffer = (buffer + " " + para).strip()
        else:
            # Flush current buffer
            if len(buffer) >= min_size:
                chunks.append(Chunk(
                    chunk_id=f"{doc.doc_id}_para_{chunk_idx}",
                    doc_id=doc.doc_id,
                    source=doc.source,
                    source_name=doc.source_name,
                    text=buffer,
                    metadata={**doc.metadata, "chunk_index": chunk_idx},
                    strategy="paragraph_aware",
                ))
                chunk_idx += 1
            buffer = para

    # Final buffer
    if len(buffer) >= min_size:
        chunks.append(Chunk(
            chunk_id=f"{doc.doc_id}_para_{chunk_idx}",
            doc_id=doc.doc_id,
            source=doc.source,
            source_name=doc.source_name,
            text=buffer,
            metadata={**doc.metadata, "chunk_index": chunk_idx},
            strategy="paragraph_aware",
        ))

    return chunks


# ─────────────────────────────────────────────────────────────────────
#  STRATEGY 3 — Row-Level (CSV rows grouped by N)
# ─────────────────────────────────────────────────────────────────────

def chunk_csv_rows(
    docs: List[RawDocument],
    group_size: int = 1,
) -> List[Chunk]:
    """
    For CSV documents: group `group_size` rows into one chunk.
    group_size=1 keeps individual rows (best precision).
    group_size=5 creates broader thematic chunks (better recall for multi-row queries).
    """
    chunks: List[Chunk] = []
    group: List[RawDocument] = []
    group_idx = 0

    for doc in docs:
        if doc.source != "csv":
            continue
        group.append(doc)
        if len(group) == group_size:
            combined_text = "\n".join(d.content for d in group)
            combined_meta = {
                "rows": [d.doc_id for d in group],
                "chunk_index": group_idx,
            }
            chunks.append(Chunk(
                chunk_id=f"csv_group_{group_idx}",
                doc_id=f"csv_group_{group_idx}",
                source="csv",
                source_name="Ghana Election Results",
                text=combined_text,
                metadata=combined_meta,
                strategy=f"row_group_{group_size}",
            ))
            group = []
            group_idx += 1

    # Remaining rows
    if group:
        combined_text = "\n".join(d.content for d in group)
        chunks.append(Chunk(
            chunk_id=f"csv_group_{group_idx}",
            doc_id=f"csv_group_{group_idx}",
            source="csv",
            source_name="Ghana Election Results",
            text=combined_text,
            metadata={"rows": [d.doc_id for d in group], "chunk_index": group_idx},
            strategy=f"row_group_{group_size}",
        ))

    return chunks


# ─────────────────────────────────────────────────────────────────────
#  MAIN CHUNKING ENTRY POINT
# ─────────────────────────────────────────────────────────────────────

def chunk_documents(
    documents: List[RawDocument],
    pdf_strategy: str = "fixed",       # "fixed" | "paragraph"
    csv_group_size: int = 1,
    chunk_size: int = 512,
    overlap: int = 100,
) -> List[Chunk]:
    """
    Route each document to the appropriate chunking strategy:
      • PDF pages  → fixed-size or paragraph-aware
      • CSV rows   → row-level grouping
    """
    all_chunks: List[Chunk] = []

    csv_docs  = [d for d in documents if d.source == "csv"]
    pdf_docs  = [d for d in documents if d.source == "pdf"]

    # ── CSV ────────────────────────────────────────────────────────────
    all_chunks.extend(chunk_csv_rows(csv_docs, group_size=csv_group_size))

    # ── PDF ────────────────────────────────────────────────────────────
    for doc in pdf_docs:
        if pdf_strategy == "paragraph":
            all_chunks.extend(chunk_paragraph_aware(doc))
        else:
            all_chunks.extend(chunk_fixed_size(doc, chunk_size, overlap))

    print(f"[Chunker] Strategy={pdf_strategy}, CSV group={csv_group_size}  "
          f"→  {len(all_chunks)} total chunks  "
          f"(CSV: {len([c for c in all_chunks if c.source=='csv'])}, "
          f"PDF: {len([c for c in all_chunks if c.source=='pdf'])})")
    return all_chunks


# ─────────────────────────────────────────────────────────────────────
#  COMPARATIVE ANALYSIS HELPER
# ─────────────────────────────────────────────────────────────────────

def compare_chunking_strategies(documents: List[RawDocument]) -> dict:
    """
    Run all three strategies and return comparison statistics.
    Used in experiment logging (Part A deliverable).
    """
    results = {}

    for strategy, kwargs in [
        ("fixed_512_ov100",    {"pdf_strategy": "fixed",     "chunk_size": 512, "overlap": 100}),
        ("fixed_256_ov50",     {"pdf_strategy": "fixed",     "chunk_size": 256, "overlap": 50}),
        ("fixed_1024_ov200",   {"pdf_strategy": "fixed",     "chunk_size": 1024,"overlap": 200}),
        ("paragraph_aware",    {"pdf_strategy": "paragraph"}),
        ("csv_row_group_1",    {"csv_group_size": 1}),
        ("csv_row_group_5",    {"csv_group_size": 5}),
    ]:
        chunks = chunk_documents(documents, **kwargs)
        texts  = [c.text for c in chunks]
        lengths = [len(t) for t in texts]
        results[strategy] = {
            "total_chunks": len(chunks),
            "avg_length":   round(sum(lengths) / len(lengths), 1) if lengths else 0,
            "min_length":   min(lengths) if lengths else 0,
            "max_length":   max(lengths) if lengths else 0,
        }

    return results
