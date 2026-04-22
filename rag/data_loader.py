"""
RAG Data Loader
Handles ingestion of CSV (Ghana Election Results) and PDF (2025 Budget Statement).
No third-party RAG frameworks used — all manual implementation.
"""

import os
import re
import io
import requests
import pandas as pd
import pdfplumber
from typing import List, Dict, Any
from dataclasses import dataclass


@dataclass
class RawDocument:
    """Represents a cleaned, raw document before chunking."""
    doc_id: str
    source: str          # 'csv' | 'pdf'
    source_name: str     # human-readable source label
    content: str
    metadata: Dict[str, Any]


# ─────────────────────────────────────────────────────────────────────
#  CSV LOADER  (Ghana Election Results)
# ─────────────────────────────────────────────────────────────────────

CSV_URL = (
    "https://raw.githubusercontent.com/GodwinDansoAcity/acitydataset/"
    "main/Ghana_Election_Result.csv"
)

def load_election_csv(local_path: str = "data/Ghana_Election_Result.csv") -> List[RawDocument]:
    """
    Load and clean the Ghana Election Results CSV.
    Each row becomes one RawDocument, with key fields serialised to plain text
    so the embedding model can process them uniformly.
    """
    # ── Download if not cached ──────────────────────────────────────────
    if not os.path.exists(local_path):
        print(f"[DataLoader] Downloading election CSV from GitHub …")
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        r = requests.get(CSV_URL, timeout=30)
        r.raise_for_status()
        with open(local_path, "wb") as f:
            f.write(r.content)

    df = pd.read_csv(local_path, encoding="utf-8", on_bad_lines="skip")

    # ── Cleaning ────────────────────────────────────────────────────────
    # 1. Strip leading/trailing whitespace from all string columns
    df = df.apply(lambda col: col.map(lambda x: x.strip() if isinstance(x, str) else x))

    # 2. Drop fully-empty rows
    df.dropna(how="all", inplace=True)

    # 3. Normalise column names
    df.columns = [
        re.sub(r"\s+", "_", col.strip().lower()) for col in df.columns
    ]

    # 4. Fill NaN numerics with 0, strings with "Unknown"
    for col in df.columns:
        if df[col].dtype == object:
            df[col].fillna("Unknown", inplace=True)
        else:
            df[col].fillna(0, inplace=True)

    # 5. Remove duplicate rows
    before = len(df)
    df.drop_duplicates(inplace=True)
    after = len(df)
    print(f"[DataLoader] CSV cleaned: {before} → {after} rows after dedup.")

    # ── Convert rows to RawDocument ─────────────────────────────────────
    documents: List[RawDocument] = []
    for i, row in df.iterrows():
        # Build a natural-language sentence from the row
        parts = []
        for col, val in row.items():
            label = col.replace("_", " ").title()
            parts.append(f"{label}: {val}")
        content = " | ".join(parts)

        documents.append(RawDocument(
            doc_id=f"csv_{i}",
            source="csv",
            source_name="Ghana Election Results",
            content=content,
            metadata=row.to_dict(),
        ))

    print(f"[DataLoader] {len(documents)} election documents loaded.")
    return documents


# ─────────────────────────────────────────────────────────────────────
#  PDF LOADER  (2025 Budget Statement)
# ─────────────────────────────────────────────────────────────────────

PDF_URL = (
    "https://mofep.gov.gh/sites/default/files/budget-statements/"
    "2025-Budget-Statement-and-Economic-Policy_v4.pdf"
)

def load_budget_pdf(local_path: str = "data/2025_Budget.pdf") -> List[RawDocument]:
    """
    Download (if needed) and extract text from the 2025 Ghana Budget PDF.
    Returns one RawDocument per page, preserving page-level metadata.
    """
    if not os.path.exists(local_path):
        print(f"[DataLoader] Downloading budget PDF …")
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        r = requests.get(PDF_URL, timeout=120, stream=True)
        r.raise_for_status()
        with open(local_path, "wb") as f:
            for chunk in r.iter_content(8192):
                f.write(chunk)
        print(f"[DataLoader] PDF saved to {local_path}")

    documents: List[RawDocument] = []
    with pdfplumber.open(local_path) as pdf:
        total = len(pdf.pages)
        print(f"[DataLoader] Extracting text from {total} PDF pages …")
        for page_num, page in enumerate(pdf.pages, start=1):
            raw_text = page.extract_text() or ""

            # ── Cleaning ───────────────────────────────────────────────
            # Remove hyphenation at line ends
            raw_text = re.sub(r"-\n", "", raw_text)
            # Collapse excessive whitespace
            raw_text = re.sub(r"[ \t]+", " ", raw_text)
            # Remove repeated newlines → single newline
            raw_text = re.sub(r"\n{3,}", "\n\n", raw_text)
            # Remove non-printable characters
            raw_text = re.sub(r"[^\x20-\x7E\n]", " ", raw_text)
            raw_text = raw_text.strip()

            if len(raw_text) < 30:      # skip near-empty pages (headers only)
                continue

            documents.append(RawDocument(
                doc_id=f"pdf_p{page_num}",
                source="pdf",
                source_name="2025 Ghana Budget Statement",
                content=raw_text,
                metadata={"page": page_num, "total_pages": total},
            ))

    print(f"[DataLoader] {len(documents)} PDF page documents loaded.")
    return documents


# ─────────────────────────────────────────────────────────────────────
#  UNIFIED LOADER
# ─────────────────────────────────────────────────────────────────────

def load_all_documents() -> List[RawDocument]:
    """Load both data sources and return combined document list."""
    election_docs = load_election_csv()
    budget_docs   = load_budget_pdf()
    all_docs = election_docs + budget_docs
    print(f"[DataLoader] Total raw documents: {len(all_docs)}")
    return all_docs
