#!/usr/bin/env python3
"""
extract.py — Extract text from a folder of PDFs into a structured JSON corpus.

Usage:
    python scripts/extract.py <pdf_folder> [--output-dir <dir>]

Outputs:
    corpus.json  — every page from every PDF
    sample.json  — first 10 pages (for quick review)
    Prints corpus statistics to stdout.
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path

import pdfplumber


# ---------------------------------------------------------------------------
# Text cleanup
# ---------------------------------------------------------------------------


def clean_text(text: str) -> str:
    """Clean raw PDF-extracted text.

    Handles:
      - Collapse runs of whitespace on a single line into one space
      - Rejoin lines that were broken mid-sentence (heuristic: if a line
        doesn't end with sentence-ending punctuation or a colon, join it
        to the next line)
      - Collapse 3+ consecutive blank lines into two
      - Strip leading/trailing whitespace
    """
    if not text:
        return ""

    # Normalize weird unicode spaces to regular spaces
    text = re.sub(r"[^\S\n]+", " ", text)

    # Rejoin lines broken mid-sentence:
    # If a line does NOT end with sentence-ending punctuation, a colon, or a
    # list bullet / number marker on the *next* line, merge it with the next.
    lines = text.split("\n")
    merged: list[str] = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            merged.append("")
            continue

        if (
            merged
            and merged[-1]  # previous line is non-empty
            and not re.search(r"[.!?:;)\]\"']$", merged[-1])  # doesn't end a sentence
            and not re.match(
                r"^\s*[-•●▪◦\d]+[.)]\s", stripped
            )  # next isn't a list item
            and not re.match(r"^[A-Z][A-Z\s]{4,}", stripped)  # next isn't a heading
        ):
            merged[-1] = merged[-1] + " " + stripped
        else:
            merged.append(stripped)

    text = "\n".join(merged)

    # Collapse 3+ blank lines into 2
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Collapse multiple spaces within lines (again, after merge)
    text = re.sub(r" {2,}", " ", text)

    return text.strip()


def strip_repeated_headers_footers(pages: list[dict]) -> list[dict]:
    """Remove repeated first/last lines across pages (likely headers/footers).

    Heuristic: if the same trimmed line appears as the first (or last) line on
    more than half the pages, it's probably a header (or footer).
    """
    if len(pages) < 4:
        return pages  # not enough pages to detect patterns

    threshold = len(pages) * 0.5

    # Gather first and last non-empty lines
    first_lines: dict[str, int] = {}
    last_lines: dict[str, int] = {}

    for p in pages:
        lines = [l.strip() for l in p["text"].split("\n") if l.strip()]
        if lines:
            first_lines[lines[0]] = first_lines.get(lines[0], 0) + 1
            last_lines[lines[-1]] = last_lines.get(lines[-1], 0) + 1

    headers = {line for line, count in first_lines.items() if count >= threshold}
    footers = {line for line, count in last_lines.items() if count >= threshold}

    if not headers and not footers:
        return pages

    cleaned = []
    for p in pages:
        lines = p["text"].split("\n")

        # Strip header
        while lines:
            if lines[0].strip() in headers:
                lines.pop(0)
            else:
                break

        # Strip footer
        while lines:
            if lines[-1].strip() in footers:
                lines.pop()
            else:
                break

        cleaned.append({**p, "text": "\n".join(lines).strip()})

    return cleaned


# ---------------------------------------------------------------------------
# PDF extraction
# ---------------------------------------------------------------------------


def extract_pdf(pdf_path: Path) -> list[dict]:
    """Extract all pages from a single PDF, returning a list of page dicts."""
    pages = []
    filename = pdf_path.name

    try:
        with pdfplumber.open(pdf_path) as pdf:
            for i, page in enumerate(pdf.pages, start=1):
                raw = page.extract_text() or ""
                text = clean_text(raw)
                pages.append(
                    {
                        "source": filename,
                        "page": i,
                        "char_count": len(text),
                        "text": text,
                    }
                )
    except Exception as e:
        print(f"  ⚠  Failed to read {filename}: {e}", file=sys.stderr)

    return pages


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

NEAR_EMPTY_THRESHOLD = 50


def print_stats(corpus: list[dict]) -> None:
    sources = {p["source"] for p in corpus}
    total_pages = len(corpus)
    total_chars = sum(p["char_count"] for p in corpus)
    avg_chars = total_chars / total_pages if total_pages else 0
    empty_pages = sum(1 for p in corpus if p["char_count"] < NEAR_EMPTY_THRESHOLD)

    print("\n── Corpus Statistics ─────────────────────")
    print(f"  Documents (PDFs):          {len(sources)}")
    print(f"  Total pages:               {total_pages}")
    print(f"  Total characters:          {total_chars:,}")
    print(f"  Avg characters per page:   {avg_chars:,.0f}")
    print(
        f"  Empty / near-empty pages:  {empty_pages}  (< {NEAR_EMPTY_THRESHOLD} chars)"
    )
    print("──────────────────────────────────────────\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract text from a folder of PDFs into corpus.json"
    )
    parser.add_argument(
        "pdf_folder",
        type=Path,
        help="Path to folder containing PDF files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for output files (default: same as pdf_folder)",
    )
    args = parser.parse_args()

    pdf_folder: Path = args.pdf_folder.resolve()
    output_dir: Path = (args.output_dir or pdf_folder.parent).resolve()

    if not pdf_folder.is_dir():
        print(f"Error: {pdf_folder} is not a directory.", file=sys.stderr)
        sys.exit(1)

    os.makedirs(output_dir, exist_ok=True)

    pdf_files = sorted(pdf_folder.glob("*.pdf"))
    if not pdf_files:
        print(f"No PDF files found in {pdf_folder}.", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(pdf_files)} PDF(s) in {pdf_folder}\n")

    corpus: list[dict] = []

    for pdf_path in pdf_files:
        print(f"  Extracting: {pdf_path.name} …", end=" ", flush=True)
        pages = extract_pdf(pdf_path)
        # Remove repeated headers/footers within each document
        pages = strip_repeated_headers_footers(pages)
        # Re-compute char_count after header/footer removal
        for p in pages:
            p["char_count"] = len(p["text"])
        print(f"{len(pages)} pages")
        corpus.extend(pages)

    # Write corpus.json
    corpus_path = output_dir / "corpus.json"
    with open(corpus_path, "w", encoding="utf-8") as f:
        json.dump(corpus, f, ensure_ascii=False, indent=2)
    print(f"\nSaved full corpus  → {corpus_path}")

    # Write sample.json (first 10 pages)
    sample = corpus[:10]
    sample_path = output_dir / "sample.json"
    with open(sample_path, "w", encoding="utf-8") as f:
        json.dump(sample, f, ensure_ascii=False, indent=2)
    print(f"Saved sample       → {sample_path}")

    print_stats(corpus)


if __name__ == "__main__":
    main()
