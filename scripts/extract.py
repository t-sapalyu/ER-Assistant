import json
import fitz
import re
from pathlib import Path
from collections import Counter

INPUT_DIR = "data/pdfs"
OUTPUT_CORPUS = "data/corpus.json"
OUTPUT_SAMPLE = "data/sample.json"
REPORT_FILE = "scripts/EXTRACTION_REPORT.md"

EMPTY_THRESHOLD = 50


def clean_text(text):
    # Normalize line endings
    text = text.replace('\r', '\n')

    # Fix broken lines inside sentences
    text = re.sub(r'\n(?=[a-z])', ' ', text)

    # Normalize bullets
    text = text.replace("•", "- ")
    text = re.sub(r'-\s*\n', '- ', text)

    # Reduce excessive newlines
    text = re.sub(r'\n{2,}', '\n', text)

    # Normalize spaces
    text = re.sub(r'[ \t]+', ' ', text)

    return text.strip()


def detect_repeated_lines(all_pages_text, threshold_ratio=0.3):
    all_lines = []

    for text in all_pages_text:
        lines = [line.strip() for line in text.split("\n") if line.strip()]
        all_lines.extend(lines)

    line_counts = Counter(all_lines)
    total_pages = len(all_pages_text)

    repeated_lines = {
        line for line, count in line_counts.items()
        if count > total_pages * threshold_ratio and len(line) < 200
    }

    return repeated_lines


def remove_headers_footers(text, repeated_lines):
    lines = text.split("\n")

    cleaned_lines = [
        line for line in lines
        if line.strip() not in repeated_lines
    ]

    return "\n".join(cleaned_lines)


def process_pdf(pdf_path):
    doc = fitz.open(pdf_path)

    raw_pages = []

    # Step 1: Extract raw text
    for page in doc:
        raw_text = page.get_text("text")
        raw_pages.append(raw_text)

    # Step 2: Detect repeated lines across pages
    repeated_lines = detect_repeated_lines(raw_pages)

    results = []

    # Step 3: Clean + remove headers
    for page_num, raw_text in enumerate(raw_pages, start=1):
        text = clean_text(raw_text)
        text = remove_headers_footers(text, repeated_lines)

        char_count = len(text)

        results.append({
            "source": pdf_path.name,
            "page": page_num,
            "char_count": char_count,
            "text": text
        })

    return results


def generate_report(pdf_count, total_pages, total_chars, avg_chars, empty_pages, empty_page_details):
    lines = []

    lines.append("# Extraction Report\n")
    lines.append("## Summary\n")
    lines.append(f"- Number of PDFs: {pdf_count}")
    lines.append(f"- Total pages: {total_pages}")
    lines.append(f"- Total characters: {total_chars}")
    lines.append(f"- Average characters per page: {avg_chars:.2f}")
    lines.append(f"- Empty or near-empty pages (<{EMPTY_THRESHOLD} chars): {empty_pages}\n")

    lines.append("## Empty / Near-Empty Pages\n")

    if not empty_page_details:
        lines.append("None \n")
    else:
        lines.append("| Source | Page | Character Count |")
        lines.append("|--------|------|-----------------|")

        for item in empty_page_details:
            lines.append(f"| {item['source']} | {item['page']} | {item['char_count']} |")

    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main():
    all_data = []

    total_pages = 0
    total_chars = 0
    empty_pages = 0
    pdf_count = 0

    empty_page_details = []

    pdf_files = list(Path(INPUT_DIR).glob("*.pdf"))

    if not pdf_files:
        print(f"No PDF files found in '{INPUT_DIR}'")
        return

    for pdf_file in pdf_files:
        print(f"Processing: {pdf_file.name}")
        pdf_count += 1

        pages = process_pdf(pdf_file)

        for entry in pages:
            total_pages += 1
            total_chars += entry["char_count"]

            if entry["char_count"] < EMPTY_THRESHOLD:
                empty_pages += 1
                empty_page_details.append({
                    "source": entry["source"],
                    "page": entry["page"],
                    "char_count": entry["char_count"]
                })

        all_data.extend(pages)

    # Save corpus
    with open(OUTPUT_CORPUS, "w", encoding="utf-8") as f:
        json.dump(all_data, f, ensure_ascii=False, indent=2)

    # Save sample
    with open(OUTPUT_SAMPLE, "w", encoding="utf-8") as f:
        json.dump(all_data[:10], f, ensure_ascii=False, indent=2)

    avg_chars = total_chars / total_pages if total_pages else 0

    print("\n=== Corpus Stats ===")
    print(f"Number of PDFs: {pdf_count}")
    print(f"Total pages: {total_pages}")
    print(f"Total characters: {total_chars}")
    print(f"Average characters/page: {avg_chars:.2f}")
    print(f"Empty/near-empty pages (<{EMPTY_THRESHOLD} chars): {empty_pages}")

    generate_report(
        pdf_count,
        total_pages,
        total_chars,
        avg_chars,
        empty_pages,
        empty_page_details
    )

    print(f"\nReport saved to: {REPORT_FILE}")
    print(f"Corpus saved to: {OUTPUT_CORPUS}")
    print(f"Sample saved to: {OUTPUT_SAMPLE}")


if __name__ == "__main__":
    main()