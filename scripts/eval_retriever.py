"""Retrieval-only evaluation: dense vs BM25 vs hybrid (no LLM calls).

For every question with a labelled page, checks where the correct page lands
in the retrieved list:
  Hit@k     — is any correct page in the top k?        (1 or 0, averaged)
  Recall@10 — what fraction of the correct pages is in the top 10?
  MRR       — 1 / rank of the first correct page (0 if not found)

Questions without a labelled page (out-of-scope, ambiguous, unanswerable) are
reported separately: their best cosine score shows how strongly irrelevant
chunks still match.

Query embeddings are cached in data/query_embedding_cache.json, so only the
first run calls the embeddings API; reruns and the sweep are free.

Usage:
    python scripts/eval_retriever.py                    # compare the 3 modes
    python scripts/eval_retriever.py --sweep            # + hybrid settings sweep
    python scripts/eval_retriever.py --compare-backup   # + dense on the old misaligned index
"""

import argparse
import json
import logging
import re
import sys
from pathlib import Path

import faiss

sys.path.insert(0, str(Path(__file__).parent))

import rag_pipeline as rp

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
CACHE_PATH = DATA_DIR / "query_embedding_cache.json"
RESULTS_PATH = DATA_DIR / "retrieval_eval_results.json"
BACKUP_INDEX_PATH = DATA_DIR / "my_index_misaligned_backup.faiss"

MODES = ("dense", "bm25", "hybrid")
HIT_KS = (1, 3, 5, 10)
REFUSAL_THRESHOLD = 0.35  # same cut-off eval_runner uses for "correct_refusal"


# ---------------------------
# Embedding cache
# ---------------------------
def install_embedding_cache():
    """Wrap rp.get_embeddings so each question is embedded once, ever."""
    cache = json.loads(CACHE_PATH.read_text(encoding="utf-8")) if CACHE_PATH.exists() else {}
    original = rp.get_embeddings
    stats = {"hits": 0, "misses": 0}

    def cached(texts):
        missing = [t for t in texts if t not in cache]
        if missing:
            vectors = original(missing)
            if len(vectors) != len(missing):
                raise RuntimeError("Embedding API failed; cannot run the eval.")
            cache.update(zip(missing, vectors))
            CACHE_PATH.write_text(json.dumps(cache), encoding="utf-8")
        stats["misses"] += len(missing)
        stats["hits"] += len(texts) - len(missing)
        return [cache[t] for t in texts]

    rp.get_embeddings = cached  # retrieve() looks this name up at call time
    return stats


# ---------------------------
# Data helpers
# ---------------------------
def normalize_category(name: str) -> str:
    """'Yes/No', 'yes_no' -> 'yes_no'; 'Multi-hop reasoning' -> 'multi_hop_reasoning'."""
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def gold_pages(q) -> set:
    """Set of (source, page) pairs that count as correct; empty if unlabelled."""
    if q.get("page") is None or not q.get("source"):
        return set()
    pages = q["page"] if isinstance(q["page"], list) else [q["page"]]
    return {(q["source"], p) for p in pages}


# ---------------------------
# Metrics
# ---------------------------
def score_question(retrieved, gold):
    """Metrics for one question from one retrieved list."""
    keys = [(c["source"], c["page"]) for c in retrieved]
    first = next((i + 1 for i, key in enumerate(keys) if key in gold), None)
    result = {
        "pages": [c["page"] for c in retrieved],
        "first_rank": first,
        "mrr": 1.0 / first if first else 0.0,
        "recall@10": len(gold & set(keys[:10])) / len(gold),
    }
    for k in HIT_KS:
        result[f"hit@{k}"] = int(first is not None and first <= k)
    return result


def average(rows, metric):
    return sum(r[metric] for r in rows) / len(rows) if rows else 0.0


def run_mode(questions, index, chunks, mode, k):
    """Retrieve for every question in one mode; returns per-question records."""
    records = []
    for q in questions:
        retrieved = rp.retrieve(q["question"], index, chunks, k=k, mode=mode)
        gold = gold_pages(q)
        record = {
            "question": q["question"],
            "category": normalize_category(q.get("category", "unknown")),
            "gold": sorted(p for _, p in gold),
            "best_cosine": round(rp.best_score(retrieved), 4),
            "n_results": len(retrieved),
        }
        if gold:
            record.update(score_question(retrieved, gold))
        else:
            record["pages"] = [c["page"] for c in retrieved]
        records.append(record)
    return records


def summarize(records):
    labelled = [r for r in records if r["gold"]]
    summary = {m: round(average(labelled, m), 3) for m in [f"hit@{k}" for k in HIT_KS] + ["recall@10", "mrr"]}
    summary["n"] = len(labelled)
    return summary


# ---------------------------
# Printing
# ---------------------------
def print_summary(results):
    cols = [f"hit@{k}" for k in HIT_KS] + ["recall@10", "mrr"]
    print(f"\n{'=' * 72}\nSUMMARY ({results[MODES[0]]['summary']['n']} labelled questions)\n{'=' * 72}")
    print(f"{'mode':<10}" + "".join(f"{c:>10}" for c in cols))
    for mode, data in results.items():
        s = data["summary"]
        print(f"{mode:<10}" + "".join(f"{s[c]:>10.3f}" for c in cols))


def print_categories(results):
    print(f"\n{'=' * 72}\nBY CATEGORY (hit@10 / mrr)\n{'=' * 72}")
    categories = sorted({r["category"] for r in results["dense"]["records"] if r["gold"]})
    print(f"{'category':<34}{'n':>3}" + "".join(f"{m:>12}" for m in MODES))
    for cat in categories:
        cells = []
        for mode in MODES:
            rows = [r for r in results[mode]["records"] if r["gold"] and r["category"] == cat]
            cells.append(f"{average(rows, 'hit@10'):.2f}/{average(rows, 'mrr'):.2f}")
        print(f"{cat:<34}{len(rows):>3}" + "".join(f"{c:>12}" for c in cells))


def print_wins_losses(results):
    """Per question: did hybrid rank the first correct page higher or lower than dense?"""
    print(f"\n{'=' * 72}\nHYBRID vs DENSE - first correct page rank (lower is better)\n{'=' * 72}")
    wins = losses = ties = 0
    for d, h in zip(results["dense"]["records"], results["hybrid"]["records"]):
        if not d["gold"]:
            continue
        dr = d["first_rank"] or 99  # 99 = not found in top k
        hr = h["first_rank"] or 99
        if dr == hr:
            ties += 1
            continue
        tag = "WIN " if hr < dr else "LOSS"
        wins += hr < dr
        losses += hr > dr
        show = lambda r: str(r) if r != 99 else "-"
        print(f"  {tag} dense={show(dr):>2} hybrid={show(hr):>2}  [{d['category']}] {d['question'][:60]}")
    print(f"\n  wins={wins}  losses={losses}  ties={ties}")


def print_unlabelled(results):
    print(f"\n{'=' * 72}\nNO-ANSWER QUESTIONS - best cosine (lower is better; <{REFUSAL_THRESHOLD} = treated as refusal)\n{'=' * 72}")
    rows = [list(group) for group in zip(*(results[m]["records"] for m in MODES))]
    for group in rows:
        if group[0]["gold"]:
            continue
        cells = "".join(f"{r['best_cosine']:>9.3f}" for r in group)
        print(f"  [{group[0]['category']:<14}]{cells}   {group[0]['question'][:50]}")
    print(f"  {'':<16}" + "".join(f"{m:>9}" for m in MODES))


# ---------------------------
# Sweep
# ---------------------------
def run_sweep(questions, index, chunks):
    """Try hybrid settings; module constants are read at call time, so we set them directly."""
    print(f"\n{'=' * 72}\nHYBRID SETTINGS SWEEP\n{'=' * 72}")
    print(f"{'rrf_k':>6}{'pool':>6}{'k':>4}{'hit@1':>9}{'hit@3':>9}{'hit@10':>9}{'recall@10':>11}{'mrr':>8}")
    defaults = (rp.RRF_K, rp.CANDIDATE_POOL)
    rows = []
    for rrf_k in (10, 30, 60):
        for pool in (20, 50):
            for k in (5, 10):
                rp.RRF_K, rp.CANDIDATE_POOL = rrf_k, pool
                s = summarize(run_mode(questions, index, chunks, "hybrid", k))
                rows.append({"rrf_k": rrf_k, "pool": pool, "k": k, **s})
                print(f"{rrf_k:>6}{pool:>6}{k:>4}{s['hit@1']:>9.3f}{s['hit@3']:>9.3f}"
                      f"{s['hit@10']:>9.3f}{s['recall@10']:>11.3f}{s['mrr']:>8.3f}")
    rp.RRF_K, rp.CANDIDATE_POOL = defaults
    return rows


# ---------------------------
# Main
# ---------------------------
def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--questions", default=str(DATA_DIR / "questions_with_additional.json"))
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--sweep", action="store_true", help="also sweep RRF_K, pool size and k")
    parser.add_argument("--compare-backup", action="store_true", help="also run dense on the old misaligned index")
    args = parser.parse_args()

    rp.logger.setLevel(logging.WARNING)  # don't flood chunk_log.txt with eval queries
    cache_stats = install_embedding_cache()

    index, chunks = rp.load_index()
    if index.ntotal != len(chunks):
        print(f"WARNING: index has {index.ntotal} vectors but chunks.json has {len(chunks)} entries.")
    questions = json.loads(Path(args.questions).read_text(encoding="utf-8"))
    print(f"Loaded {len(questions)} questions, {len(chunks)} chunks. k={args.k}, "
          f"RRF_K={rp.RRF_K}, pool={rp.CANDIDATE_POOL}")

    results = {}
    for mode in MODES:
        records = run_mode(questions, index, chunks, mode, args.k)
        results[mode] = {"summary": summarize(records), "records": records}

    print_summary(results)
    print_categories(results)
    print_wins_losses(results)
    print_unlabelled(results)

    output = {
        "settings": {"k": args.k, "rrf_k": rp.RRF_K, "pool": rp.CANDIDATE_POOL,
                     "min_score": rp.MIN_SCORE, "questions": Path(args.questions).name},
        "results": results,
    }

    if args.compare_backup:
        if BACKUP_INDEX_PATH.exists():
            old_index = faiss.read_index(str(BACKUP_INDEX_PATH))
            s = summarize(run_mode(questions, old_index, chunks, "dense", args.k))
            output["dense_on_misaligned_index"] = s
            print(f"\nDense on OLD misaligned index: hit@1={s['hit@1']:.3f} hit@10={s['hit@10']:.3f} "
                  f"recall@10={s['recall@10']:.3f} mrr={s['mrr']:.3f}")
        else:
            print(f"\n--compare-backup: {BACKUP_INDEX_PATH.name} not found, skipped.")

    if args.sweep:
        output["sweep"] = run_sweep(questions, index, chunks)

    RESULTS_PATH.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nEmbeddings: {cache_stats['misses']} new, {cache_stats['hits']} from cache.")
    print(f"Full results saved to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
