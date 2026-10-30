# The Emergency Responder Assistant Project
## Team Name
Subitis

## Team Members
1. Tran Duy Thai
2. Bernard junior Seka
3. Ram Navlani
4. Thin Sapal Yu

## Evaluation

### Retrieval: dense vs BM25 vs hybrid (`scripts/eval_retriever.py`)

Compares three retrieval modes, without any LLM calls:

- **dense**: vector search with OpenAI `text-embedding-3-small` embeddings in a FAISS index (cosine similarity).
- **bm25**: keyword search with BM25 (`rank-bm25`).
- **hybrid**: dense and BM25 results merged with Reciprocal Rank Fusion (RRF, `rrf_k=60`, 50 candidates from each retriever).

For each question that has a labelled correct page, it checks where that page appears in the top 10 results:

- **Hit@k**: is a correct page in the top k?
- **Recall@10**: what fraction of the correct pages is in the top 10?
- **MRR**: 1 / rank of the first correct page.

Results on 32 labelled questions (`data/questions_with_additional.json`):

| Mode | Hit@1 | Hit@3 | Hit@5 | Hit@10 | Recall@10 | MRR |
|---|---|---|---|---|---|---|
| dense | 0.688 | 0.906 | 0.938 | **1.000** | **1.000** | 0.798 |
| bm25 | 0.594 | 0.781 | 0.875 | 0.938 | 0.906 | 0.709 |
| hybrid | **0.719** | 0.906 | 0.938 | 0.969 | 0.961 | **0.822** |

- Hybrid ranks the correct page first most often (best Hit@1 and MRR).
- Dense has the best coverage: it finds a correct page in the top 10 for every question. Hybrid misses one procedural question that dense ranks 6th.
- BM25 alone is the weakest mode, but it helps hybrid on cross-reference and multi-hop questions.
- With 32 questions, one question is about 0.03, so the gaps between dense and hybrid are small.

```bash
python scripts/eval_retriever.py
```

Query embeddings are cached in `data/query_embedding_cache.json`, so reruns are free. Full results are saved to `data/retrieval_eval_results.json`.

### Answer quality: LLM-as-a-judge (`scripts/eval_runner.py`)

Runs the full RAG pipeline (retrieve top 10, then generate an answer with `gpt-4o-mini`) on `data/questions.json`. A `gpt-4o-mini` judge sorts each answer into one of four verdicts:

- **faithful** (pass): the answer is supported by the retrieved context.
- **correct_refusal** (pass): the context has no relevant information and the answer declines.
- **missed** (fail): the context had the answer, but the system said it didn't know.
- **hallucinated** (fail): the answer makes claims that aren't in the context.

If the answer is a refusal and the top retrieval score is below 0.35, it counts as `correct_refusal` without calling the judge.

Results on 20 questions: **19/20 passed (95%)**, 0 hallucinated.

| Verdict | Count |
|---|---|
| faithful | 16 |
| correct_refusal | 3 |
| missed | 1 |
| hallucinated | 0 |

| Category | Passed |
|---|---|
| factual | 11/12 |
| cross-reference | 3/3 |
| ambiguous | 2/2 |
| out-of-scope | 3/3 |

The one failure was "What are the four modes of disease transmission?". The retrieved context mentions that there are four modes but does not list them, so the answer was incomplete.

```bash
python scripts/eval_runner.py
```

This calls the OpenAI API: one answer and up to one judge call per question. Results are saved to `data/eval_results.json`.
