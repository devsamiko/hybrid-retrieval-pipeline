# Retrieval Pipeline

A minimal hybrid semantic search pipeline: document chunking, embeddings,
optional LLM-based compression, vector + keyword search fused with
Reciprocal Rank Fusion, and cross-encoder reranking with fine-tuning.

This is extracted and simplified from a larger personal RAG system (a
production knowledge base indexing hundreds of documents, running 24/7).
The extraction kept the parts that are genuinely general-purpose retrieval
technique — chunking, hybrid search, reranking, fine-tuning — and left
behind the parts that were specific to that system's personal architecture
(a multi-project knowledge graph, usage-based ranking, telemetry). What's
here is meant to be readable end-to-end and runnable on your own documents,
not a slice of a bigger system you'd need the rest of to understand.

## Architecture

```
Documents
    |
Structure-aware chunking          (src/chunking.py)
    |
    +---------------------------------+
    |                                 |
Raw chunk                    LLM compression       (src/llm_compression.py, optional)
    |                                 |
    +----------------+----------------+
                      |
                 Embeddings                          (src/embeddings.py)
                      |
        +-------------+-------------+
        |                           |
   Vector index                Keyword index         (src/retrieval.py)
   (FAISS)                     (SQLite FTS5)
        |                           |
        +-------------+-------------+
                      |
              Reciprocal Rank Fusion                  (src/retrieval.py)
                      |
              Cross-encoder reranker                  (src/reranking.py)
                      |
                Top-k results
                      |
                  Evaluation                          (evaluation/)
```

**Why each stage exists:**

- **Chunking** is structure-aware (splits on headings/pages/paragraphs, not
  a fixed character count) because retrieval precision depends on each
  indexed unit covering one coherent idea, not an arbitrary slice of text.
- **Embeddings** turn text into vectors for semantic similarity search —
  the core enabler of "find conceptually related text," not just keyword
  matches.
- **LLM compression** (optional) replaces the raw chunk's embedding target
  with an LLM-generated compressed representation, on the hypothesis that
  raw text often carries noise that dilutes the embedding's signal. See
  `evaluation/results.md` — this repo measured this hypothesis honestly and
  it did **not** hold on the small example corpus here, which is itself
  the interesting finding (see below).
- **Vector + keyword hybrid search** exists because the two signals fail
  differently: vector search misses exact terminology matches an embedding
  model doesn't represent distinctly; keyword search misses conceptual
  matches that use different words for the same idea. Combining both
  outperforms either alone.
- **Reciprocal Rank Fusion** combines the two ranked lists without needing
  their scores to be on a comparable scale — it works purely on rank
  position, which is always comparable.
- **Cross-encoder reranking** scores query and document jointly (full
  cross-attention) rather than comparing precomputed independent vectors,
  which is far more precise but too slow to run on every candidate — so
  it's applied only to the RRF-fused shortlist.
- **Fine-tuning the reranker** on domain-specific query/document pairs
  closes the gap between a generic public-benchmark model and your actual
  corpus's vocabulary and notion of relevance.

## Results

Two separate, clearly-labeled results — full detail and honest caveats in
[`evaluation/results.md`](evaluation/results.md).

**Key finding:** LLM-based retrieval compression appears to be
corpus-dependent — it can hurt retrieval on short, clean, information-dense
documents, while substantially improving retrieval on a large, noisy,
redundant knowledge corpus. The interesting result isn't just that
compression improved Recall@1 in production — it's that the same technique
produced a negative result on clean data and a large positive result on a
realistic corpus, which is what's actually worth investigating.

1. **Reranker fine-tuning**: +9.3 percentage points Recall@1 (58.5% →
   67.8%), measured on the original 118-query production evaluation set.
   Verified against the original system's saved eval artifact.

2. **LLM compression ablation, two scales:**
   - On the bundled **8-document demo set** (run fresh, in this repo):
     compression **did not help** — both a generic LLM summary and a
     retrieval-oriented compressed representation scored *below* the
     raw-chunk baseline (−6.2 and −12.5 percentage points Recall@1).
   - On the source system's **real 402K-vector production corpus** (47
     held-out book/reference queries, live hybrid retrieval, read-only):
     compression **helped substantially** — +14.9 percentage points
     Recall@1 (61.7% → 76.6%), which confirms and slightly exceeds a
     historical, previously-unverified "+13%" claim from the source
     system's architecture doc.

   Read together, these two runs support a corpus-dependent hypothesis:
   compression only helps once the source corpus has enough redundancy and
   noise for compression to actually strip out. A small, clean,
   low-redundancy demo corpus doesn't have that noise, so compression just
   discards useful signal there; a large, heterogeneous real corpus does,
   so compression helps. Full numbers, methodology, and caveats for both
   runs are in `evaluation/results.md`.

Both results are presented deliberately, not just the flattering one — an
ablation that shows a plausible technique failing at small scale, and then
confirms it at the scale it was actually designed for, with a stated and
tested hypothesis explaining the difference, is a more useful engineering
artifact than a single success number with no caveats.

## Running it

```bash
pip install -r requirements.txt

# Full pipeline demo on the bundled example documents
python examples/basic_pipeline.py "how does reciprocal rank fusion work"

# Retrieval evaluation (raw-embedding baseline only, no LLM needed)
python evaluation/evaluate_recall.py --skip-llm

# Full 3-way ablation (raw / summary / AKA-compressed) — needs a local LLM
# wired up in evaluation/evaluate_recall.py's generate_fn
python evaluation/evaluate_recall.py

# Reranker fine-tuning pipeline (demo-scale; see training/train_reranker.py
# docstring for why the resulting number isn't a claim)
python training/train_reranker.py --all
```

**Note (macOS):** if you hit a segfault on import, it's a known `faiss` +
`torch` OpenMP conflict — `faiss` must be imported before `torch` (see the
import order in `src/retrieval.py`), and you may also need
`KMP_DUPLICATE_LIB_OK=TRUE` set in your environment.

## Structure

```
retrieval-pipeline/
├── src/
│   ├── chunking.py         # structure-aware document splitting
│   ├── embeddings.py       # sentence-transformers wrapper
│   ├── llm_compression.py  # optional LLM-based compressed representations
│   ├── retrieval.py        # vector + keyword search, RRF fusion
│   └── reranking.py        # cross-encoder reranking
├── training/
│   └── train_reranker.py   # fine-tuning pipeline
├── data/
│   ├── example_documents/  # 8 short original documents (not copyrighted material)
│   └── golden_queries.json # 16 query/expected-document pairs
├── evaluation/
│   ├── evaluate_recall.py  # Recall@1 + 3-way compression ablation
│   └── results.md          # full write-up, both results, honest caveats
└── examples/
    └── basic_pipeline.py   # end-to-end demo
```

## What's deliberately not here

Graph-based document expansion, project/collection routing, a learned
usage-based reranker, and telemetry logging were part of the source
system but are specific to *that* system's personal multi-collection
architecture, not general retrieval technique — they were left out rather
than ported over as-is.
