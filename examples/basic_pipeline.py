#!/usr/bin/env python3
"""
basic_pipeline.py — End-to-end demo: chunk -> embed -> hybrid search -> rerank
=================================================================================
Runs the full pipeline (minus the LLM compression step, which needs a local
model configured — see evaluation/evaluate_recall.py for that) over the
bundled example documents, for one example query.

Usage:
    python examples/basic_pipeline.py "how does reciprocal rank fusion work"
"""

import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

import faiss  # noqa: E402,F401 — import before torch, see src/retrieval.py
from chunking import semantic_chunk       # noqa: E402
from embeddings import embed_texts        # noqa: E402
from retrieval import Index, hybrid_search  # noqa: E402
from reranking import rerank              # noqa: E402

DOCS_DIR = ROOT / "data" / "example_documents"


def build_index() -> Index:
    index = Index(":memory:")
    all_texts, all_doc_ids = [], []
    for f in sorted(DOCS_DIR.glob("*.md")):
        text = f.read_text(encoding="utf-8")
        for chunk in semantic_chunk(text, ".md"):
            all_texts.append(chunk)
            all_doc_ids.append(f.name)
    embeddings = embed_texts(all_texts)
    for doc_id, text, emb in zip(all_doc_ids, all_texts, embeddings):
        index.add(doc_id, text, emb)
    return index


def main():
    query = sys.argv[1] if len(sys.argv) > 1 else "how does reciprocal rank fusion work"
    print(f"Query: {query}\n")

    print("Indexing example documents...")
    index = build_index()

    print("Stage 1: hybrid search (vector + keyword, RRF fusion)")
    candidates = hybrid_search(index, query, top_k=10, pool=20)
    for i, c in enumerate(candidates[:5]):
        print(f"  [{i}] {c['doc_id']}  rrf={c['rrf_score']:.4f}")

    print("\nStage 2: cross-encoder reranking")
    reranked = rerank(query, candidates, top_k=5)
    for i, c in enumerate(reranked):
        print(f"  [{i}] {c['doc_id']}  rerank_score={c['rerank_score']:.4f}")

    print(f"\nTop result: {reranked[0]['doc_id']}")
    print(f"  {reranked[0]['text'][:200]}...")


if __name__ == "__main__":
    main()
