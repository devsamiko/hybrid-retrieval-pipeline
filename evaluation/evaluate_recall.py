#!/usr/bin/env python3
"""
evaluate_recall.py — Retrieval evaluation, including a 3-way compression ablation
====================================================================================
Measures Recall@1 (was the correct document ranked first?) on the golden
query set in data/golden_queries.json, across three embedding strategies,
through the SAME full pipeline (hybrid vector+keyword+RRF search, then
cross-encoder reranking — src/retrieval.py + src/reranking.py) for all three:

    1. raw       — embed the chunk's raw text directly (baseline)
    2. summary   — embed a generic LLM-generated summary of the chunk
    3. compressed — embed a retrieval-oriented compressed representation
                    of the chunk (the "AKA" technique, src/compression.py)

Variant 2 exists purely as an ablation control: it isolates whether any
improvement from variant 3 comes from the text being shorter (which a plain
summary also achieves) or from the retrieval-oriented compression prompt
specifically (arrows/pipes/entity-preserving shorthand, not natural
readability).

Usage:
    python evaluate_recall.py                  # all 3 variants
    python evaluate_recall.py --skip-llm        # raw only (no LLM required)
"""

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent.parent

import faiss                                 # noqa: E402,F401 — see retrieval.py: must
                                              # import before torch to avoid a macOS segfault
from src.chunking import semantic_chunk      # noqa: E402
from src.embeddings import embed_texts       # noqa: E402
from src.retrieval import Index, hybrid_search  # noqa: E402
from src.reranking import rerank             # noqa: E402

DOCS_DIR = ROOT / "data" / "example_documents"
GOLDEN_FILE = ROOT / "data" / "golden_queries.json"
RESULTS_FILE = Path(__file__).parent / "results.json"
RERANK_POOL = 10  # candidates pulled from hybrid search before reranking


def load_documents() -> dict[str, str]:
    docs = {}
    for f in sorted(DOCS_DIR.glob("*.md")):
        docs[f.name] = f.read_text(encoding="utf-8")
    return docs


def load_golden():
    with open(GOLDEN_FILE) as f:
        return json.load(f)


def build_index(chunks_by_doc: dict[str, list[str]]) -> Index:
    """Build a fresh in-memory index from {doc_id: [chunk_text, ...]}."""
    index = Index(":memory:")
    all_texts = []
    all_doc_ids = []
    for doc_id, chunks in chunks_by_doc.items():
        for chunk in chunks:
            all_texts.append(chunk)
            all_doc_ids.append(doc_id)
    embeddings = embed_texts(all_texts)
    for doc_id, text, emb in zip(all_doc_ids, all_texts, embeddings):
        index.add(doc_id, text, emb)
    return index


def evaluate_variant(index: Index, golden: list[dict]) -> dict:
    """Recall@1 through the FULL pipeline: hybrid (vector+keyword+RRF) search,
    then cross-encoder reranking — the same path examples/basic_pipeline.py
    uses, not vector search alone. A vector-only eval would measure a
    different, easier system than the one this repo actually presents."""
    hits_at_1 = 0
    total = 0
    for gq in golden:
        candidates = hybrid_search(index, gq["query"], top_k=RERANK_POOL)
        results = rerank(gq["query"], candidates, top_k=5)
        total += 1
        if results and results[0]["doc_id"] == gq["expected_doc"]:
            hits_at_1 += 1
    return {"recall_at_1": hits_at_1 / total if total else 0.0, "total": total, "hits": hits_at_1}


def run_raw(docs: dict[str, str], golden: list[dict]) -> dict:
    chunks_by_doc = {doc_id: semantic_chunk(text, ".md") for doc_id, text in docs.items()}
    index = build_index(chunks_by_doc)
    return evaluate_variant(index, golden)


def run_compressed_variant(docs: dict[str, str], golden: list[dict], mode: str) -> dict | None:
    """mode: 'compressed' (AKA-style) or 'summary' (plain). Requires a local LLM."""
    try:
        sys.path.insert(0, str(ROOT.parent.parent / "scripts"))
        from local_llm import generate as llm_generate, is_available
        from gpu_schedule import gpu_session
    except ImportError as e:
        print(f"  [skip] local LLM backend unavailable: {e}")
        return None

    if not is_available():
        print("  [skip] local LLM not available (server not running)")
        return None

    from src.llm_compression import compress_for_retrieval, summarize_plain

    def generate_fn(prompt: str) -> str:
        return llm_generate(prompt, max_tokens=100)

    chunks_by_doc = {}
    with gpu_session("foreground", f"retrieval-pipeline-eval-{mode}") as have_gpu:
        if not have_gpu:
            print("  [skip] GPU busy, could not acquire session")
            return None
        for doc_id, text in docs.items():
            chunks = semantic_chunk(text, ".md")
            transformed = []
            for chunk in chunks:
                if mode == "compressed":
                    rep = compress_for_retrieval(chunk, generate_fn)
                else:
                    rep = summarize_plain(chunk, generate_fn)
                transformed.append(rep if rep else chunk)
            chunks_by_doc[doc_id] = transformed

    index = build_index(chunks_by_doc)
    return evaluate_variant(index, golden)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-llm", action="store_true",
                         help="Only run the raw-embedding baseline (no LLM required)")
    args = parser.parse_args()

    docs = load_documents()
    golden = load_golden()
    print(f"Documents: {len(docs)}  Golden queries: {len(golden)}\n")

    results = {}

    print("[1/3] raw chunk -> embedding (baseline)...")
    t0 = time.time()
    results["raw"] = run_raw(docs, golden)
    results["raw"]["seconds"] = round(time.time() - t0, 1)
    print(f"  Recall@1: {results['raw']['recall_at_1']:.3f} "
          f"({results['raw']['hits']}/{results['raw']['total']})\n")

    if not args.skip_llm:
        print("[2/3] plain LLM summary -> embedding (ablation control)...")
        t0 = time.time()
        r = run_compressed_variant(docs, golden, "summary")
        if r:
            r["seconds"] = round(time.time() - t0, 1)
            results["summary"] = r
            print(f"  Recall@1: {r['recall_at_1']:.3f} ({r['hits']}/{r['total']})\n")

        print("[3/3] retrieval-oriented compression (AKA) -> embedding...")
        t0 = time.time()
        r = run_compressed_variant(docs, golden, "compressed")
        if r:
            r["seconds"] = round(time.time() - t0, 1)
            results["compressed"] = r
            print(f"  Recall@1: {r['recall_at_1']:.3f} ({r['hits']}/{r['total']})\n")

    baseline = results["raw"]["recall_at_1"]
    print("=" * 60)
    print(f"{'Variant':<15} {'Recall@1':>10} {'vs baseline':>14}")
    print("-" * 60)
    for name, r in results.items():
        delta = r["recall_at_1"] - baseline
        print(f"{name:<15} {r['recall_at_1']:>10.3f} {delta:>+13.3f}")
    print("=" * 60)

    with open(RESULTS_FILE, "w") as f:
        json.dump({"results": results, "n_documents": len(docs),
                    "n_golden_queries": len(golden),
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}, f, indent=2)
    print(f"\nSaved: {RESULTS_FILE}")


if __name__ == "__main__":
    main()
