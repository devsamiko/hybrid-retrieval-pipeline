#!/usr/bin/env python3
"""
train_reranker.py — Fine-tune a cross-encoder reranker on domain query/document pairs
=========================================================================================
Adapted from a production fine-tuning pipeline. The algorithm is unchanged:

    1. Mine training pairs from golden queries, using the existing hybrid
       search as the source of hard negatives (candidates that rank highly
       but aren't the correct document — a much more useful negative than a
       random unrelated document).
    2. Fine-tune a small cross-encoder (ms-marco-MiniLM-L-6-v2, 22M params)
       with binary cross-entropy loss.
    3. Evaluate before/after on a held-out split of the same golden queries.

NOTE on the example dataset: this repo's data/golden_queries.json has 16
queries over 8 documents — enough to demonstrate the pipeline runs
end-to-end, but far too small to produce a meaningful fine-tuning result
(a rough rule of thumb in practice is a minimum of ~100 golden examples to
detect a 10% difference reliably — this is general evaluation-sizing
guidance, not a specific cited source). Running this on
the bundled example data will complete without error but the resulting
delta is not a claim worth reporting. The real measured result this
technique produced (+9.3 percentage points Recall@1, base 58.5% -> tuned
67.8%, on 118 golden queries) is documented in evaluation/results.md with
its original source data, not re-claimed here from a toy run.

Usage:
    python train_reranker.py --generate      # build training pairs
    python train_reranker.py --train         # fine-tune (also runs before/after eval)
    python train_reranker.py --all           # both steps in sequence
"""

import argparse
import json
import os
import random
import time
from pathlib import Path

ROOT = Path(__file__).parent.parent

GOLDEN_FILE = ROOT / "data" / "golden_queries.json"
DOCS_DIR = ROOT / "data" / "example_documents"
TRAINING_DATA_FILE = Path(__file__).parent / "training_data.json"
MODEL_OUTPUT_DIR = Path(__file__).parent / "reranker-finetuned"
EVAL_RESULTS_FILE = Path(__file__).parent / "training_eval.json"
BASE_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

EPOCHS = 3
BATCH_SIZE = 8
LEARNING_RATE = 2e-5
WARMUP_RATIO = 0.1
EVAL_SPLIT = 0.2


def _load_golden():
    with open(GOLDEN_FILE) as f:
        return json.load(f)


def _build_search_index():
    """Build an in-memory hybrid-search index over the example documents."""
    import faiss  # noqa: F401 — import before torch, see src/retrieval.py
    from src.chunking import semantic_chunk
    from src.embeddings import embed_texts
    from src.retrieval import Index

    index = Index(":memory:")
    docs = {f.name: f.read_text(encoding="utf-8") for f in sorted(DOCS_DIR.glob("*.md"))}
    all_texts, all_doc_ids = [], []
    for doc_id, text in docs.items():
        for chunk in semantic_chunk(text, ".md"):
            all_texts.append(chunk)
            all_doc_ids.append(doc_id)
    embeddings = embed_texts(all_texts)
    for doc_id, text, emb in zip(all_doc_ids, all_texts, embeddings):
        index.add(doc_id, text, emb)
    return index


def generate_training_data():
    """Mine positive/hard-negative pairs from golden queries via hybrid search."""
    from src.retrieval import hybrid_search

    index = _build_search_index()
    golden = _load_golden()
    random.shuffle(golden)

    training_pairs = []
    stats = {"positives": 0, "hard_negatives": 0, "queries": 0}

    for gq in golden:
        query, expected = gq["query"], gq["expected_doc"]
        results = hybrid_search(index, query, top_k=10, pool=20)
        for r in results:
            is_match = r["doc_id"] == expected
            label = 1.0 if is_match else 0.0
            if is_match:
                stats["positives"] += 1
            else:
                stats["hard_negatives"] += 1
            training_pairs.append({"query": query, "text": r["text"], "label": label})
        stats["queries"] += 1

    with open(TRAINING_DATA_FILE, "w") as f:
        json.dump({"pairs": training_pairs, "stats": stats}, f, indent=2)

    print(f"Training data saved: {TRAINING_DATA_FILE}")
    print(f"  Queries: {stats['queries']}  Positives: {stats['positives']}  "
          f"Hard negatives: {stats['hard_negatives']}  Total pairs: {len(training_pairs)}")
    return training_pairs


def _score(result):
    return result if isinstance(result, (int, float)) else max(result.values())


def _prepare_train_eval_split():
    """Load saved pairs, shuffle, and split into train/eval sets."""
    from sentence_transformers.readers import InputExample
    from sentence_transformers.cross_encoder.evaluation import CEBinaryClassificationEvaluator

    with open(TRAINING_DATA_FILE) as f:
        pairs = json.load(f)["pairs"]
    random.shuffle(pairs)

    split_idx = max(1, int(len(pairs) * (1 - EVAL_SPLIT)))
    train_pairs, eval_pairs = pairs[:split_idx], pairs[split_idx:]
    if not eval_pairs:
        eval_pairs = train_pairs[-1:]

    print(f"Training: {len(train_pairs)} pairs, Eval: {len(eval_pairs)} pairs "
          f"(NOTE: too small for a meaningful result, see module docstring)")

    train_samples = [InputExample(texts=[p["query"], p["text"]], label=float(p["label"]))
                      for p in train_pairs]
    evaluator = CEBinaryClassificationEvaluator(
        sentence_pairs=[(p["query"], p["text"]) for p in eval_pairs],
        labels=[int(p["label"]) for p in eval_pairs],
        name="retrieval-pipeline-demo-eval",
    )
    return train_samples, eval_pairs, evaluator


def train():
    """Fine-tune the cross-encoder. See module docstring re: example-data scale."""
    from sentence_transformers import CrossEncoder
    from torch.utils.data import DataLoader

    train_samples, eval_pairs, evaluator = _prepare_train_eval_split()
    model = CrossEncoder(BASE_MODEL, num_labels=1, max_length=512)

    pre_score = _score(evaluator(model))
    print(f"Pre-training eval score: {pre_score:.4f}")

    train_dataloader = DataLoader(train_samples, batch_size=min(BATCH_SIZE, len(train_samples)),
                                   shuffle=True)
    warmup_steps = max(1, int(len(train_samples) / BATCH_SIZE * EPOCHS * WARMUP_RATIO))
    t0 = time.time()
    model.fit(
        train_dataloader=train_dataloader,
        evaluator=evaluator,
        epochs=EPOCHS,
        warmup_steps=warmup_steps,
        optimizer_params={"lr": LEARNING_RATE},
        output_path=str(MODEL_OUTPUT_DIR),
        show_progress_bar=True,
    )
    elapsed = time.time() - t0

    os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)
    model.save(str(MODEL_OUTPUT_DIR))

    post_score = _score(evaluator(model))
    print(f"Post-training eval score: {post_score:.4f}  (delta {post_score - pre_score:+.4f})")

    results = {
        "base_model": BASE_MODEL, "training_pairs": len(train_samples),
        "eval_pairs": len(eval_pairs), "pre_training_score": pre_score,
        "post_training_score": post_score, "delta": post_score - pre_score,
        "training_time_seconds": elapsed,
        "note": "Demo-scale run on 8 documents / 16 golden queries — not a claimed result. "
                "See evaluation/results.md for the real, source-cited fine-tuning result "
                "measured on the original 118-query production evaluation set.",
    }
    with open(EVAL_RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved: {EVAL_RESULTS_FILE}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune a retrieval reranker (demo pipeline)")
    parser.add_argument("--generate", action="store_true")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()

    if args.all:
        generate_training_data()
        train()
    elif args.generate:
        generate_training_data()
    elif args.train:
        train()
    else:
        parser.print_help()
