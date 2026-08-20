# Evaluation Results

Two separate results, from two different evaluation runs — kept clearly
separate because they were measured under different conditions and neither
should be read as validating the other.

## 1. Cross-encoder reranker fine-tuning (verified, from the source system)

Measured on the original production system this pipeline was extracted
from, on a 118-query golden evaluation set (not the 16-query demo set
bundled with this repo):

| Metric | Base model | Fine-tuned | Delta |
|---|---|---|---|
| Recall@1 | 58.5% | 67.8% | **+9.3 points** |
| Recall@5 | 78.8% | 80.5% | +1.7 points |

Source: `reranker_eval.json` from the original system's evaluation run
(2026-05-17), base model `cross-encoder/ms-marco-MiniLM-L-6-v2` fine-tuned
on domain query/document pairs mined via hard-negative sampling (see
`training/train_reranker.py` for the unmodified algorithm). Recall@1 is
the metric that moved; Recall@5 barely changed, meaning fine-tuning mostly
helped pull the correct document from a lower rank up to #1, not recover
documents the base model missed entirely.

Re-running the training pipeline on this repo's 16-query demo set (as a
pipeline smoke-test, not a claim) completes without error but produces no
meaningful signal — 16 queries is far below the ~100-query minimum
generally needed to detect a 10% difference reliably (Huyen, *AI
Engineering*, Table 4-7). Don't read anything into a demo-scale run's
numbers.

## 2. LLM compression ablation (measured fresh, on this repo's example data)

The source system's architecture doc reports an AKA-compression result of
"+13% Recall@1, +11% MRR" from a 2026-05-03 A/B test. That claim has **no
underlying raw evaluation artifact** in the source system — no saved query
set, no per-query results, nothing to independently verify against, unlike
the reranker result above which does have one. It's presented here as a
documented-but-unverified historical claim, not fact.

To get an honest, reproducible number, this repo ran its own 3-way ablation
on the bundled 8 documents / 16 golden queries, isolating whether any
retrieval improvement comes from (a) the text being LLM-compressed at all,
or (b) the specific retrieval-oriented compression prompt vs. a generic
summary:

| Variant | Recall@1 | vs. raw baseline |
|---|---|---|
| Raw chunk embedding (baseline) | 87.5% (14/16) | — |
| Generic LLM summary embedding | 81.2% (13/16) | −6.2 points |
| Retrieval-oriented compression ("AKA") embedding | 75.0% (12/16) | −12.5 points |

**On this dataset, compression made retrieval worse, and the more
aggressive retrieval-oriented compression made it worse than a plain
summary did.** This is the opposite direction from the source system's
historical claim, and it's a real, if small-sample, result — not noise
dismissed away.

### Why the result likely doesn't generalize from this dataset

The 8 example documents were written to each be short (150-250 words),
single-topic, and already low-redundancy — there's very little noise for
compression to strip out. The source system's claimed improvement was
measured over ~98,000 book sections, a corpus with far more redundant,
verbose, and noisy text per section, where compression has actual signal
to extract. Compression has a cost (it can drop specific terminology a
query might match on) and a benefit (it strips noise); which one dominates
plausibly depends on how noisy the source corpus already is. A clean,
short, single-topic document has little noise to remove and every word
already carries a query-matchable term, so compression mostly just removes
information.

This is a testable hypothesis, not a settled conclusion — the honest
takeaway is that **the compression technique's benefit is corpus-dependent,
not universal**, and claiming a fixed percentage improvement without
stating the corpus it was measured on (as the source system's original doc
did) overstates what was actually shown.
