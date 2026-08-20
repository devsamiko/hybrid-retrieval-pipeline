# Evaluation Results

Two separate experiments, run under different conditions. Don't read one as
validating the other.

**Key finding:** LLM-based compression before embedding is corpus-dependent.
It hurt retrieval on a small, clean document set and helped a lot on a
large, noisy one. The interesting part isn't that compression won in
production — it's that the same technique lost on clean data and won big on
realistic data, which is worth digging into rather than just reporting one
number.

## 1. Cross-encoder reranker fine-tuning

Verified result from the original production system, on a 118-query golden
set (not the 16-query demo set bundled here):

| Metric | Base model | Fine-tuned | Delta |
|---|---|---|---|
| Recall@1 | 58.5% | 67.8% | **+9.3 points** |
| Recall@5 | 78.8% | 80.5% | +1.7 points |

Source: `reranker_eval.json` from the original system's 2026-05-17 eval run.
Base model is `cross-encoder/ms-marco-MiniLM-L-6-v2`, fine-tuned on domain
pairs mined via hard-negative sampling (see `training/train_reranker.py`
for the unmodified algorithm). Recall@5 barely moved, so fine-tuning mostly
pulled the right document up to #1 from a lower rank — it didn't recover
documents the base model missed entirely.

I re-ran the training pipeline on this repo's 16-query demo set as a smoke
test. It completes without error but the number means nothing — 16 queries
is well below the ~100 generally needed to detect a 10% difference
reliably (Huyen, *AI Engineering*, Table 4-7).

## 2. LLM compression: small demo corpus

The source system's architecture doc claims "+13% Recall@1, +11% MRR" from
a 2026-05-03 A/B test. There's no saved query set or per-query data behind
that number — just the prose claim. So I ran my own 3-way ablation instead
of repeating it: raw chunk vs. a generic LLM summary vs. a
retrieval-oriented compressed representation ("AKA"), same 8 documents, 16
golden queries, same retrieval code for all three.

| Variant | Recall@1 | vs. raw |
|---|---|---|
| Raw chunk embedding | 87.5% (14/16) | — |
| Generic LLM summary | 81.2% (13/16) | −6.2 points |
| Retrieval-oriented compression ("AKA") | 75.0% (12/16) | −12.5 points |

Compression made things worse here, and the more aggressive
retrieval-oriented version was worse than a plain summary. Opposite
direction from the historical claim.

My read: the 8 example documents are short (150-250 words), single-topic,
and already dense — there's nothing redundant for compression to strip
out, so it just throws away words a query might match on. The original
+13% claim was measured over ~98,000 book sections, a corpus with a lot
more redundant and verbose text per section. Compression trades off a cost
(losing specific terms) against a benefit (removing noise), and which one
wins probably depends on how noisy the source text already is.

## 3. LLM compression: real production corpus

So I tested that. Ran the same comparison against the source system's real
402K-vector index — 47 held-out book/reference queries, no personal
content, read-only. Rather than build a separate raw-only vs.
compressed-only index, I compared the system's live hybrid retrieval with
AKA vectors included vs. the same retrieval with AKA results filtered out,
so both runs hit the identical pipeline.

| Metric | With AKA | Without AKA | Delta |
|---|---|---|---|
| Recall@1 | 76.6% (36/47) | 61.7% (29/47) | **+14.9 points (+24% relative)** |
| Recall@5 | 93.6% (44/47) | 80.9% (38/47) | +12.7 points |
| Recall@10 | 95.7% | 85.1% | +10.6 points |
| MRR | 0.840 | 0.702 | +0.138 |

The result is very close to the old undocumented +13% figure, but this
time there's an actual query set and evaluation behind it.

This doesn't overturn the small-demo result — it completes it. Small,
clean corpus: compression hurts. Large, redundant, real corpus: compression
helps, substantially. Two data points aren't proof of the corpus-noise
explanation, just support for it — a proper test would sweep corpus size
and redundancy directly, which I haven't done.

**Caveat on the method:** this compares AKA-included vs. AKA-excluded
within the same hybrid index, not the same raw-only vs. compressed-only
split used in the small demo, so the two numbers aren't directly
comparable. Sample size is 47 queries, one run, no bootstrap confidence
interval — trust the direction and rough size of the effect, not the exact
decimal.

**Reproducing this without the private corpus:** you can't rerun the
production numbers directly — that index is personal and partly
copyrighted. What you can rerun is the method: take any large, redundant
document collection you have rights to, chunk it, embed both raw and
AKA-compressed versions with `src/llm_compression.py`'s prompt, and run
`evaluation/evaluate_recall.py` with and without the compressed vectors.
The demo run here is that exact procedure at small scale — scaling it up
should show the same direction if the corpus-noise idea holds.

**What I'd test next:** a few corpus sizes in between 8 docs and 400K
vectors, to see where the crossover actually happens, and whether it's a
sharp threshold or a gradual curve.
