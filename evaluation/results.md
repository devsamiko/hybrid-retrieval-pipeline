# Evaluation Results

Two separate results, from two different evaluation runs — kept clearly
separate because they were measured under different conditions and neither
should be read as validating the other.

> **Key finding:** LLM-based retrieval compression appears to be
> corpus-dependent — it can hurt retrieval on short, clean,
> information-dense documents, while substantially improving retrieval on
> a large, noisy, redundant knowledge corpus. The interesting result here
> is not simply that compression improved Recall@1. It's that the *same*
> technique produced a negative result on clean data and a large positive
> result on a realistic corpus, which is what motivated the follow-up
> investigation in Section 3 rather than stopping at either number alone.

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

## 3. Follow-up: testing the hypothesis on the real production corpus

The hypothesis above is testable, so it was tested — against the source
system's actual 402K-vector production index, not a reconstruction of it.
47 held-out queries (book/reference-domain only, no personal content) were
run through the system's real hybrid retrieval twice: once with
AKA-compressed vectors participating in the index as they normally do, and
once with AKA-derived results filtered out of the same retrieval, so both
runs hit the identical live pipeline and differ only in whether compressed
representations are allowed to contribute.

| Metric | With AKA | Without AKA | Delta |
|---|---|---|---|
| Recall@1 | 76.6% (36/47) | 61.7% (29/47) | **+14.9 points (+24% relative)** |
| Recall@5 | 93.6% (44/47) | 80.9% (38/47) | +12.7 points |
| Recall@10 | 95.7% | 85.1% | +10.6 points |
| MRR | 0.840 | 0.702 | +0.138 |

**This confirms, and slightly exceeds, the source system's original
"+13% Recall@1" claim** — this time with a real, inspectable query set and
methodology behind the number, rather than an undocumented prose line.

This doesn't overturn Section 2's small-demo result — it completes it.
Read together, the two runs are consistent with a corpus-dependence
hypothesis: compression has a cost (it can drop specific terminology a
query might match on) and a potential benefit (it can strip redundant or
noisy text and produce a representation better suited to semantic
retrieval). On short, clean documents there's little redundant or noisy
information to remove, so the cost dominates and compression mostly just
deletes useful detail. On a large book/knowledge corpus with verbose,
redundant, noisy text, the benefit can dominate instead.

**This explanation is a hypothesis, not a proven causal mechanism.** Two
data points (one small clean corpus, one large real corpus) are consistent
with it, but they don't establish it — a controlled sweep across corpus
sizes/redundancy levels, which this repo does not attempt, would be needed
to actually test it rather than just support it directionally.

**Methodology note:** this production run measures "compressed vectors
included in vs. excluded from the same hybrid index," which is not
identical to Section 2's strictly separate raw-only-index vs.
compressed-only-index comparison — it's the more production-relevant
question, but the two numbers aren't directly comparable apples-to-apples.
Sample size (47 queries, single run, no bootstrap confidence interval) is
also modest; treat the direction and rough magnitude as reliable, not the
exact percentage.

**Reproducing this without the private corpus:** the production run can't
be reproduced as-is outside the source system, since it depends on a
402K-vector personal/book index that isn't published (copyrighted books +
private content). What *is* reproducible: the method itself — build any
sizeable, redundant document corpus (e.g. a public book/paper collection
you have rights to index), generate AKA-style compressed representations
per chunk with `src/llm_compression.py`'s prompt, index both raw and
compressed vectors, and run `evaluation/evaluate_recall.py` with and
without the compressed vectors included. The demo's 8-document run is that
exact procedure at small scale; scaling it up should reproduce the
direction of the production finding if the corpus-dependence hypothesis
holds.

**Bottom line:** don't use a small, clean, low-redundancy corpus to decide
whether LLM-compression-before-embedding is worth doing — on this system,
the technique's benefit only showed up once the corpus had real redundancy
and noise for compression to remove, and the small-corpus test alone would
have led to the wrong conclusion.
