# Evaluating Retrieval Quality

Recall@k measures whether the correct document appears anywhere in the top
k results — Recall@1 specifically asks whether it was ranked first. This is
a strict, binary-per-query metric: partial credit isn't given for ranking
the correct document second.

Mean Reciprocal Rank (MRR) is a softer alternative: for each query, take
1 / rank of the first correct result (1.0 if it's ranked first, 0.5 if
second, and so on), then average across all queries. This rewards getting
the correct document close to the top even when it isn't exactly first.

Both metrics require a "golden" evaluation set: a list of realistic queries
paired with the document(s) known to be the correct answer. Since the
sample size of golden queries is usually small, reporting a confidence
interval (for example via bootstrap resampling) alongside the point
estimate is important — a difference of a few percentage points on 20
queries may not be statistically distinguishable from noise, while the
same difference on 200 queries usually is.
