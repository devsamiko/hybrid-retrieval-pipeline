# Reciprocal Rank Fusion

Reciprocal Rank Fusion (RRF) is a method for combining multiple ranked
result lists into a single ranking, without needing the underlying scores
to be on comparable scales. For each document, RRF sums 1 / (k + rank)
across every list it appears in, where rank is its zero-based position in
that list and k is a small constant (commonly 60) that dampens the impact
of very top-ranked items so the fusion isn't dominated by a single list's
outlier.

The technique was introduced for combining multiple retrieval systems in
information retrieval competitions, where different systems (BM25, language
models, learned rankers) produce scores that cannot be directly averaged. RRF
sidesteps this by working purely on rank position, which is always
comparable regardless of the underlying scoring function.

In hybrid search systems, RRF is commonly used to combine a vector
similarity search with a keyword search such as BM25 or SQLite FTS5. A
document that ranks highly in both lists receives a much higher fused score
than a document that only appears in one, which is exactly the behaviour
you want: strong agreement between two different retrieval signals is
strong evidence of relevance.
