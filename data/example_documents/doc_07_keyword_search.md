# Keyword Search and BM25

Keyword search matches queries against documents based on exact or
near-exact term overlap, ranked by a scoring function such as BM25, which
weighs term frequency in the document against how rare that term is across
the whole collection (inverse document frequency), with diminishing
returns for repeated occurrences of the same term.

Keyword search excels at exact matches — an error code, a proper noun, a
specific technical term — that a semantic vector search can sometimes miss
if the embedding model doesn't represent that specific token distinctly.
Conversely, vector search excels at conceptual matches where the query and
the relevant document use different words for the same idea.

This complementary weakness is exactly why hybrid search — combining both
signals rather than picking one — tends to outperform either alone,
particularly on domain-specific corpora where exact terminology matters
alongside conceptual meaning.
