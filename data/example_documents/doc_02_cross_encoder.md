# Cross-Encoder Reranking

A bi-encoder embeds a query and a document independently into vectors and
compares them with cosine similarity or dot product. This is fast — the
document embeddings can be precomputed and indexed — but it never lets the
query and document "see" each other during encoding, which limits how
precisely it can judge relevance.

A cross-encoder instead takes the query and document together as a single
input and passes them jointly through a transformer, producing a single
relevance score. This lets the model perform full cross-attention between
query tokens and document tokens, which is much more accurate but also much
slower, since it must be run once per candidate document rather than being
precomputed.

The standard pattern is to use a fast bi-encoder (or a hybrid of vector and
keyword search) to retrieve a broad set of candidates, then apply a
cross-encoder only to the top few dozen candidates to rerank them precisely.
This two-stage design gets most of the speed of a bi-encoder with most of
the accuracy of a cross-encoder.
