# Compressed Representations for Retrieval

Instead of embedding a document chunk's raw text directly, some retrieval
systems first pass the chunk through an LLM to produce a compressed
representation, and embed that instead. The idea is that a raw chunk often
contains filler, boilerplate, or verbose phrasing that dilutes the signal
an embedding model needs to represent the chunk's actual meaning.

The prompt used for this compression matters. A generic summarization
prompt asks for a natural-language summary optimized for human readability.
A retrieval-oriented compression prompt instead asks for a dense
representation explicitly optimized to preserve entities, relationships,
and actions — the things a search query is likely to match against — even
at the cost of natural readability, since the compressed text is only ever
seen by the embedding model, not a human reader.

Measuring whether this actually helps requires an ablation: compare
retrieval performance using raw-chunk embeddings, generic-summary
embeddings, and retrieval-oriented compressed embeddings on the same
evaluation set. If the retrieval-oriented variant outperforms the
generic-summary variant despite both being similarly short, that isolates
the improvement to the compression strategy itself, not merely to the text
being shorter.
