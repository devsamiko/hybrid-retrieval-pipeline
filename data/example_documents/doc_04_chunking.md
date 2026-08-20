# Document Chunking for Retrieval

Splitting a long document into smaller chunks before embedding is
necessary because embedding models have a limited context window, and
because retrieval precision improves when each indexed unit covers one
coherent topic rather than an entire document's worth of mixed content.

Naive fixed-size chunking (e.g. every 500 characters) is simple but breaks
sentences and sections arbitrarily, sometimes splitting a definition from
its explanation. Structure-aware chunking instead respects document
boundaries — splitting on headings for structured text, on page breaks for
PDFs, and on paragraph boundaries as a fallback — so that each chunk stays
topically coherent.

Adding a small overlap (carrying the tail of one chunk into the start of
the next) helps recover context that would otherwise be lost exactly at a
chunk boundary, at the cost of some duplicated content across the index.
