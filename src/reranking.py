"""
reranking.py — Cross-encoder reranking of hybrid search candidates
=====================================================================
RRF fusion is a rank-based heuristic — fast, but it doesn't actually read
query and document together. A cross-encoder scores each (query, document)
pair jointly, which is slower (must run once per candidate) but much more
precise, so it's applied only to the top RRF candidates, not the whole pool.

Final score blends the cross-encoder's judgment with the RRF signal
(0.7 / 0.3 split) rather than trusting the reranker alone, since RRF still
carries useful information the cross-encoder's narrower context window
sometimes misses.
"""

from functools import lru_cache

DEFAULT_RERANKER = "cross-encoder/ms-marco-MiniLM-L-6-v2"
RERANK_DEPTH = 30


@lru_cache(maxsize=1)
def _get_reranker(model_name: str = DEFAULT_RERANKER):
    from sentence_transformers import CrossEncoder
    return CrossEncoder(model_name)


def rerank(query: str, candidates: list[dict], model_name: str = DEFAULT_RERANKER,
           top_k: int = 10) -> list[dict]:
    """Rerank RRF-fused candidates with a cross-encoder.

    Expects each candidate dict to have "text" and "rrf_score" (from
    retrieval.reciprocal_rank_fusion). Returns candidates sorted by blended
    score, truncated to top_k.
    """
    if not candidates:
        return []

    pool = candidates[:RERANK_DEPTH]
    reranker = _get_reranker(model_name)
    pairs = [(query, c["text"]) for c in pool]
    raw_scores = reranker.predict(pairs)

    # Min-max normalization here is relative to THIS query's candidate pool,
    # not a fixed scale — the worst candidate in any pool always maps to 0
    # and the best to 1, regardless of whether any candidate is actually a
    # good match. That's a real limitation: it discards the reranker's
    # absolute-confidence signal and keeps only relative order within an
    # already-filtered pool. Acceptable for reranking a pre-filtered top-N
    # (the ordering is what matters here), but don't reuse this pattern
    # anywhere an absolute confidence threshold is needed.
    s_min, s_max = min(raw_scores), max(raw_scores)
    s_range = (s_max - s_min) or 1.0
    rrf_max = max(c.get("rrf_score", 0.0) for c in pool) or 1.0

    for cand, score in zip(pool, raw_scores):
        rerank_norm = (float(score) - float(s_min)) / float(s_range)
        rrf_norm = float(cand.get("rrf_score", 0.0)) / float(rrf_max)
        cand["rerank_score"] = 0.7 * rerank_norm + 0.3 * rrf_norm

    pool.sort(key=lambda x: x["rerank_score"], reverse=True)
    return pool[:top_k]
