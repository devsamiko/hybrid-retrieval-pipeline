"""
retrieval.py — Hybrid retrieval: vector search + keyword search + RRF fusion
==============================================================================
A minimal local index (FAISS + SQLite FTS5) and a hybrid search combining
both signals with Reciprocal Rank Fusion, then an optional cross-encoder
reranking pass (see reranking.py).

This is the retrieval core extracted from a production 6-stage pipeline that
also included graph expansion and project-routing stages — those were
specific to that system's personal multi-project knowledge graph, not
generic retrieval technique, so they're not reproduced here. What's kept
(vector search, keyword search, RRF, reranking) is the actually reusable
retrieval algorithm.
"""

import sqlite3

import faiss  # noqa: F401 — imported early, before torch, to avoid a macOS
              # OpenMP conflict between faiss and torch that segfaults if
              # torch (via sentence-transformers) loads first.
import numpy as np

from .embeddings import embed_text


class Index:
    """A local FAISS + SQLite FTS5 index over a set of (id, text, metadata) chunks."""

    def __init__(self, db_path: str = ":memory:"):
        self.db = sqlite3.connect(db_path)
        self.db.row_factory = sqlite3.Row
        self.db.execute("""
            CREATE TABLE IF NOT EXISTS chunks (
                id INTEGER PRIMARY KEY,
                doc_id TEXT,
                text TEXT
            )
        """)
        self.db.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
                text, content='chunks', content_rowid='id'
            )
        """)
        self.db.commit()
        self._faiss_index: faiss.Index | None = None
        self._dim = None

    def add(self, doc_id: str, text: str, embedding: list[float]) -> int:
        """Add one chunk. Returns its rowid (used as the FAISS vector id)."""
        cur = self.db.execute(
            "INSERT INTO chunks (doc_id, text) VALUES (?, ?)", (doc_id, text)
        )
        rowid = cur.lastrowid
        self.db.execute(
            "INSERT INTO chunks_fts (rowid, text) VALUES (?, ?)", (rowid, text)
        )
        self.db.commit()

        vec = np.array([embedding], dtype=np.float32)
        if self._faiss_index is None:
            self._dim = len(embedding)
            self._faiss_index = faiss.IndexIDMap2(faiss.IndexFlatIP(self._dim))
        self._faiss_index.add_with_ids(vec, np.array([rowid], dtype=np.int64))
        return rowid

    def vector_search(self, query: str, top_k: int = 10) -> list[dict]:
        if self._faiss_index is None or self._faiss_index.ntotal == 0:
            return []
        query_emb = np.array([embed_text(query)], dtype=np.float32)
        search_k = min(top_k, self._faiss_index.ntotal)
        distances, indices = self._faiss_index.search(query_emb, search_k)
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < 0:
                continue
            row = self.db.execute(
                "SELECT id, doc_id, text FROM chunks WHERE id = ?", (int(idx),)
            ).fetchone()
            if not row:
                continue
            results.append({
                "id": row["id"], "doc_id": row["doc_id"], "text": row["text"],
                "score": float(dist), "source": "vector",
            })
        return results

    def keyword_search(self, query: str, top_k: int = 10) -> list[dict]:
        words = [w.strip('"') for w in query.split() if len(w) > 1]
        if not words:
            return []
        fts_query = words[0] if len(words) == 1 else " OR ".join(words)
        try:
            rows = self.db.execute(
                "SELECT c.id, c.doc_id, c.text, chunks_fts.rank "
                "FROM chunks_fts JOIN chunks c ON c.id = chunks_fts.rowid "
                "WHERE chunks_fts MATCH ? ORDER BY chunks_fts.rank LIMIT ?",
                (fts_query, top_k),
            ).fetchall()
        except sqlite3.OperationalError:
            return []
        return [
            {"id": r["id"], "doc_id": r["doc_id"], "text": r["text"],
             "score": -float(r["rank"]), "source": "keyword"}
            for r in rows
        ]


def reciprocal_rank_fusion(
    result_lists: list[list[dict]], k: int = 10, rrf_k: int = 60
) -> list[dict]:
    """Combine multiple ranked result lists into one, by Reciprocal Rank Fusion.

    rrf_k=60 is the standard literature default (Cormack et al., 2009). The
    source system this pipeline was extracted from tunes it lower (RRF_K=10)
    for its specific corpus — that's a deliberate choice for that system, not
    reproduced here; 60 is the right value to start from on an unfamiliar
    corpus.

    RRF score for a document = sum over lists of 1 / (rrf_k + rank), where rank
    is that document's 0-based position in each list it appears in. Documents
    absent from a list contribute nothing from it. This is rank-based (not
    score-based), which is what makes it safe to combine signals on
    incomparable scales — vector cosine similarity and FTS5 BM25-derived rank
    don't live on the same scale, but "how high did each method rank this doc"
    always does.
    """
    fused: dict[int, dict] = {}
    for results in result_lists:
        for rank, r in enumerate(results):
            key = r["id"]
            if key not in fused:
                fused[key] = {"score": 0.0, "result": r}
            fused[key]["score"] += 1.0 / (rrf_k + rank + 1)

    merged = []
    for entry in fused.values():
        entry["result"]["rrf_score"] = entry["score"]
        merged.append(entry["result"])
    merged.sort(key=lambda x: x["rrf_score"], reverse=True)
    return merged[:k]


def hybrid_search(index: Index, query: str, top_k: int = 10, pool: int = 30) -> list[dict]:
    """Vector search + keyword search, fused with RRF. No reranking (see reranking.py)."""
    vec_results = index.vector_search(query, pool)
    kw_results = index.keyword_search(query, pool)
    return reciprocal_rank_fusion([vec_results, kw_results], k=top_k)
