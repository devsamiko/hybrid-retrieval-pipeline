# Vector Databases and Approximate Nearest Neighbor Search

A vector database stores high-dimensional embeddings and answers nearest
neighbor queries: given a query vector, return the k stored vectors closest
to it by some distance metric, usually cosine similarity or dot product.

Exact nearest neighbor search requires comparing the query against every
stored vector, which is linear in the size of the index and becomes slow
past a few hundred thousand vectors. Approximate nearest neighbor (ANN)
algorithms trade a small amount of recall for a large speedup — HNSW
(Hierarchical Navigable Small World graphs) is a common choice, building a
multi-layer graph structure that lets search jump quickly toward the
right neighborhood instead of scanning linearly.

In practice, many systems keep a lightweight exact-search fallback (a flat
index) alongside the ANN index, either for small collections where the
speed difference doesn't matter, or as a resilience measure if the primary
ANN-backed store becomes unavailable.
