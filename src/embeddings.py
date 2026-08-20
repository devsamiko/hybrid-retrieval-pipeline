"""
embeddings.py — Embedding interface
====================================
Thin wrapper around sentence-transformers. The original system routed this
through a shared embedding-server process (to avoid loading the model twice
across concurrent processes); that's a deployment optimization, not part of
the retrieval algorithm, so this standalone version calls the model directly.
"""

from functools import lru_cache

# The source system runs bge-large-en-v1.5. This repo defaults to bge-small
# instead — smaller download, faster to run for anyone cloning this to try
# it out, and the demo corpus is tiny enough that model capacity isn't the
# bottleneck. Swap DEFAULT_MODEL to bge-large for closer parity with the
# source system's numbers.
DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"


@lru_cache(maxsize=1)
def _get_model(model_name: str = DEFAULT_MODEL):
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(model_name)


def embed_text(text: str, model_name: str = DEFAULT_MODEL) -> list[float]:
    """Embed a single text string."""
    model = _get_model(model_name)
    return model.encode(text, normalize_embeddings=True).tolist()


def embed_texts(texts: list[str], model_name: str = DEFAULT_MODEL) -> list[list[float]]:
    """Batch embed multiple texts."""
    model = _get_model(model_name)
    return model.encode(texts, normalize_embeddings=True, show_progress_bar=False).tolist()
