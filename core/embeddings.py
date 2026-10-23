"""
embeddings.py
-------------
Three interchangeable embedding backends behind one interface, picked
automatically by config.active_backend():

- GeminiEmbedder: Google's gemini-embedding-001 via the free-tier-capable
  Gemini API (aistudio.google.com key required — NOT the same as a
  Gemini Pro app subscription).

- OpenAIEmbedder: OpenAI's text-embedding-3-small.

- LocalEmbedder: a zero-cost, zero-external-call fallback using
  scikit-learn's HashingVectorizer. Not "semantic" in the transformer
  sense, but fixed-dimension, needs no training/download, and keeps the
  whole app runnable with no API key at all.

embed_texts() is the single function the rest of the app calls — nothing
else in the codebase needs to know which backend is active.
"""
import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer
import config

_LOCAL_DIM = 512
_local_vectorizer = HashingVectorizer(
    n_features=_LOCAL_DIM,
    norm="l2",
    alternate_sign=False,
)


def _embed_local(texts):
    matrix = _local_vectorizer.transform(texts)
    return matrix.toarray().astype("float32")


def _embed_openai(texts):
    from openai import OpenAI
    client = OpenAI(api_key=config.OPENAI_API_KEY)
    response = client.embeddings.create(
        model=config.OPENAI_EMBED_MODEL,
        input=texts,
    )
    vectors = [item.embedding for item in response.data]
    return np.array(vectors, dtype="float32")


def _embed_gemini(texts):
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=config.GEMINI_API_KEY)
    result = client.models.embed_content(
        model=config.GEMINI_EMBED_MODEL,
        contents=texts,
        config=types.EmbedContentConfig(
            output_dimensionality=config.GEMINI_EMBED_DIM
        ),
    )
    vectors = [e.values for e in result.embeddings]
    return np.array(vectors, dtype="float32")


def embed_texts(texts):
    """
    texts: list[str]
    returns: np.ndarray of shape (len(texts), dim), dtype float32
    """
    if not texts:
        return np.zeros((0, embedding_dim()), dtype="float32")

    backend = config.active_backend()
    if backend == "gemini":
        return _embed_gemini(texts)
    if backend == "openai":
        return _embed_openai(texts)
    return _embed_local(texts)


def embedding_dim():
    backend = config.active_backend()
    if backend == "gemini":
        return config.GEMINI_EMBED_DIM
    if backend == "openai":
        return 1536
    return _LOCAL_DIM


def backend_name():
    return config.active_backend()
