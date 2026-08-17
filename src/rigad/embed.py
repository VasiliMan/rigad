"""Turn researcher profile text into vectors.

Two backends, same interface:

``neural``  a small multilingual sentence-transformer. Better at recognising
            that "digital transformation" and "IT-enabled organisational
            change" are the same thing. Multilingual matters for a European
            alliance where not everyone publishes in English.
``tfidf``   TF-IDF plus SVD, using only scikit-learn. Requires no model
            download and no torch, so the notebooks run anywhere. It is also
            a legitimate baseline: comparing the two tells us whether the
            neural model is actually earning its install size.

All vectors are L2-normalised, so cosine similarity is a plain dot product.
"""

from __future__ import annotations

import logging
import os
import warnings

import numpy as np

# The sentence-transformers stack is chatty on import and on first load: a
# weight-loading progress bar, a notice about unauthenticated Hugging Face
# requests, and a tokenizer-parallelism caveat. None of it is actionable for
# someone who just wants to know which track their paper fits, and the model
# is served from a local cache after the first run. Quietened here, before the
# libraries are imported, because several of these are read at import time.
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def quieten_model_loading() -> None:
    """Silence progress bars and hub notices from the embedding libraries."""
    for name in ("transformers", "sentence_transformers", "huggingface_hub"):
        logging.getLogger(name).setLevel(logging.ERROR)
    warnings.filterwarnings(
        "ignore", message=".*unauthenticated requests.*", module=".*huggingface_hub.*"
    )
    try:
        from huggingface_hub.utils import logging as hub_logging

        hub_logging.disable_progress_bars()
    except Exception:  # noqa: BLE001 - older/newer hub versions differ
        pass

# Small, fast, multilingual. ~470MB on disk; the first call downloads it.
NEURAL_MODEL = "intfloat/multilingual-e5-small"

# e5 models are trained with two prefixes and degrade noticeably without them.
# Which one to use depends on the comparison:
#
#   symmetric  — researcher against researcher, draft against draft. Both
#                sides are the same kind of thing, so both get "query: ".
#   asymmetric — a draft searching for the track it belongs in. The draft is
#                the query; the track description is the passage being
#                retrieved. Using "query: " for both collapses the score range
#                and makes every track look equally close.
E5_PREFIXES = {"query": "query: ", "passage": "passage: "}

# Profiles can run to tens of thousands of characters. Sentence transformers
# truncate to their context window anyway; cutting here keeps things fast and
# makes the two backends see comparable amounts of text.
MAX_CHARS = 4000


quieten_model_loading()


def embed_texts(
    texts: list[str],
    backend: str = "auto",
    *,
    kind: str = "query",
    n_components: int = 256,
    seed: int = 0,
) -> tuple[np.ndarray, str]:
    """Embed texts and report which backend was actually used.

    ``kind`` is ``"query"`` or ``"passage"`` and only affects the neural
    backend — use ``"passage"`` for the side being searched (track
    descriptions), ``"query"`` for the side doing the searching (a draft), and
    ``"query"`` for both when comparing like with like.

    ``backend="auto"`` prefers the neural model and silently falls back to
    TF-IDF if sentence-transformers is not installed, so a fresh checkout
    always works.

    Returns ``(vectors, backend_used)`` with vectors shaped (len(texts), dim).
    """
    trimmed = [t[:MAX_CHARS] for t in texts]

    if backend == "auto":
        try:
            import sentence_transformers  # noqa: F401

            backend = "neural"
        except ImportError:
            backend = "tfidf"

    if backend == "neural":
        return _embed_neural(trimmed, kind=kind), "neural"
    if backend == "tfidf":
        return _embed_tfidf(trimmed, n_components=n_components, seed=seed), "tfidf"
    raise ValueError(f"unknown backend {backend!r}; expected 'neural', 'tfidf' or 'auto'")


def _embed_neural(texts: list[str], *, kind: str = "query") -> np.ndarray:
    quieten_model_loading()
    from sentence_transformers import SentenceTransformer

    try:
        prefix = E5_PREFIXES[kind]
    except KeyError:
        raise ValueError(f"unknown kind {kind!r}; expected 'query' or 'passage'") from None

    model = SentenceTransformer(NEURAL_MODEL)
    vectors = model.encode(
        [prefix + t for t in texts],
        normalize_embeddings=True,
        show_progress_bar=False,
        batch_size=32,
    )
    return np.asarray(vectors, dtype=np.float32)


class Embedder:
    """An embedder fitted on a reference corpus, able to embed new text into it.

    This exists because the two backends differ in a way that matters. The
    neural model is fixed: any text can be embedded at any time and the results
    are comparable. TF-IDF is not — it learns a vocabulary from whatever it is
    fitted on, so vectors from two separate calls live in two unrelated spaces
    and comparing them is meaningless.

    Track matching compares drafts against track descriptions, which are
    embedded at different times. It therefore needs a fitted, reusable
    embedder rather than a one-shot function.
    """

    def __init__(self, backend: str, model=None, vectorizer=None, svd=None) -> None:
        self.backend = backend
        self._model = model
        self._vectorizer = vectorizer
        self._svd = svd

    def transform(self, texts: list[str], *, kind: str = "query") -> np.ndarray:
        trimmed = [t[:MAX_CHARS] for t in texts]
        if self.backend == "neural":
            vectors = self._model.encode(
                [E5_PREFIXES[kind] + t for t in trimmed],
                normalize_embeddings=True,
                show_progress_bar=False,
                batch_size=32,
            )
            return normalize_rows(np.asarray(vectors, dtype=np.float32))

        # TF-IDF: reuse the vocabulary and projection learned when fitting.
        sparse = self._vectorizer.transform(trimmed)
        dense = self._svd.transform(sparse)
        return normalize_rows(np.asarray(dense, dtype=np.float32))

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"Embedder(backend={self.backend!r})"


def fit_embedder(
    texts: list[str],
    backend: str = "auto",
    *,
    n_components: int = 256,
    seed: int = 0,
) -> Embedder:
    """Fit an embedder on a reference corpus so later text lands in that space."""
    if backend == "auto":
        try:
            import sentence_transformers  # noqa: F401

            backend = "neural"
        except ImportError:
            backend = "tfidf"

    if backend == "neural":
        quieten_model_loading()
        from sentence_transformers import SentenceTransformer

        return Embedder("neural", model=SentenceTransformer(NEURAL_MODEL))

    if backend == "tfidf":
        from sklearn.decomposition import TruncatedSVD
        from sklearn.feature_extraction.text import TfidfVectorizer

        trimmed = [t[:MAX_CHARS] for t in texts]
        # min_df=1 here: the reference corpus can be small (a few dozen track
        # facets), and dropping every term that appears once would empty it.
        vectorizer = TfidfVectorizer(
            stop_words="english", max_features=20_000, ngram_range=(1, 2), min_df=1
        )
        sparse = vectorizer.fit_transform(trimmed)
        k = min(n_components, min(sparse.shape) - 1)
        svd = TruncatedSVD(n_components=max(1, k), random_state=seed).fit(sparse)
        return Embedder("tfidf", vectorizer=vectorizer, svd=svd)

    raise ValueError(f"unknown backend {backend!r}; expected 'neural', 'tfidf' or 'auto'")


def center(vectors: np.ndarray, mean: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Remove the corpus mean direction, then re-normalise.

    Sentence embeddings are anisotropic: they cluster in a narrow cone, so
    every pair looks similar and score differences are tiny. Subtracting the
    mean direction spreads them out and makes the ranking far more decisive.
    It changes the absolute numbers — a centred cosine is not comparable to an
    uncentred one — but preserves and sharpens the ordering, which is what a
    ranked recommendation actually depends on.

    Pass a ``mean`` computed elsewhere to centre two sets consistently.
    """
    if mean is None:
        mean = vectors.mean(axis=0)
    return normalize_rows(vectors - mean), mean


def _embed_tfidf(texts: list[str], *, n_components: int, seed: int) -> np.ndarray:
    from sklearn.decomposition import TruncatedSVD
    from sklearn.feature_extraction.text import TfidfVectorizer

    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=20_000,
        ngram_range=(1, 2),
        min_df=2,
    )
    sparse = vectorizer.fit_transform(texts)

    # SVD cannot produce more components than the data has dimensions.
    k = min(n_components, min(sparse.shape) - 1)
    dense = TruncatedSVD(n_components=k, random_state=seed).fit_transform(sparse)
    return normalize_rows(np.asarray(dense, dtype=np.float32))


def normalize_rows(matrix: np.ndarray) -> np.ndarray:
    """Scale each row to unit length, leaving degenerate rows as zeros.

    Defensive because the input is whatever text a user's documents happened
    to contain. A draft that shares no vocabulary with the reference corpus
    embeds to all zeros, and a corrupt or empty extraction can yield NaN; both
    would otherwise propagate into the similarity matmul and surface to the
    user as "divide by zero encountered in matmul", which is alarming and
    tells them nothing. A zero vector simply scores zero against everything,
    which is the honest answer for a document we could not read.
    """
    matrix = np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


def similarity_matrix(vectors: np.ndarray) -> np.ndarray:
    """Pairwise cosine similarity. Assumes rows are already normalised."""
    return vectors @ vectors.T
