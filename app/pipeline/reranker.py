"""Cross-encoder reranking via sentence-transformers CrossEncoder."""

import threading

from sentence_transformers import CrossEncoder

from ..core.config import get_settings


class CrossEncoderModel:
    """Reranker model wrapper — scores (query, document) relevance pairs."""

    def __init__(self, params: dict):
        self.params = params
        self.reranker_model = self.initialize_model()

    def initialize_model(self) -> CrossEncoder:
        token = get_settings().hf_token or None
        return CrossEncoder(self.params["model"], token=token)

    def __call__(self, query: str, documents: list[str]) -> list[float]:
        scores = self.reranker_model.predict([(query, doc) for doc in documents])
        return [float(s) for s in scores]

    def rank(self, query: str, documents: list[str], top_k: int | None = None):
        return self.reranker_model.rank(query, documents, top_k=top_k)

    def get_llm(self) -> CrossEncoder:
        return self.reranker_model

    def model_name(self) -> str:
        return self.params["model"]

    def model_type(self) -> str:
        return "CrossEncoder"


_lock = threading.Lock()
_reranker: CrossEncoderModel | None = None


def get_reranker() -> CrossEncoderModel:
    global _reranker
    if _reranker is None:
        with _lock:
            if _reranker is None:
                _reranker = CrossEncoderModel(
                    {"model": get_settings().reranker_model}
                )
    return _reranker


def rerank(query: str, texts: list[str]) -> list[float]:
    """Return a relevance score per text (higher = more relevant)."""
    if not texts:
        return []
    return get_reranker()(query, texts)
