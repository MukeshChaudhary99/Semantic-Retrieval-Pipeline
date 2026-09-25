import threading
from typing import Iterable

from fastembed import SparseTextEmbedding
from sentence_transformers import SentenceTransformer

from ..core.config import get_settings


class SentenceTransformerModel:
    """Dense embedding model wrapper."""

    def __init__(self, params: dict):
        self.params = params
        self.embedding_model = self.initialize_model()

    def initialize_model(self) -> SentenceTransformer:
        token = get_settings().hf_token or None
        return SentenceTransformer(self.params["model"], token=token)

    def __call__(self, input: list[str]) -> list[list[float]]:
        return self.embedding_model.encode(input, convert_to_tensor=False).tolist()

    def get_llm(self) -> SentenceTransformer:
        return self.embedding_model

    def model_name(self) -> str:
        return self.params["model"]

    def model_type(self) -> str:
        return "SentenceTransformer"


# models are heavy — load once, guard against races between background tasks
_lock = threading.Lock()
_dense: SentenceTransformerModel | None = None
_sparse: SparseTextEmbedding | None = None


def get_dense_encoder() -> SentenceTransformerModel:
    global _dense
    if _dense is None:
        with _lock:
            if _dense is None:
                _dense = SentenceTransformerModel(
                    {"model": get_settings().dense_model}
                )
    return _dense


def get_sparse_encoder() -> SparseTextEmbedding:
    global _sparse
    if _sparse is None:
        with _lock:
            if _sparse is None:
                _sparse = SparseTextEmbedding(model_name=get_settings().sparse_model)
    return _sparse


def embed_dense(texts: list[str]) -> list[list[float]]:
    return get_dense_encoder()(texts)


def embed_sparse(texts: Iterable[str]) -> list[dict]:
    """Return [{'indices': [...], 'values': [...]}, ...] in Qdrant format."""
    model = get_sparse_encoder()
    out = []
    for emb in model.embed(list(texts)):
        out.append(
            {
                "indices": emb.indices.tolist(),
                "values": emb.values.tolist(),
            }
        )
    return out
