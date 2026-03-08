from typing import Tuple

import numpy as np


class DenseVectorIndex:
    """
    A minimal, fully in-Python vector index for semantic search.

    - Stores a normalized embedding matrix of shape [N, D].
    - Performs cosine similarity search via a single matrix-vector multiply.
    """

    def __init__(self, embeddings: np.ndarray):
        assert embeddings.ndim == 2, "Expected [N, D] embeddings"
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-12
        self._embeddings = (embeddings / norms).astype("float32")

    @property
    def size(self) -> int:
        return self._embeddings.shape[0]

    @property
    def dim(self) -> int:
        return self._embeddings.shape[1]

    def search(
        self,
        query_vec: np.ndarray,
        top_k: int = 5,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return (similarities, indices) of the top_k most similar documents.
        - query_vec must already be L2-normalized.
        - similarities are cosine similarities in [-1, 1].
        """
        assert query_vec.ndim == 1 and query_vec.shape[0] == self.dim

        sims = self._embeddings @ query_vec  # [N]
        if top_k >= sims.shape[0]:
            idx = np.argsort(-sims)
        else:
            top_idx = np.argpartition(-sims, top_k)[:top_k]
            idx = top_idx[np.argsort(-sims[top_idx])]

        return sims[idx], idx
