from __future__ import annotations

from dataclasses import dataclass
from threading import Lock
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class CacheEntry:
    query: str
    embedding: np.ndarray
    result: str
    cluster_distribution: np.ndarray


class SemanticCache:
    """
    Cluster-aware, from-scratch semantic cache.

    - Entries partitioned by dominant cluster (argmax of soft assignment).
    - Lookup:
      1) Use soft cluster distribution for incoming query.
      2) Search only top-M clusters by membership probability.
      3) Within each, compute cosine sims against cached embeddings.
      4) If best_sim >= similarity_threshold, reuse.
    """

    def __init__(
        self,
        embedding_dim: int,
        num_clusters: int,
        similarity_threshold: float = 0.85,
        max_entries_per_cluster: int = 1000,
        top_m_clusters: int = 3,
    ):
        self.embedding_dim = embedding_dim
        self.num_clusters = num_clusters
        self.similarity_threshold = float(similarity_threshold)
        self.max_entries_per_cluster = int(max_entries_per_cluster)
        self.top_m_clusters = int(top_m_clusters)

        self._cluster_embeddings: Dict[int, List[np.ndarray]] = {
            c: [] for c in range(num_clusters)
        }
        self._cluster_entries: Dict[int, List[CacheEntry]] = {
            c: [] for c in range(num_clusters)
        }

        self._hit_count = 0
        self._miss_count = 0

        self._lock = Lock()

    def lookup(
        self,
        query_embedding: np.ndarray,
        cluster_distribution: np.ndarray,
    ) -> Tuple[bool, Optional[CacheEntry], float, int]:
        """
        Returns:
          (cache_hit, entry_or_None, similarity_score, dominant_cluster_for_match)
        """
        assert query_embedding.ndim == 1
        assert cluster_distribution.ndim == 1
        assert cluster_distribution.shape[0] == self.num_clusters

        dominant_cluster = int(np.argmax(cluster_distribution))

        with self._lock:
            best_sim = -1.0
            best_entry: Optional[CacheEntry] = None

            candidate_clusters = np.argsort(-cluster_distribution)[
                : self.top_m_clusters
            ]

            for c in candidate_clusters:
                c = int(c)
                if not self._cluster_embeddings[c]:
                    continue

                mat = np.stack(self._cluster_embeddings[c], axis=0)  # [Nc, D]
                sims = mat @ query_embedding
                idx = int(np.argmax(sims))
                sim = float(sims[idx])
                if sim > best_sim:
                    best_sim = sim
                    best_entry = self._cluster_entries[c][idx]
                    dominant_cluster = c

            if best_entry is not None and best_sim >= self.similarity_threshold:
                self._hit_count += 1
                return True, best_entry, best_sim, dominant_cluster

            self._miss_count += 1
            best_sim = max(best_sim, 0.0)
            return False, None, best_sim, dominant_cluster

    def insert(
        self,
        query: str,
        embedding: np.ndarray,
        cluster_distribution: np.ndarray,
        result: str,
    ) -> int:
        assert embedding.ndim == 1
        assert cluster_distribution.ndim == 1
        dominant_cluster = int(np.argmax(cluster_distribution))

        entry = CacheEntry(
            query=query,
            embedding=embedding.astype("float32"),
            result=result,
            cluster_distribution=cluster_distribution.astype("float32"),
        )

        with self._lock:
            emb_list = self._cluster_embeddings[dominant_cluster]
            entry_list = self._cluster_entries[dominant_cluster]

            if len(entry_list) >= self.max_entries_per_cluster:
                emb_list.pop(0)
                entry_list.pop(0)

            emb_list.append(entry.embedding)
            entry_list.append(entry)

        return dominant_cluster

    def stats(self) -> dict:
        with self._lock:
            hits = self._hit_count
            misses = self._miss_count
            total = hits + misses
            hit_rate = (hits / total) if total > 0 else 0.0
            total_entries = sum(
                len(lst) for lst in self._cluster_entries.values()
            )

        return {
            "total_entries": total_entries,
            "hit_count": hits,
            "miss_count": misses,
            "hit_rate": hit_rate,
            "similarity_threshold": self.similarity_threshold,
        }

    def flush(self):
        with self._lock:
            for c in range(self.num_clusters):
                self._cluster_embeddings[c].clear()
                self._cluster_entries[c].clear()
            self._hit_count = 0
            self._miss_count = 0
