"""
tf-idf ranking for memory indices

ranks indices by how task-specific they are: high tf-idf means
frequently accessed in this batch but rare in background corpus
"""

from logging import getLogger
from typing import Dict, Set, Optional
from collections import defaultdict
import numpy as np
import torch

logger = getLogger(__name__)


class BackgroundCorpusIndexTracker:
    """
    tracks which memory indices appear in background corpus batches.
    used to compute idf (inverse document frequency).
    """

    def __init__(self, num_batches: int = 1000):
        self.num_batches = num_batches
        self.index_to_batches: Dict[int, Set[int]] = defaultdict(set)
        self.current_batch_id = 0
        self.is_collecting = False

    def start_collection(self):
        self.is_collecting = True
        self.current_batch_id = 0
        self.index_to_batches.clear()
        logger.info("started background index collection")

    def stop_collection(self):
        self.is_collecting = False
        logger.info(
            f"collected {len(self.index_to_batches)} unique indices "
            f"from {self.current_batch_id} batches"
        )

    def add_batch_indices(self, indices: torch.Tensor):
        """add indices from one batch. flattens input tensor."""
        if not self.is_collecting or self.current_batch_id >= self.num_batches:
            return

        unique_indices = torch.unique(indices.flatten().cpu()).tolist()
        for idx in unique_indices:
            self.index_to_batches[idx].add(self.current_batch_id)

        self.current_batch_id += 1
        if self.current_batch_id >= self.num_batches:
            self.stop_collection()

    def get_document_frequency(self, index: int) -> int:
        """returns number of batches where this index appeared."""
        return len(self.index_to_batches.get(index, set()))

    def save(self, path: str):
        torch.save(
            {
                "index_to_batches": {
                    k: list(v) for k, v in self.index_to_batches.items()
                },
                "num_batches": self.current_batch_id,
            },
            path,
        )
        logger.info(f"saved background indices to {path}")

    def load(self, path: str):
        data = torch.load(path, weights_only=False)
        self.index_to_batches = defaultdict(
            set, {k: set(v) for k, v in data["index_to_batches"].items()}
        )
        self.current_batch_id = data["num_batches"]
        logger.info(f"loaded {len(self.index_to_batches)} indices from {path}")


class TFIDFRanker:
    """
    ranks memory indices by tf-idf score.

    formula (from paper section 4):
        tf-idf(i) = [c(i) / sum(c)] * log[(|B| + 1) / (df(i) + 1)]

    where c(i) = count in batch, |B| = num background batches,
    df(i) = document frequency (batches containing i).
    """

    def __init__(
        self,
        background_tracker: BackgroundCorpusIndexTracker,
        use_idf: bool = True,
        smoothing: float = 1.0,
    ):
        self.background_tracker = background_tracker
        self.use_idf = use_idf
        self.smoothing = smoothing
        self.num_background_batches = background_tracker.current_batch_id

    def rank_indices(
        self,
        batch_indices: torch.Tensor,
        top_k: int,
    ) -> torch.Tensor:
        """
        returns top-k indices sorted by tf-idf (highest first).
        input tensor is flattened before processing.
        """
        flat_indices = batch_indices.flatten().cpu()
        unique_indices, counts = torch.unique(flat_indices, return_counts=True)

        # term frequency: normalized count
        tf = counts.float() / counts.sum()

        if self.use_idf:
            idf_scores = []
            for idx in unique_indices.tolist():
                df = self.background_tracker.get_document_frequency(idx)
                idf = np.log(
                    (self.num_background_batches + self.smoothing)
                    / (df + self.smoothing)
                )
                idf_scores.append(idf)

            idf_scores = torch.tensor(idf_scores, dtype=tf.dtype)
            tfidf_scores = tf * idf_scores
        else:
            tfidf_scores = tf

        # always sort by score, return top-k
        k = min(top_k, len(tfidf_scores))
        _, top_positions = torch.topk(tfidf_scores, k=k, largest=True)
        return unique_indices[top_positions]
