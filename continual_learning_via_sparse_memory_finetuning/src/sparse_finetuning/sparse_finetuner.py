"""
sparse memory finetuner

coordinates background tracking, tf-idf ranking and gradient masking
"""

from dataclasses import dataclass
from logging import getLogger
from typing import Dict, Optional
import torch
import torch.nn as nn

from .tfidf_ranker import TFIDFRanker, BackgroundCorpusIndexTracker
from .gradient_mask import SparseMemoryGradientMask

logger = getLogger(__name__)


@dataclass
class SparseMemoryFinetuningArgs:
    """config for sparse memory finetuning."""

    enabled: bool = False
    top_t: int = 500  # number of indices to update per batch
    background_indices_path: Optional[str] = None
    num_background_batches: int = 1000
    use_idf: bool = True
    idf_smoothing: float = 1.0


class SparseMemoryFinetuner:
    """
    main class for sparse memory finetuning.

    usage:
        finetuner = SparseMemoryFinetuner(args, memory_values)

        # collect background (once)
        finetuner.background_tracker.start_collection()
        for batch in background_loader:
            indices = get_memory_indices(model, batch)
            finetuner.background_tracker.add_batch_indices(indices)
        finetuner.save_background_indices("bg.pt")

        # training
        finetuner.load_background_indices("bg.pt")
        for batch in train_loader:
            indices = get_memory_indices(model, batch)
            finetuner.update_trainable_indices(indices)
            loss.backward()  # only top-t get gradients
            optimizer.step()
    """

    def __init__(
        self,
        args: SparseMemoryFinetuningArgs,
        memory_values_param: Optional[nn.Parameter] = None,
    ):
        self.args = args
        self.memory_values_param = memory_values_param

        self.background_tracker = BackgroundCorpusIndexTracker(
            num_batches=args.num_background_batches
        )

        self.ranker: Optional[TFIDFRanker] = None

        self.gradient_mask: Optional[SparseMemoryGradientMask] = None
        if memory_values_param is not None:
            self.gradient_mask = SparseMemoryGradientMask(memory_values_param)
            if args.enabled:
                self.gradient_mask.register_hook()
                logger.info("sparse finetuning enabled")

    def collect_background_indices(
        self,
        dataloader,
        model,
        max_batches: Optional[int] = None,
    ):
        """collect indices from background corpus (e.g. dclm)."""
        if max_batches is not None:
            self.background_tracker.num_batches = max_batches

        self.background_tracker.start_collection()
        model.eval()

        logger.info(f"collecting {self.background_tracker.num_batches} batches...")

        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                if batch_idx >= self.background_tracker.num_batches:
                    break

                _, memory_indices = model(batch, return_memory_indices=True)
                self.background_tracker.add_batch_indices(memory_indices)

                if (batch_idx + 1) % 100 == 0:
                    logger.info(f"collected {batch_idx + 1} batches")

        self.background_tracker.stop_collection()
        self._create_ranker()

    def _create_ranker(self):
        self.ranker = TFIDFRanker(
            self.background_tracker,
            use_idf=self.args.use_idf,
            smoothing=self.args.idf_smoothing,
        )

    def save_background_indices(self, path: str):
        self.background_tracker.save(path)

    def load_background_indices(self, path: str):
        self.background_tracker.load(path)
        self._create_ranker()

    def update_trainable_indices(self, batch_memory_indices: torch.Tensor):
        """
        call after forward, before backward.
        ranks indices by tf-idf and sets top-t as trainable.
        """
        if not self.args.enabled or self.ranker is None:
            return

        if self.gradient_mask is None:
            logger.warning("no gradient mask configured")
            return

        top_indices = self.ranker.rank_indices(
            batch_memory_indices, top_k=self.args.top_t
        )
        self.gradient_mask.set_trainable_indices(top_indices)

    def get_stats(self) -> Dict[str, float]:
        stats = {}

        if self.gradient_mask and self.gradient_mask.trainable_indices is not None:
            num_trainable = len(self.gradient_mask.trainable_indices)
            total = self.memory_values_param.size(0)
            stats["num_trainable_indices"] = num_trainable
            stats["total_memory_indices"] = total
            stats["trainable_pct"] = 100.0 * num_trainable / total

        if self.ranker:
            stats["num_background_batches"] = self.ranker.num_background_batches

        return stats

    def cleanup(self):
        if self.gradient_mask:
            self.gradient_mask.remove_hook()
