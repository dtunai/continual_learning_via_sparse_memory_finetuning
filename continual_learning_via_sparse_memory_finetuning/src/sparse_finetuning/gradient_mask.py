"""
gradient masking for sparse memory updates

zeros out gradients for all memory indices except the selected top-t

uses pytorch backward hooks
"""

from logging import getLogger
from typing import Optional
import torch
import torch.nn as nn

logger = getLogger(__name__)


class SparseMemoryGradientMask:
    """
    masks gradients so only selected indices get updates.
    call set_trainable_indices() before each backward pass.
    """

    def __init__(self, memory_values_param: nn.Parameter):
        self.memory_values_param = memory_values_param
        self.trainable_indices: Optional[torch.Tensor] = None
        self.hook_handle = None
        self.device = memory_values_param.device

    def set_trainable_indices(self, indices: torch.Tensor):
        """set which indices should receive gradients this step."""
        self.trainable_indices = indices.to(self.device)

    def gradient_mask_hook(self, grad: torch.Tensor) -> torch.Tensor:
        """hook that zeros non-trainable indices."""
        if self.trainable_indices is None:
            return grad

        mask = torch.zeros_like(grad)
        mask[self.trainable_indices] = 1.0
        return grad * mask

    def register_hook(self):
        if self.hook_handle is not None:
            self.hook_handle.remove()

        self.hook_handle = self.memory_values_param.register_hook(
            self.gradient_mask_hook
        )
        logger.info("registered gradient mask hook")

    def remove_hook(self):
        if self.hook_handle is not None:
            self.hook_handle.remove()
            self.hook_handle = None
            logger.info("removed gradient mask hook")

    def __del__(self):
        self.remove_hook()
