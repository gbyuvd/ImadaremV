"""
Numerically-stable Chebyshev learning rate scheduler.
Implements the same interface as torch.optim.lr_scheduler._LRScheduler,
and provides helper functions compatible with Ranger21 optimizer.

Original implementation is from https://arxiv.org/abs/2103.01338v1
"""

import math
import torch
from torch.optim.lr_scheduler import _LRScheduler


class ChebyshevLR(_LRScheduler):
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        total_steps: int,
        lr_start: float = 1.0,
        lr_end: float = 0.0,
        last_epoch: int = -1,
        min_lr: float = 1e-12,
    ):
        if total_steps <= 0:
            raise ValueError("total_steps must be > 0 for ChebyshevLR")
        self.total_steps = int(total_steps)
        self.lr_start = float(lr_start)
        self.lr_end = float(lr_end)
        self.min_lr = float(min_lr)
        super().__init__(optimizer, last_epoch)

    def _alpha_at_step(self, step: int) -> float:
        """Stable Chebyshev evaluation returning α in [0,1] for interpolation."""
        t = min(max(float(step) / float(max(1, self.total_steps)), 0.0), 1.0)
        x = 1.0 - 2.0 * t  # map t ∈ [0,1] to x ∈ [1, -1]
        n = self.total_steps
        if abs(x) <= 1.0:
            theta = math.acos(x)
            Tn = math.cos(n * theta)
        else:
            # Fallback recurrence (should be rare)
            x64 = float(x)
            T0, T1 = 1.0, x64
            for _ in range(2, n + 1):
                T2 = 2.0 * x64 * T1 - T0
                T0, T1 = T1, T2
            Tn = T1
        alpha = 0.5 * (Tn + 1.0)
        return min(max(alpha, 0.0), 1.0)

    def get_lr(self):
        step = max(0, self.last_epoch)
        alpha = self._alpha_at_step(step)
        out = []
        for base_lr in self.base_lrs:
            lr = self.lr_end + (self.lr_start - self.lr_end) * alpha
            if lr < self.min_lr:
                lr = self.min_lr
            out.append(lr)
        return out

    @staticmethod
    def total_steps_from(epochs: int, steps_per_epoch: int) -> int:
        """Compute total steps given epochs and batches per epoch."""
        return int(max(1, int(epochs) * int(steps_per_epoch)))


# ================================================================
# Ranger21-Compatible Helper Functions
# ================================================================

def get_chebs(num_epochs: int, steps_per_epoch: int = 1):
    """Generate a Chebyshev LR schedule array for the given epoch length."""
    total_steps = ChebyshevLR.total_steps_from(num_epochs, steps_per_epoch)
    schedule = []
    for step in range(total_steps):
        t = step / total_steps
        x = 1.0 - 2.0 * t
        theta = math.acos(max(-1.0, min(1.0, x)))
        Tn = math.cos(total_steps * theta)
        alpha = 0.5 * (Tn + 1.0)
        schedule.append(alpha)
    return torch.tensor(schedule, dtype=torch.float32)


def get_cheb_lr(lr: float, iteration: int, num_batches_per_epoch: int, num_epochs: int):
    """
    Compute the learning rate multiplier for the given iteration and training span.
    This version mirrors Ranger21's expected API.
    """
    total_steps = ChebyshevLR.total_steps_from(num_epochs, num_batches_per_epoch)
    step = max(0, min(iteration, total_steps - 1))
    t = step / total_steps
    x = 1.0 - 2.0 * t
    theta = math.acos(max(-1.0, min(1.0, x)))
    Tn = math.cos(total_steps * theta)
    alpha = 0.5 * (Tn + 1.0)
    return lr * alpha


__all__ = ["ChebyshevLR", "get_chebs", "get_cheb_lr"]

