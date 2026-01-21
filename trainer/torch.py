from bisect import bisect_right
from collections.abc import Iterator

import torch
from torch.utils.data.distributed import DistributedSampler

from trainer.generic_utils import is_pytorch_at_least_2_4

if is_pytorch_at_least_2_4():
    from torch.optim.lr_scheduler import _warn_get_lr_called_within_step


class DistributedSamplerWrapper(DistributedSampler):
    """Wrapper over Sampler for distributed training.

    It allows you to use any sampler in distributed mode.
    It is especially useful in conjunction with torch.nn.parallel.DistributedDataParallel. In such a case, each
    process can pass a torch.utils.data.DistributedSampler instance as a torch.utils.data.DataLoader sampler,
    and load a subset of the original dataset that is exclusive to it.

    .. note:
        Dataset is assumed to be of constant size.

    Args:
        sampler: Sampler used for subsampling.
        num_replicas (int, optional): Number of processes participating in distributed training. By default,
            world_size is retrieved from the current distributed group.
        rank (int, optional): Rank of the current process within num_replicas. By default, rank is retrieved
            from the current distributed group.
        shuffle (bool, optional): If True, sampler will shuffle the indices. Default: True.
        seed (int, optional): random seed used to shuffle the sampler if shuffle=True. This number should be
            identical across all processes in the distributed group. Default: 0.

    Reference: https://github.com/pytorch/pytorch/issues/23430

    """

    def __init__(
        self,
        sampler,
        *,
        num_replicas: int | None = None,
        rank: int | None = None,
        shuffle: bool = True,
        seed: int = 0,
    ) -> None:
        super().__init__(
            sampler,
            num_replicas=num_replicas,
            rank=rank,
            shuffle=shuffle,
            seed=seed,
        )

    def __iter__(self) -> Iterator:
        indices = list(self.dataset)[: self.total_size]  # type: ignore[call-overload]

        # Add extra samples to make it evenly divisible
        indices += indices[: (self.total_size - len(indices))]
        assert len(indices) == self.total_size, f"{len(indices)} != {self.total_size}"

        # Subsample
        offset = self.num_samples * self.rank
        indices = indices[offset : offset + self.num_samples]
        assert len(indices) == self.num_samples, f"{len(indices)} != {self.num_samples}"

        return iter(indices)

    def set_epoch(self, epoch: int) -> None:
        super().set_epoch(epoch)
        if hasattr(self.dataset, "set_epoch"):
            self.dataset.set_epoch(epoch)
        elif hasattr(self.dataset, "generator"):
            self.dataset.generator = torch.Generator().manual_seed(self.seed + epoch)

    def state_dict(self) -> dict:
        return self.dataset.state_dict()  # type: ignore[attr-defined]

    def load_state_dict(self, state_dict: dict) -> None:
        self.dataset.load_state_dict(state_dict)  # type: ignore[attr-defined]


# pylint: disable=protected-access
class NoamLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer: torch.optim.Optimizer, warmup_steps: float = 0.1, last_epoch: int = -1) -> None:
        self.warmup_steps = float(warmup_steps)
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> list[float | torch.Tensor]:
        if is_pytorch_at_least_2_4():
            _warn_get_lr_called_within_step(self)
        return self._get_closed_form_lr()

    def _get_closed_form_lr(self) -> list[float | torch.Tensor]:
        step = self.last_epoch + 1
        scale = self.warmup_steps**0.5 * min(step * self.warmup_steps**-1.5, step**-0.5)
        return [base_lr * scale for base_lr in self.base_lrs]


# pylint: disable=protected-access
class StepwiseGradualLR(torch.optim.lr_scheduler._LRScheduler):
    """Hardcoded step-wise learning rate scheduling.

    Necessary for CapacitronVAE.
    """

    def __init__(self, optimizer: torch.optim.Optimizer, gradual_learning_rates, last_epoch: int = -1) -> None:
        self.step_thresholds, self.learning_rates = zip(*gradual_learning_rates, strict=True)
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> list[float | torch.Tensor]:
        if is_pytorch_at_least_2_4():
            _warn_get_lr_called_within_step(self)

        if self.last_epoch not in self.step_thresholds:
            return [group["lr"] for group in self.optimizer.param_groups]

        index = self.step_thresholds.index(self.last_epoch)
        lr = self.learning_rates[index]
        return [lr for _ in self.optimizer.param_groups]

    def _get_closed_form_lr(self) -> list[float | torch.Tensor]:
        index = self._find_index(self.last_epoch)
        lr = self.learning_rates[index]
        return [lr for _ in self.base_lrs]

    def _find_index(self, step: int) -> int:
        # Locate the most recent threshold <= step
        index = bisect_right(self.step_thresholds, step) - 1
        return max(0, min(index, len(self.learning_rates) - 1))
