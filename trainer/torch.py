from collections.abc import Iterator

import torch
from torch.utils.data.distributed import DistributedSampler


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

    def get_lr(self) -> list[float]:
        step = max(self.last_epoch, 1)
        return [
            base_lr * self.warmup_steps**0.5 * min(step * self.warmup_steps**-1.5, step**-0.5)
            for base_lr in self.base_lrs
        ]


# pylint: disable=protected-access
class StepwiseGradualLR(torch.optim.lr_scheduler._LRScheduler):
    """Hardcoded step-wise learning rate scheduling.

    Necessary for CapacitronVAE.
    """

    def __init__(self, optimizer: torch.optim.Optimizer, gradual_learning_rates, last_epoch: int = -1) -> None:
        self.gradual_learning_rates = gradual_learning_rates
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        step = max(self.last_epoch, 1)
        step_thresholds = [x[0] for x in self.gradual_learning_rates]
        rates = [x[1] for x in self.gradual_learning_rates]

        # Ignore steps larger than the last step in the list
        valid_indices = [i for i, threshold in enumerate(step_thresholds) if threshold <= step]
        last_true_idx = valid_indices[-1] if valid_indices else 0
        lr = rates[last_true_idx]

        # Return last lr if step is above the set threshold
        lr = rates[-1] if step > step_thresholds[-1] else lr
        # Return first lr if step is below the second threshold - first is initial lr
        lr = rates[0] if step < step_thresholds[1] else lr

        # Return learning rate list of the same size as base_lrs
        return [lr] * len(self.base_lrs)
