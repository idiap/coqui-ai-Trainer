import pytest
import torch

from tests.utils.mnist import MnistModel, MnistModelConfig
from trainer import Trainer, TrainerArgs


def _create_trainer(tmp_path, model=None, **config_kwargs):
    return Trainer(
        TrainerArgs(),
        MnistModelConfig(run_eval=False, **config_kwargs),
        output_path=tmp_path,
        model=model or MnistModel(),
        parse_command_line_args=False,
    )


class MultiOptimizerModel(MnistModel):
    """Defines its own optimizers and a single scheduler for one of them."""

    def __init__(self, num_optimizers=2, scheduled_idx=1, num_criteria=1, num_schedulers=1) -> None:
        super().__init__()
        self.num_optimizers = num_optimizers
        self.scheduled_idx = scheduled_idx
        self.num_criteria = num_criteria
        self.num_schedulers = num_schedulers

    def get_optimizer(self):
        return [torch.optim.SGD(self.parameters(), lr=0.1) for _ in range(self.num_optimizers)]

    def get_scheduler(self, optimizer):
        if self.num_schedulers > 1:
            return [None] * self.num_schedulers
        if self.scheduled_idx is None:
            target = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=0.1)
        else:
            target = optimizer[self.scheduled_idx]
        return torch.optim.lr_scheduler.StepLR(target, step_size=1, gamma=0.5)

    def get_criterion(self):
        criteria = [torch.nn.NLLLoss() for _ in range(self.num_criteria)]
        return criteria if self.num_criteria > 1 else criteria[0]


def test_multiple_optimizers_with_shared_params(tmp_path):
    trainer = _create_trainer(tmp_path, optimizer=["Adam", "SGD"], lr=0.01)

    assert isinstance(trainer.optimizer[0], torch.optim.Adam)
    assert isinstance(trainer.optimizer[1], torch.optim.SGD)
    for optimizer in trainer.optimizer:
        assert optimizer.param_groups[0]["lr"] == pytest.approx(0.01)


def test_multiple_optimizers_with_per_optimizer_params(tmp_path):
    trainer = _create_trainer(
        tmp_path,
        optimizer=["Adam", "SGD"],
        lr=[0.01, 0.02],
        optimizer_params=[{"betas": (0.8, 0.9)}, {"momentum": 0.5}],
    )

    adam, sgd = trainer.optimizer
    assert adam.param_groups[0]["lr"] == pytest.approx(0.01)
    assert adam.param_groups[0]["betas"] == (0.8, 0.9)
    assert sgd.param_groups[0]["lr"] == pytest.approx(0.02)
    assert sgd.param_groups[0]["momentum"] == pytest.approx(0.5)


@pytest.mark.parametrize(
    "config_kwargs",
    [
        {"lr": [0.01, 0.02]},
        {"optimizer_params": [{"betas": (0.8, 0.9)}]},
    ],
)
def test_single_optimizer_rejects_lists(tmp_path, config_kwargs):
    with pytest.raises(TypeError, match="can only be lists"):
        _create_trainer(tmp_path, optimizer="Adam", **config_kwargs)


def test_multiple_schedulers(tmp_path):
    trainer = _create_trainer(tmp_path, optimizer=["Adam", "SGD"], lr_scheduler=["ConstantLR", "LinearLR"])

    assert isinstance(trainer.scheduler[0], torch.optim.lr_scheduler.ConstantLR)
    assert isinstance(trainer.scheduler[1], torch.optim.lr_scheduler.LinearLR)
    for scheduler, optimizer in zip(trainer.scheduler, trainer.optimizer, strict=True):
        assert scheduler is not None
        assert scheduler.optimizer is optimizer


def test_scheduler_list_length_mismatch(tmp_path):
    with pytest.raises(ValueError, match="zip"):
        _create_trainer(tmp_path, optimizer=["Adam", "SGD"], lr_scheduler=["ConstantLR"])


def test_single_scheduler_name_schedules_each_optimizer(tmp_path):
    trainer = _create_trainer(
        tmp_path,
        optimizer=["SGD", "SGD"],
        lr=[0.1, 0.2],
        lr_scheduler="StepLR",
        lr_scheduler_params={"step_size": 1, "gamma": 0.5},
    )

    for scheduler, optimizer in zip(trainer.scheduler, trainer.optimizer, strict=True):
        assert scheduler is not None
        assert scheduler.optimizer is optimizer
        optimizer.step()
        scheduler.step()
    assert [o.param_groups[0]["lr"] for o in trainer.optimizer] == pytest.approx([0.05, 0.1])


def test_single_model_scheduler_applies_to_its_optimizer(tmp_path):
    trainer = _create_trainer(tmp_path, model=MultiOptimizerModel(scheduled_idx=1))

    assert trainer.scheduler[0] is None
    scheduler = trainer.scheduler[1]
    assert scheduler is not None
    assert scheduler.optimizer is trainer.optimizer[1]


def test_single_model_scheduler_without_optimizer(tmp_path):
    with pytest.raises(RuntimeError, match="not bound to any of the optimizers"):
        _create_trainer(tmp_path, model=MultiOptimizerModel(scheduled_idx=None))


def test_mismatched_optimizer_scheduler_criterion_lengths(tmp_path):
    # 3 optimizers, but only 2 schedulers and 2 criteria
    model = MultiOptimizerModel(num_optimizers=3, num_criteria=2, num_schedulers=2)
    with pytest.raises(RuntimeError, match="must return the same number of items"):
        _create_trainer(tmp_path, model=model)
