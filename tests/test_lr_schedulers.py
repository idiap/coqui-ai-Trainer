import warnings

import pytest
import torch
from torch.optim.lr_scheduler import EPOCH_DEPRECATION_WARNING

from tests.utils.mnist import MnistModel, MnistModelConfig, create_trainer, run_steps
from trainer.torch import StepwiseGradualLR

is_cuda = torch.cuda.is_available()


@pytest.fixture
def dummy_optimizer():
    model = torch.nn.Linear(2, 1)
    return torch.optim.SGD(model.parameters(), lr=0.1)


@pytest.fixture
def lr_schedule():
    return [
        (0, 0.01),  # initial learning rate
        (5, 0.02),  # increase at epoch 5
        (10, 0.05),  # increase at epoch 10
        (20, 0.1),  # final level
    ]


@pytest.fixture
def lr_schedule_targets():
    return {
        4: 0.01,  # Before second threshold
        5: 0.02,  # Exact match with second threshold
        7: 0.02,  # Between 5 and 10
        10: 0.05,  # Exact match with third threshold
        11: 0.05,  # Between 10 and 20
        25: 0.1,  # After last threshold
    }


def _test_get_last_lr(scheduler, targets, epochs):
    """Test chainable form. Adapted from the scheduler tests in Pytorch."""
    for epoch in range(epochs):
        lr = scheduler.get_last_lr()
        scheduler.optimizer.step()
        scheduler.step()
        if epoch in targets:
            assert lr == pytest.approx([targets[epoch]])


def _test_with_epoch(scheduler, targets, epochs):
    """Test closed form. Adapted from the scheduler tests in Pytorch."""
    for epoch in range(epochs):
        with warnings.catch_warnings(record=True) as w:
            scheduler.optimizer.step()
            scheduler.step(epoch)
            assert len(w) == 1
            assert len(w[0].message.args) == 1
            assert w[0].message.args[0] == EPOCH_DEPRECATION_WARNING
        if epoch in targets:
            assert scheduler.optimizer.param_groups[0]["lr"] == pytest.approx(targets[epoch])


def test_stepwise_lr(dummy_optimizer, lr_schedule, lr_schedule_targets):
    scheduler = StepwiseGradualLR(dummy_optimizer, lr_schedule)
    _test_get_last_lr(scheduler, lr_schedule_targets, epochs=25)


def test_stepwise_lr_with_epoch(dummy_optimizer, lr_schedule, lr_schedule_targets):
    scheduler = StepwiseGradualLR(dummy_optimizer, lr_schedule)
    _test_with_epoch(scheduler, lr_schedule_targets, epochs=25)


def test_train_mnist(tmp_path):
    LR_1 = 1e-3
    LR_2 = 1e-4

    config = MnistModelConfig(
        lr_scheduler="StepwiseGradualLR",
        lr_scheduler_params={"gradual_learning_rates": [(0, LR_1), (2, LR_2)]},
        scheduler_after_epoch=False,
    )
    trainer = create_trainer(config, MnistModel(), tmp_path, gpu=0 if is_cuda else None)

    assert trainer.scheduler[0].get_last_lr() == [LR_1]

    run_steps(trainer, 0, 1)
    assert trainer.scheduler[0].get_last_lr() == [LR_1]

    run_steps(trainer, 1, 2)
    assert trainer.scheduler[0].get_last_lr() == [LR_2]
