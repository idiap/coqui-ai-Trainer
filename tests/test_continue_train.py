import sys

import pytest
import torch

from tests.utils.mnist import MnistModel, MnistModelConfig, create_trainer, run_steps
from tests.utils.train_mnist import main as train_mnist


def test_continue_train(tmp_path, monkeypatch):
    with monkeypatch.context() as m:
        m.setattr(sys, "argv", [sys.argv[0], "--coqpit.output_path", str(tmp_path)])
        train_mnist()

    continue_path = max(tmp_path.iterdir(), key=lambda p: p.stat().st_mtime)
    number_of_checkpoints = len(list(continue_path.glob("*.pth")))

    # Continue training from the best model
    with monkeypatch.context() as m:
        m.setattr(sys, "argv", [sys.argv[0], "--continue_path", str(continue_path), "--coqpit.run_eval_steps", "1"])
        train_mnist()

    assert number_of_checkpoints < len(list(continue_path.glob("*.pth")))

    # Continue training from the last checkpoint
    for best in continue_path.glob("best_model*"):
        best.unlink()

    # Continue training from a specific checkpoint
    restore_path = continue_path / "checkpoint_5.pth"
    with monkeypatch.context() as m:
        m.setattr(
            sys, "argv", [sys.argv[0], "--restore_path", str(restore_path), "--coqpit.output_path", str(tmp_path)]
        )
        train_mnist()


def test_lr_continue_vs_restore_stepwise(tmp_path):
    gpu = 0 if torch.cuda.is_available() else None
    LR_1 = 1e-3
    LR_2 = 2e-4
    LR_3 = 3e-5
    train_steps = 3
    config = MnistModelConfig(
        lr_scheduler="StepwiseGradualLR",
        lr_scheduler_params={"gradual_learning_rates": [(0, LR_1), (2, LR_2), (4, LR_3)]},
        scheduler_after_epoch=False,
        save_step=train_steps - 1,
    )

    # 1. Train for a few steps and save checkpoint
    trainer = create_trainer(config, MnistModel(), tmp_path, gpu)

    run_steps(trainer, 0, train_steps)
    assert trainer.scheduler[0].get_last_lr() == [LR_2]

    # 2. Continue from checkpoint (should preserve LR state)
    continue_path = max(tmp_path.iterdir(), key=lambda p: p.stat().st_mtime)
    trainer_continue = create_trainer(
        config, MnistModel(), tmp_path / "continue", gpu, continue_path=str(continue_path)
    )

    assert trainer_continue.scheduler[0].get_last_lr() == [LR_2]

    last_epoch = trainer_continue.scheduler[0].last_epoch
    assert last_epoch == train_steps
    run_steps(trainer_continue, last_epoch, last_epoch + 1)
    assert trainer_continue.scheduler[0].get_last_lr() == [LR_3]

    # 3. Restore from checkpoint (should reset LR)
    checkpoint_path = continue_path / f"checkpoint_{train_steps - 1}.pth"
    restored_path = tmp_path / "restored"
    trainer_restored = create_trainer(config, MnistModel(), restored_path, gpu, restore_path=str(checkpoint_path))

    assert trainer_restored.scheduler[0].last_epoch == 0
    assert trainer_restored.scheduler[0].get_last_lr() == [LR_1]


def test_lr_continue_vs_restore_multistep(tmp_path):
    gpu = 0 if torch.cuda.is_available() else None
    LR_1 = 1e-3
    LR_2 = 1e-4
    LR_3 = 1e-5
    train_steps = 2
    config = MnistModelConfig(
        lr_scheduler="MultiStepLR",
        lr_scheduler_params={"milestones": [2, 4]},
        scheduler_after_epoch=False,
        save_step=train_steps - 1,
    )

    # 1. Train for a few steps and save checkpoint
    trainer = create_trainer(config, MnistModel(), tmp_path, gpu)

    run_steps(trainer, 0, train_steps)
    assert trainer.scheduler[0].get_last_lr() == [LR_2]

    # 2. Continue from checkpoint (should preserve LR state)
    continue_path = max(tmp_path.iterdir(), key=lambda p: p.stat().st_mtime)
    trainer_continue = create_trainer(
        config, MnistModel(), tmp_path / "continue", gpu, continue_path=str(continue_path)
    )

    assert trainer_continue.scheduler[0].get_last_lr() == [LR_2]

    last_epoch = trainer_continue.scheduler[0].last_epoch
    assert last_epoch == train_steps
    run_steps(trainer_continue, last_epoch, last_epoch + 2)
    assert trainer_continue.scheduler[0].get_last_lr() == pytest.approx([LR_3])

    # 3. Restore from checkpoint (should reset LR)
    checkpoint_path = continue_path / f"checkpoint_{train_steps - 1}.pth"
    restored_path = tmp_path / "restored"
    trainer_restored = create_trainer(config, MnistModel(), restored_path, gpu, restore_path=str(checkpoint_path))

    assert trainer_restored.scheduler[0].last_epoch == 0
    assert trainer_restored.scheduler[0].get_last_lr() == [LR_1]


def test_lr_continue_vs_restore_noam(tmp_path):
    gpu = 0 if torch.cuda.is_available() else None
    LR_0 = 0.0003333333333333333
    LR_1 = 0.001
    LR_2 = 0.0008660254037844386
    LR_3 = 0.0007745966692414833
    warmup_steps = 3
    config = MnistModelConfig(
        lr_scheduler="NoamLR",
        lr_scheduler_params={"warmup_steps": warmup_steps},
        scheduler_after_epoch=False,
        save_step=warmup_steps - 1,
    )

    # 1. Train for a few steps and save checkpoint
    trainer = create_trainer(config, MnistModel(), tmp_path, gpu)

    run_steps(trainer, 0, warmup_steps - 1)
    assert trainer.scheduler[0].get_last_lr() == pytest.approx([LR_1])

    run_steps(trainer, warmup_steps - 1, warmup_steps)
    assert trainer.scheduler[0].get_last_lr() == pytest.approx([LR_2])

    # 2. Continue from checkpoint (should preserve LR state)
    continue_path = max(tmp_path.iterdir(), key=lambda p: p.stat().st_mtime)
    trainer_continue = create_trainer(
        config, MnistModel(), tmp_path / "continue", gpu, continue_path=str(continue_path)
    )

    last_epoch = trainer_continue.scheduler[0].last_epoch
    assert last_epoch == warmup_steps
    assert trainer_continue.scheduler[0].get_last_lr() == [LR_2]
    run_steps(trainer_continue, last_epoch, last_epoch + 1)
    assert trainer_continue.scheduler[0].get_last_lr() == [LR_3]

    # 3. Restore from checkpoint (should reset LR)
    checkpoint_path = continue_path / f"checkpoint_{warmup_steps - 1}.pth"
    restored_path = tmp_path / "restored"
    trainer_restored = create_trainer(config, MnistModel(), restored_path, gpu, restore_path=str(checkpoint_path))

    assert trainer_restored.scheduler[0].last_epoch == 0
    assert trainer_restored.scheduler[0].get_last_lr() == pytest.approx([LR_0])
