import json
import logging
from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader

from tests.utils.mnist import MnistModel, MnistModelConfig, get_mnist_samples
from trainer import Trainer, TrainerArgs

is_cuda = torch.cuda.is_available()


def test_train_mnist(tmp_path):
    model = MnistModel()
    train_samples = get_mnist_samples(is_eval=False)
    eval_samples = get_mnist_samples(is_eval=True)

    # Parsing command line args
    trainer = Trainer(
        TrainerArgs(),
        MnistModelConfig(),
        output_path=tmp_path,
        model=model,
        gpu=0 if is_cuda else None,
        train_samples=train_samples,
        eval_samples=eval_samples,
    )

    trainer.fit()
    loss1 = trainer.keep_avg_train["avg_loss"]

    trainer.fit()
    loss2 = trainer.keep_avg_train["avg_loss"]

    assert loss1 > loss2

    # Without parsing command line args
    args = TrainerArgs()
    args.small_run = 192

    trainer2 = Trainer(
        args,
        MnistModelConfig(),
        output_path=tmp_path,
        model=model,
        gpu=0 if is_cuda else None,
        parse_command_line_args=False,
        train_samples=train_samples,
        eval_samples=eval_samples,
    )
    trainer2.fit()
    loss3 = trainer2.keep_avg_train["avg_loss"]

    args.continue_path = str(max(tmp_path.iterdir(), key=lambda p: p.stat().st_mtime))

    trainer3 = Trainer(
        args,
        MnistModelConfig(),
        output_path=tmp_path,
        model=model,
        gpu=0 if is_cuda else None,
        parse_command_line_args=False,
        train_samples=train_samples,
        eval_samples=eval_samples,
    )
    trainer3.fit()
    loss4 = trainer3.keep_avg_train["avg_loss"]

    assert loss3 > loss4

    with pytest.raises(ValueError, match="cannot be None"):
        Trainer(
            args,
            MnistModelConfig(),
            output_path=tmp_path,
            model=None,
            train_samples=train_samples,
            eval_samples=eval_samples,
        )


def test_failed_run_is_removed(tmp_path, monkeypatch):
    trainer = Trainer(
        TrainerArgs(),
        MnistModelConfig(),
        output_path=tmp_path,
        model=MnistModel(),
        train_samples=get_mnist_samples(is_eval=False),
        eval_samples=get_mnist_samples(is_eval=True),
        parse_command_line_args=False,
    )
    handler = trainer._log_handler
    assert (trainer.output_path / "trainer_0_log.txt").is_file()

    def fail():
        msg = "boom"
        raise RuntimeError(msg)

    monkeypatch.setattr(trainer, "_fit", fail)
    with pytest.raises(SystemExit):
        trainer.fit()

    # the log file must be closed, otherwise the folder can't be removed on Windows
    assert handler.stream is None
    assert handler not in logging.getLogger("trainer").handlers
    assert not trainer.output_path.exists()


def test_train_samples_or_loader_required_for_training(tmp_path):
    trainer = Trainer(
        TrainerArgs(),
        MnistModelConfig(),
        output_path=tmp_path,
        model=MnistModel(),
        eval_samples=get_mnist_samples(is_eval=True),
        parse_command_line_args=False,
    )
    with pytest.raises(ValueError, match="`train_samples` or `train_loader`"):
        trainer._fit()


def test_eval_only_without_train_samples(tmp_path):
    trainer = Trainer(
        TrainerArgs(),
        MnistModelConfig(),
        output_path=tmp_path,
        model=MnistModel(),
        eval_samples=get_mnist_samples(is_eval=True),
        parse_command_line_args=False,
    )
    trainer.eval_epoch()

    assert trainer.keep_avg_eval is not None
    assert "avg_loss" in trainer.keep_avg_eval.avg_values


def test_eval_samples_required_when_run_eval_enabled(tmp_path):
    config = MnistModelConfig()
    config.run_eval = True
    model = MnistModel()

    # Passing train_samples with eval_samples=None must raise ValueError
    with pytest.raises(ValueError, match="Evaluation is enabled"):
        Trainer(
            TrainerArgs(),
            config,
            output_path=tmp_path,
            model=model,
            train_samples=["dummy_train"],
            eval_samples=None,
            parse_command_line_args=False,
        )

    # Passing train_samples with empty eval_samples must raise ValueError
    with pytest.raises(ValueError, match="Evaluation is enabled"):
        Trainer(
            TrainerArgs(),
            config,
            output_path=tmp_path,
            model=model,
            train_samples=["dummy_train"],
            eval_samples=[],
            parse_command_line_args=False,
        )

    # no output folder should be left behind
    assert not any(tmp_path.iterdir())

    # If run_eval is False, eval_samples=None is allowed
    config_no_eval = MnistModelConfig()
    config_no_eval.run_eval = False
    trainer = Trainer(
        TrainerArgs(),
        config_no_eval,
        output_path=tmp_path,
        model=model,
        train_samples=["dummy_train"],
        eval_samples=None,
        parse_command_line_args=False,
    )
    assert trainer.eval_samples is None


def test_empty_dataloader_raises_valueerror(tmp_path):
    class EmptyLoaderModel(MnistModel):
        def get_data_loader(self, config, *, is_eval=False, samples, verbose=False):
            return DataLoader([])  # type: ignore[arg-type]

    config = MnistModelConfig()
    config.run_eval = False
    trainer = Trainer(
        TrainerArgs(),
        config,
        output_path=tmp_path,
        model=EmptyLoaderModel(),
        train_samples=["dummy_train"],
        parse_command_line_args=False,
    )
    with pytest.raises(ValueError, match="Training DataLoader is empty"):
        trainer._fit()


def test_custom_eval_loader_allowed_without_eval_samples(tmp_path):
    config = MnistModelConfig()
    config.run_eval = True
    model = MnistModel()
    dummy_loader = DataLoader([{"input": torch.zeros(1, 28, 28), "target": torch.zeros(1, dtype=torch.long)}])  # type: ignore[arg-type]

    # Providing custom eval_loader should succeed even when eval_samples=None
    trainer = Trainer(
        TrainerArgs(),
        config,
        output_path=tmp_path,
        model=model,
        train_samples=["dummy_train"],
        eval_samples=None,
        eval_loader=dummy_loader,
        parse_command_line_args=False,
    )
    assert trainer.eval_loader is dummy_loader


def test_train_loader_without_eval_loader_raises_when_run_eval(tmp_path):
    config = MnistModelConfig()
    config.run_eval = True
    model = MnistModel()
    dummy_loader = DataLoader([{"input": torch.zeros(1, 28, 28), "target": torch.zeros(1, dtype=torch.long)}])  # type: ignore[arg-type]

    # Providing train_loader but omitting eval_loader and eval_samples must raise
    with pytest.raises(ValueError, match="Evaluation is enabled"):
        Trainer(
            TrainerArgs(),
            config,
            output_path=tmp_path,
            model=model,
            train_loader=dummy_loader,
            eval_loader=None,
            eval_samples=None,
            parse_command_line_args=False,
        )


def test_empty_eval_loader_raises_before_training_epochs(tmp_path):
    class EmptyEvalModel(MnistModel):
        def get_data_loader(self, config, *, is_eval=False, samples, verbose=False):
            if is_eval:
                return DataLoader([])  # type: ignore[arg-type]
            return super().get_data_loader(config, is_eval=is_eval, samples=samples, verbose=verbose)

    config = MnistModelConfig()
    config.run_eval = True
    config.epochs = 5
    model = EmptyEvalModel()

    trainer = Trainer(
        TrainerArgs(),
        config,
        output_path=tmp_path,
        model=model,
        train_samples=get_mnist_samples(is_eval=False),
        eval_samples=["dummy_eval"],
        parse_command_line_args=False,
    )

    # Must raise before any training epoch runs
    with pytest.raises(ValueError, match="Evaluation DataLoader is empty"):
        trainer._fit()

    assert trainer.total_steps_done == 0


def test_loaders_see_on_epoch_start_changes(tmp_path):
    def set_batch_size(trainer):
        trainer.config.batch_size = 8

    trainer = Trainer(
        TrainerArgs(),
        MnistModelConfig(),
        output_path=tmp_path,
        model=MnistModel(),
        train_samples=get_mnist_samples(is_eval=False),
        eval_samples=get_mnist_samples(is_eval=True),
        parse_command_line_args=False,
        callbacks={"on_epoch_start": set_batch_size},
    )
    trainer._fit()

    assert trainer.train_loader is not None
    assert trainer.train_loader.batch_size == 8
    assert trainer.eval_loader is not None
    assert trainer.eval_loader.batch_size == 8


def test_log_model_step_at_least_save_step(tmp_path):
    config = MnistModelConfig()
    config.save_step = 2
    config.log_model_step = 1
    trainer = Trainer(
        TrainerArgs(),
        config,
        output_path=tmp_path,
        model=MnistModel(),
        train_samples=get_mnist_samples(is_eval=False),
        eval_samples=get_mnist_samples(is_eval=True),
        parse_command_line_args=False,
    )
    assert trainer.log_model_step == 2
    saved_config = json.loads((trainer.output_path / "config.json").read_text())
    assert saved_config["log_model_step"] == 2


def test_checkpoint_artifact_only_on_log_model_step(tmp_path, monkeypatch):
    """Ensure checkpoint artifact is only logged at log_model_step and targets the specific .pth file."""
    config = MnistModelConfig()
    config.epochs = 1
    config.save_step = 1
    config.log_model_step = 2
    config.save_checkpoints = True
    config.batch_size = 8

    trainer = Trainer(
        TrainerArgs(),
        config,
        output_path=tmp_path,
        model=MnistModel(),
        train_samples=get_mnist_samples(is_eval=False, num_samples=5 * config.batch_size),
        eval_samples=get_mnist_samples(is_eval=True, num_samples=config.batch_size),
        gpu=None,
        parse_command_line_args=False,
    )

    artifact_calls = []

    def mock_add_artifact(file_or_dir, name, artifact_type, aliases=None):
        artifact_calls.append(
            {
                "file_or_dir": str(file_or_dir),
                "name": name,
                "type": artifact_type,
                "step": trainer.total_steps_done,
            }
        )

    monkeypatch.setattr(trainer.dashboard_logger, "add_artifact", mock_add_artifact)

    trainer.fit()

    # Checkpoints are saved at every step, but only uploaded at every log_model_step (even steps)
    checkpoint_artifacts = [call for call in artifact_calls if call["name"] == "checkpoint"]
    expected_steps = list(range(2, trainer.total_steps_done, 2))
    assert len(expected_steps) > 1
    assert [call["step"] for call in checkpoint_artifacts] == expected_steps
    assert [Path(call["file_or_dir"]).name for call in checkpoint_artifacts] == [
        f"checkpoint_{step}.pth" for step in expected_steps
    ]
