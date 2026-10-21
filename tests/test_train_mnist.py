import logging

import pytest
import torch

from tests.utils.mnist import MnistModel, MnistModelConfig
from trainer import Trainer, TrainerArgs

is_cuda = torch.cuda.is_available()


def test_train_mnist(tmp_path):
    model = MnistModel()

    # Parsing command line args
    trainer = Trainer(TrainerArgs(), MnistModelConfig(), output_path=tmp_path, model=model, gpu=0 if is_cuda else None)

    trainer.fit()
    loss1 = trainer.keep_avg_train["avg_loss"]

    trainer.fit()
    loss2 = trainer.keep_avg_train["avg_loss"]

    assert loss1 > loss2

    # Without parsing command line args
    args = TrainerArgs()
    args.small_run = 4

    trainer2 = Trainer(
        args,
        MnistModelConfig(),
        output_path=tmp_path,
        model=model,
        gpu=0 if is_cuda else None,
        parse_command_line_args=False,
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
    )
    trainer3.fit()
    loss4 = trainer3.keep_avg_train["avg_loss"]

    assert loss3 > loss4

    with pytest.raises(ValueError, match="cannot be None"):
        Trainer(args, MnistModelConfig(), output_path=tmp_path, model=None)


def test_failed_run_is_removed(tmp_path, monkeypatch):
    trainer = Trainer(
        TrainerArgs(), MnistModelConfig(), output_path=tmp_path, model=MnistModel(), parse_command_line_args=False
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
