import pytest
import torch
from torch.utils.data import DataLoader

from tests.utils.mnist import MnistModel, MnistModelConfig, get_mnist_samples
from trainer import Trainer, TrainerArgs

is_cuda = torch.cuda.is_available()


def test_train_largest_batch_mnist(tmp_path):
    model = MnistModel()
    trainer = Trainer(
        TrainerArgs(),
        MnistModelConfig(),
        output_path=tmp_path,
        model=model,
        train_samples=get_mnist_samples(is_eval=False),
        eval_samples=get_mnist_samples(is_eval=True),
        gpu=0 if is_cuda else None,
    )

    trainer.fit_with_largest_batch_size(starting_batch_size=2048)
    loss1 = trainer.keep_avg_train["avg_loss"]

    trainer.fit_with_largest_batch_size(starting_batch_size=2048)
    loss2 = trainer.keep_avg_train["avg_loss"]

    assert loss1 > loss2


def test_batch_size_is_reduced_on_oom(tmp_path):
    class OomModel(MnistModel):
        def train_step(self, batch, criterion, optimizer_idx=None):
            if len(batch["input"]) > 32:
                msg = "CUDA out of memory."
                raise RuntimeError(msg)
            return super().train_step(batch, criterion, optimizer_idx)

    trainer = Trainer(
        TrainerArgs(),
        MnistModelConfig(),
        output_path=tmp_path,
        model=OomModel(),
        train_samples=get_mnist_samples(is_eval=False),
        eval_samples=get_mnist_samples(is_eval=True),
        parse_command_line_args=False,
    )

    trainer.fit_with_largest_batch_size(starting_batch_size=256)

    assert trainer.config.batch_size == 32
    assert trainer.train_loader is not None
    assert trainer.train_loader.batch_size == 32
    assert trainer.eval_loader is not None
    assert trainer.eval_loader.batch_size == 32


def test_batch_size_finder_requires_train_samples(tmp_path):
    trainer = Trainer(
        TrainerArgs(),
        MnistModelConfig(),
        output_path=tmp_path,
        model=MnistModel(),
        train_loader=DataLoader(get_mnist_samples(is_eval=False)),  # ty: ignore[invalid-argument-type]
        eval_samples=get_mnist_samples(is_eval=True),
        parse_command_line_args=False,
    )

    with pytest.raises(ValueError, match="needs `train_samples`"):
        trainer.fit_with_largest_batch_size()
