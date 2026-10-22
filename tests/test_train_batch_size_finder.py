import torch

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
