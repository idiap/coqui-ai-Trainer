import time
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST

from trainer import Trainer, TrainerArgs, TrainerConfig, TrainerModel
from trainer.generic_utils import KeepAverage


@dataclass
class MnistModelConfig(TrainerConfig):
    optimizer: str = "Adam"
    lr: float = 0.001
    epochs: int = 1
    print_step: int = 1
    save_step: int = 5
    plot_step: int = 5
    dashboard_logger: str = "tensorboard"


class MnistModel(TrainerModel):
    def __init__(self) -> None:
        super().__init__()

        # mnist images are (1, 28, 28) (channels, height, width)
        self.layer_1 = nn.Linear(28 * 28, 128)
        self.layer_2 = nn.Linear(128, 256)
        self.layer_3 = nn.Linear(256, 10)

    def forward(self, x):
        batch_size, _, _, _ = x.size()

        # (b, 1, 28, 28) -> (b, 1*28*28)
        x = x.view(batch_size, -1)
        x = self.layer_1(x)
        x = F.relu(x)
        x = self.layer_2(x)
        x = F.relu(x)
        x = self.layer_3(x)

        return F.log_softmax(x, dim=1)

    def train_step(self, batch, criterion):
        x, y = batch
        logits = self(x)
        loss = criterion(logits, y)
        return {"model_outputs": logits}, {"loss": loss}

    def eval_step(self, batch, criterion):
        x, y = batch
        logits = self(x)
        loss = criterion(logits, y)
        return {"model_outputs": logits}, {"loss": loss}

    def get_criterion(self):
        return torch.nn.NLLLoss()

    def get_data_loader(self, config, *, is_eval, samples=None, verbose=False):
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        dataset = MNIST(Path.cwd(), train=not is_eval, download=True, transform=transform)
        dataset.data = dataset.data[:256]
        dataset.targets = dataset.targets[:256]
        return DataLoader(dataset, batch_size=config.batch_size)


def create_trainer(config, model, output_path, gpu, continue_path=None, restore_path=None):
    args = TrainerArgs(continue_path=continue_path, restore_path=restore_path)
    trainer = Trainer(args, config, output_path=output_path, model=model, gpu=gpu)
    trainer.train_loader = trainer.get_train_dataloader(trainer.train_samples)
    trainer.keep_avg_train = KeepAverage()
    return trainer


def run_steps(trainer, start_step, end_step):
    """Step through training manually."""
    for step in range(start_step, end_step):
        batch = next(iter(trainer.train_loader))
        trainer.train_step(batch, len(trainer.train_loader), step, time.time())
