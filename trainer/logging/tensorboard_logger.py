import os
from typing import Any

import torch
from torch.utils.tensorboard import SummaryWriter

from trainer._types import Audio, Figure
from trainer.logging.base_dash_logger import BaseDashboardLogger


class TensorboardLogger(BaseDashboardLogger):
    def __init__(self, log_dir: str | os.PathLike[Any], model_name: str) -> None:
        self.model_name = model_name
        self.writer = SummaryWriter(log_dir)

    def model_weights(self, model: torch.nn.Module, step: int) -> None:
        layer_num = 1
        for name, param in model.named_parameters():
            if param.numel() == 1:
                self.writer.add_scalar(f"layer{layer_num}-{name}/value", param.max(), step)
            else:
                self.writer.add_scalar(f"layer{layer_num}-{name}/max", param.max(), step)
                self.writer.add_scalar(f"layer{layer_num}-{name}/min", param.min(), step)
                self.writer.add_scalar(f"layer{layer_num}-{name}/mean", param.mean(), step)
                self.writer.add_scalar(f"layer{layer_num}-{name}/std", param.std(), step)
                self.writer.add_histogram(f"layer{layer_num}-{name}/param", param, step)
                if param.grad is not None:
                    self.writer.add_histogram(f"layer{layer_num}-{name}/grad", param.grad, step)
            layer_num += 1

    def add_scalar(self, title: str, value: float, step: int) -> None:
        self.writer.add_scalar(title, value, step)

    def add_audio(self, title: str, audio: Audio, step: int, sample_rate: int) -> None:
        if audio.dtype == "float16":
            audio = audio.astype("float32")
        self.writer.add_audio(title, audio, step, sample_rate=sample_rate)

    def add_text(self, title: str, text: str, step: int) -> None:
        self.writer.add_text(title, text, step)

    def add_figure(self, title: str, figure: Figure, step: int) -> None:
        self.writer.add_figure(title, figure, step)

    def add_artifact(
        self, file_or_dir: str | os.PathLike[Any], name: str, artifact_type: str, aliases: list[str] | None = None
    ) -> None:
        pass

    def flush(self) -> None:
        self.writer.flush()

    def finish(self) -> None:
        self.writer.close()
