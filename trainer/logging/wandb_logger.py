# pylint: disable=W0613

import os
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch

from trainer._types import Audio, Figure
from trainer.logging.base_dash_logger import BaseDashboardLogger

try:
    import wandb
except ImportError as e:
    msg = "To use the W&B logger you need to install `wandb`"
    raise ImportError(msg) from e


class WandbLogger(BaseDashboardLogger):
    def __init__(self, **kwargs: Any) -> None:
        self.run = None
        self.run = wandb.init(**kwargs) if not wandb.run else wandb.run
        # dictionary of dictionaries - record stats per step
        self.log_dict: dict[int, dict[str, Any]] = defaultdict(dict)

    def model_weights(self, model: torch.nn.Module, step: int) -> None:
        layer_num = 1
        for name, param in model.named_parameters():
            if param.numel() == 1:
                self.add_scalars("weights", {f"layer{layer_num}-{name}/value": param.max().item()}, step)
            else:
                self.add_scalars("weights", {f"layer{layer_num}-{name}/max": param.max().item()}, step)
                self.add_scalars("weights", {f"layer{layer_num}-{name}/min": param.min().item()}, step)
                self.add_scalars("weights", {f"layer{layer_num}-{name}/mean": param.mean().item()}, step)
                self.add_scalars("weights", {f"layer{layer_num}-{name}/std": param.std().item()}, step)
                self.log_dict[step][f"weights/layer{layer_num}-{name}/param"] = wandb.Histogram(param.detach().tolist())
                if param.grad is not None:
                    self.log_dict[step][f"weights/layer{layer_num}-{name}/grad"] = wandb.Histogram(param.grad.tolist())
            layer_num += 1

    def add_text(self, title: str, text: str, step: int) -> None:
        pass

    def add_scalar(self, title: str, value: float, step: int) -> None:
        self.log_dict[step][title] = value

    def add_figure(self, title: str, figure: Figure, step: int) -> None:
        self.log_dict[step][title] = wandb.Image(figure)

    def add_audio(self, title: str, audio: Audio, step: int, sample_rate: int) -> None:
        if audio.dtype == "float16":
            audio = audio.astype("float32")
        self.log_dict[step][title] = wandb.Audio(audio, sample_rate=sample_rate)

    def flush(self) -> None:
        if self.run:
            for step in sorted(self.log_dict.keys()):
                self.run.log(self.log_dict[step], step)
        self.log_dict.clear()

    def finish(self) -> None:
        if self.run:
            self.run.finish()

    def add_artifact(
        self, file_or_dir: str | os.PathLike[Any], name: str, artifact_type: str, aliases: list[str] | None = None
    ) -> None:
        if not self.run:
            return
        name = f"{self.run.id}_{name}"
        artifact = wandb.Artifact(name, type=artifact_type)
        data_path = Path(file_or_dir)
        if data_path.is_dir():
            artifact.add_dir(str(data_path))
        elif data_path.is_file():
            artifact.add_file(str(data_path))

        self.run.log_artifact(artifact, aliases=aliases)
