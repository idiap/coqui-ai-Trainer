import os
from typing import TYPE_CHECKING, Any

from trainer._types import Audio, Figure
from trainer.config import TrainerConfig
from trainer.logging.base_dash_logger import BaseDashboardLogger

if TYPE_CHECKING:
    import torch


class DummyLogger(BaseDashboardLogger):
    """DummyLogger that implements the API but does nothing."""

    def model_weights(self, model: "torch.nn.Module", step: int) -> None:
        pass

    def add_scalar(self, title: str, value: float, step: int) -> None:
        pass

    def add_figure(self, title: str, figure: Figure, step: int) -> None:
        pass

    def add_config(self, config: TrainerConfig) -> None:
        pass

    def add_audio(self, title: str, audio: Audio, step: int, sample_rate: int) -> None:
        pass

    def add_text(self, title: str, text: str, step: int) -> None:
        pass

    def add_artifact(
        self, file_or_dir: str | os.PathLike[Any], name: str, artifact_type: str, aliases: list[str] | None = None
    ) -> None:
        pass

    def flush(self) -> None:
        pass

    def finish(self) -> None:
        pass
