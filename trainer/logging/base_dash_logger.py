import os
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

from trainer._types import Audio, Figure
from trainer.config import TrainerConfig
from trainer.io import save_fsspec
from trainer.utils.distributed import rank_zero_only

if TYPE_CHECKING:
    import torch


# pylint: disable=too-many-public-methods
class BaseDashboardLogger(ABC):
    @abstractmethod
    def model_weights(self, model: "torch.nn.Module", step: int) -> None:
        pass

    @abstractmethod
    def add_scalar(self, title: str, value: float, step: int) -> None:
        pass

    @abstractmethod
    def add_figure(self, title: str, figure: Figure, step: int) -> None:
        pass

    def add_config(self, config: TrainerConfig) -> None:
        self.add_text("model-config", f"<pre>{config.to_json()}</pre>", 0)

    @abstractmethod
    def add_audio(self, title: str, audio: Audio, step: int, sample_rate: int) -> None:
        pass

    @abstractmethod
    def add_text(self, title: str, text: str, step: int) -> None:
        pass

    @abstractmethod
    def add_artifact(
        self, file_or_dir: str | os.PathLike[Any], name: str, artifact_type: str, aliases: list[str] | None = None
    ) -> None:
        pass

    def add_scalars(self, scope_name: str, scalars: dict[str, float], step: int) -> None:
        for key, value in scalars.items():
            self.add_scalar(f"{scope_name}/{key}", value, step)

    def add_figures(self, scope_name: str, figures: dict[str, Figure], step: int) -> None:
        for key, figure in figures.items():
            self.add_figure(f"{scope_name}/{key}", figure, step)

    def add_audios(self, scope_name: str, audios: dict[str, Audio], step: int, sample_rate: int) -> None:
        for key, audio in audios.items():
            self.add_audio(f"{scope_name}/{key}", audio, step, sample_rate=sample_rate)

    @abstractmethod
    def flush(self) -> None:
        pass

    @abstractmethod
    def finish(self) -> None:
        pass

    @staticmethod
    @rank_zero_only
    def save_model(state: dict[str, Any], path: str) -> None:
        save_fsspec(state, path)

    def train_step_stats(self, step: int, stats: dict[str, float]) -> None:
        self.add_scalars(scope_name="TrainIterStats", scalars=stats, step=step)

    def train_epoch_stats(self, step: int, stats: dict[str, float]) -> None:
        self.add_scalars(scope_name="TrainEpochStats", scalars=stats, step=step)

    def train_figures(self, step: int, figures: dict[str, Figure]) -> None:
        self.add_figures(scope_name="TrainFigures", figures=figures, step=step)

    def train_audios(self, step: int, audios: dict[str, Audio], sample_rate: int) -> None:
        self.add_audios(scope_name="TrainAudios", audios=audios, step=step, sample_rate=sample_rate)

    def eval_stats(self, step: int, stats: dict[str, float]) -> None:
        self.add_scalars(scope_name="EvalStats", scalars=stats, step=step)

    def eval_figures(self, step: int, figures: dict[str, Figure]) -> None:
        self.add_figures(scope_name="EvalFigures", figures=figures, step=step)

    def eval_audios(self, step: int, audios: dict[str, Audio], sample_rate: int) -> None:
        self.add_audios(scope_name="EvalAudios", audios=audios, step=step, sample_rate=sample_rate)

    def test_audios(self, step: int, audios: dict[str, Audio], sample_rate: int) -> None:
        self.add_audios(scope_name="TestAudios", audios=audios, step=step, sample_rate=sample_rate)

    def test_figures(self, step: int, figures: dict[str, Figure]) -> None:
        self.add_figures(scope_name="TestFigures", figures=figures, step=step)
