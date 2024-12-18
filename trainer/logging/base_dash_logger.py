import os
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Union

from trainer.config import TrainerConfig
from trainer.io import save_fsspec
from trainer.utils.distributed import rank_zero_only

if TYPE_CHECKING:
    import matplotlib
    import numpy as np
    import plotly


# pylint: disable=too-many-public-methods
class BaseDashboardLogger(ABC):
    @abstractmethod
    def add_scalar(self, title: str, value: float, step: int) -> None:
        pass

    @abstractmethod
    def add_figure(
        self,
        title: str,
        figure: Union["matplotlib.figure.Figure", "plotly.graph_objects.Figure"],
        step: int,
    ) -> None:
        pass

    @abstractmethod
    def add_config(self, config: TrainerConfig) -> None:
        pass

    @abstractmethod
    def add_audio(self, title: str, audio: "np.ndarray", step: int, sample_rate: int) -> None:
        pass

    @abstractmethod
    def add_text(self, title: str, text: str, step: int) -> None:
        pass

    @abstractmethod
    def add_artifact(self, file_or_dir: str | os.PathLike[Any], name: str, artifact_type: str, aliases=None):
        pass

    @abstractmethod
    def add_scalars(self, scope_name: str, scalars: dict, step: int) -> None:
        pass

    @abstractmethod
    def add_figures(self, scope_name: str, figures: dict, step: int) -> None:
        pass

    @abstractmethod
    def add_audios(self, scope_name: str, audios: dict, step: int, sample_rate: int) -> None:
        pass

    @abstractmethod
    def flush(self) -> None:
        pass

    @abstractmethod
    def finish(self) -> None:
        pass

    @staticmethod
    @rank_zero_only
    def save_model(state: dict, path: str) -> None:
        save_fsspec(state, path)

    def train_step_stats(self, step: int, stats) -> None:
        self.add_scalars(scope_name="TrainIterStats", scalars=stats, step=step)

    def train_epoch_stats(self, step: int, stats) -> None:
        self.add_scalars(scope_name="TrainEpochStats", scalars=stats, step=step)

    def train_figures(self, step: int, figures) -> None:
        self.add_figures(scope_name="TrainFigures", figures=figures, step=step)

    def train_audios(self, step: int, audios, sample_rate) -> None:
        self.add_audios(scope_name="TrainAudios", audios=audios, step=step, sample_rate=sample_rate)

    def eval_stats(self, step: int, stats) -> None:
        self.add_scalars(scope_name="EvalStats", scalars=stats, step=step)

    def eval_figures(self, step: int, figures) -> None:
        self.add_figures(scope_name="EvalFigures", figures=figures, step=step)

    def eval_audios(self, step: int, audios, sample_rate: int) -> None:
        self.add_audios(scope_name="EvalAudios", audios=audios, step=step, sample_rate=sample_rate)

    def test_audios(self, step: int, audios, sample_rate: int) -> None:
        self.add_audios(scope_name="TestAudios", audios=audios, step=step, sample_rate=sample_rate)

    def test_figures(self, step: int, figures) -> None:
        self.add_figures(scope_name="TestFigures", figures=figures, step=step)
