from typing import TYPE_CHECKING, Union

from trainer.logging.base_dash_logger import BaseDashboardLogger

if TYPE_CHECKING:
    import matplotlib
    import numpy as np
    import plotly


class DummyLogger(BaseDashboardLogger):
    """DummyLogger that implements the API but does nothing."""

    def add_scalar(self, title: str, value: float, step: int) -> None:
        pass

    def add_figure(
        self,
        title: str,
        figure: Union["matplotlib.figure.Figure", "plotly.graph_objects.Figure"],
        step: int,
    ) -> None:
        pass

    def add_config(self, config):
        pass

    def add_audio(self, title: str, audio: "np.ndarray", step: int, sample_rate: int) -> None:
        pass

    def add_text(self, title: str, text: str, step: int) -> None:
        pass

    def add_artifact(self, file_or_dir: str, name: str, artifact_type: str, aliases=None):
        pass

    def add_scalars(self, scope_name: str, scalars: dict, step: int):
        pass

    def add_figures(self, scope_name: str, figures: dict, step: int):
        pass

    def add_audios(self, scope_name: str, audios: dict, step: int, sample_rate: int):
        pass

    def flush(self):
        pass

    def finish(self):
        pass

    def train_step_stats(self, step, stats):
        self.add_scalars(scope_name="TrainIterStats", scalars=stats, step=step)

    def train_epoch_stats(self, step, stats):
        self.add_scalars(scope_name="TrainEpochStats", scalars=stats, step=step)

    def train_figures(self, step, figures):
        self.add_figures(scope_name="TrainFigures", figures=figures, step=step)

    def train_audios(self, step, audios, sample_rate):
        self.add_audios(scope_name="TrainAudios", audios=audios, step=step, sample_rate=sample_rate)

    def eval_stats(self, step, stats):
        self.add_scalars(scope_name="EvalStats", scalars=stats, step=step)

    def eval_figures(self, step, figures):
        self.add_figures(scope_name="EvalFigures", figures=figures, step=step)

    def eval_audios(self, step, audios, sample_rate):
        self.add_audios(scope_name="EvalAudios", audios=audios, step=step, sample_rate=sample_rate)

    def test_audios(self, step, audios, sample_rate):
        self.add_audios(scope_name="TestAudios", audios=audios, step=step, sample_rate=sample_rate)

    def test_figures(self, step, figures):
        self.add_figures(scope_name="TestFigures", figures=figures, step=step)
