import os
import tempfile
from pathlib import Path
from typing import Any

import torch

from trainer._types import Audio, Figure
from trainer.logging.base_dash_logger import BaseDashboardLogger
from trainer.utils.distributed import rank_zero_only

try:
    import mlflow
    import soundfile as sf
    from mlflow.tracking import MlflowClient
    from mlflow.tracking.context.registry import resolve_tags
    from mlflow.utils.mlflow_tags import MLFLOW_RUN_NAME
except ImportError as e:
    msg = "To use the MLflow logger you need to install: mlflow, soundfile"
    raise ImportError(msg) from e


class MLFlowLogger(BaseDashboardLogger):
    def __init__(
        self,
        log_uri: str | os.PathLike[Any],
        model_name: str,
        tags: str | None = None,
    ) -> None:
        self.model_name = model_name
        log_path = Path(log_uri)
        db_path = log_path / "mlruns.db"
        artifact_location = str(log_path / "mlruns")

        tracking_uri = f"sqlite:///{db_path}"
        mlflow.set_tracking_uri(tracking_uri)
        self.client = MlflowClient(tracking_uri=tracking_uri)

        experiment = self.client.get_experiment_by_name(model_name)
        if experiment is None:
            self.experiment_id = self.client.create_experiment(name=model_name, artifact_location=artifact_location)
        else:
            self.experiment_id = experiment.experiment_id

        if tags is not None:
            self.client.set_experiment_tag(self.experiment_id, MLFLOW_RUN_NAME, tags)
        run = self.client.create_run(experiment_id=self.experiment_id, tags=resolve_tags(tags))
        self.run_id = run.info.run_id

    def model_weights(self, model: torch.nn.Module, step: int) -> None:
        layer_num = 1
        for name, param in model.named_parameters():
            if param.numel() == 1:
                self.client.log_metric(self.run_id, f"layer{layer_num}-{name}/value", param.max().item(), step=step)
            else:
                self.client.log_metric(self.run_id, f"layer{layer_num}-{name}/max", param.max().item(), step=step)
                self.client.log_metric(self.run_id, f"layer{layer_num}-{name}/min", param.min().item(), step=step)
                self.client.log_metric(self.run_id, f"layer{layer_num}-{name}/mean", param.mean().item(), step=step)
                self.client.log_metric(self.run_id, f"layer{layer_num}-{name}/std", param.std().item(), step=step)
                # MlFlow does not support histograms
                # self.client.add_histogram("layer{}-{}/param".format(layer_num, name), param, step)
                # self.client.add_histogram("layer{}-{}/grad".format(layer_num, name), param.grad, step)
            layer_num += 1

    def add_scalar(self, title: str, value: float, step: int) -> None:
        if torch.is_tensor(value):
            value = value.item()
        self.client.log_metric(self.run_id, title, value, step)

    def add_text(self, title: str, text: str, step: int) -> None:
        self.client.log_text(self.run_id, text, f"{title}/{step}.txt")

    def add_figure(self, title: str, figure: Figure, step: int) -> None:
        self.client.log_figure(self.run_id, figure, f"{title}/{step}.png")

    def add_artifact(
        self, file_or_dir: str | os.PathLike[Any], name: str, artifact_type: str, aliases: list[str] | None = None
    ) -> None:
        self.client.log_artifacts(self.run_id, str(file_or_dir))

    def add_audio(self, title: str, audio: Audio, step: int, sample_rate: int) -> None:
        if audio.dtype == "float16":
            audio = audio.astype("float32")
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            sf.write(f.name, audio, sample_rate)
            self.client.log_artifact(self.run_id, f.name, f"{title}/{step}.wav")
            Path(f.name).unlink()

    def train_step_stats(self, step: int, stats: dict[str, float]) -> None:
        self.client.set_tag(self.run_id, "Mode", "training")
        super().train_step_stats(step, stats)

    def train_epoch_stats(self, step: int, stats: dict[str, float]) -> None:
        self.client.set_tag(self.run_id, "Mode", "training")
        super().train_epoch_stats(step, stats)

    def train_figures(self, step: int, figures: dict[str, Figure]) -> None:
        self.client.set_tag(self.run_id, "Mode", "training")
        super().train_figures(step, figures)

    def train_audios(self, step: int, audios: dict[str, Audio], sample_rate: int) -> None:
        self.client.set_tag(self.run_id, "Mode", "training")
        super().train_audios(step, audios, sample_rate)

    def eval_stats(self, step: int, stats: dict[str, float]) -> None:
        self.client.set_tag(self.run_id, "Mode", "evaluation")
        super().eval_stats(step, stats)

    def eval_figures(self, step: int, figures: dict[str, Figure]) -> None:
        self.client.set_tag(self.run_id, "Mode", "evaluation")
        super().eval_figures(step, figures)

    def eval_audios(self, step: int, audios: dict[str, Audio], sample_rate: int) -> None:
        self.client.set_tag(self.run_id, "Mode", "evaluation")
        super().eval_audios(step, audios, sample_rate)

    def test_audios(self, step: int, audios: dict[str, Audio], sample_rate: int) -> None:
        self.client.set_tag(self.run_id, "Mode", "test")
        super().test_audios(step, audios, sample_rate)

    def test_figures(self, step: int, figures: dict[str, Figure]) -> None:
        self.client.set_tag(self.run_id, "Mode", "test")
        super().test_figures(step, figures)

    def flush(self) -> None:
        pass

    @rank_zero_only
    def finish(self) -> None:
        if self.client.get_run(self.run_id):
            self.client.set_terminated(self.run_id)
