import os
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch

from trainer._types import Audio, Figure
from trainer.config import TrainerConfig
from trainer.logging.base_dash_logger import BaseDashboardLogger
from trainer.utils.distributed import rank_zero_only

try:
    import soundfile as sf
    from clearml import Task
except ImportError as e:
    msg = "To use the ClearML logger you need to install: clearml, soundfile"
    raise ImportError(msg) from e

if TYPE_CHECKING:
    from clearml.logger import Logger
    from clearml.task import Task


class ClearMLLogger(BaseDashboardLogger):
    """ClearML Logger using ClearML's native API.

    Args:
        output_uri (str): URI of the ClearML repository.
        local_path (str): Path to the local directory where the model is saved.
        project_name (str): Name of the ClearML project.
        task_name (str): Name of the ClearML task.
        tags (str): Comma separated list of tags to add to the ClearML task.
    """

    def __init__(
        self,
        output_uri: str | os.PathLike[Any],
        local_path: str | os.PathLike[Any],
        project_name: str,
        task_name: str,
        tags: str | None = None,
    ) -> None:
        self._context = None
        self.local_path = local_path
        self.task_name = task_name
        self.model_name = f"{project_name}@{task_name}"
        self.tags = tags.split(",") if tags else []
        self.run: Task = Task.init(
            project_name=project_name, task_name=task_name, tags=self.tags, output_uri=str(output_uri)
        )

        if tags:
            for tag in tags.split(","):
                self.run.add_tag(tag)

        # Get the logger for scalars, plots, etc.
        self.logger: Logger = self.run.get_logger()

    def model_weights(self, model: torch.nn.Module, step: int) -> None:
        """Log model weights to ClearML."""
        layer_num = 1
        for name, param in model.named_parameters():
            if param.numel() == 1:
                self.logger.report_scalar(f"layer{layer_num}-{name}", "value", param.item(), step)
            else:
                self.logger.report_scalar(f"layer{layer_num}-{name}", "max", param.max().item(), step)
                self.logger.report_scalar(f"layer{layer_num}-{name}", "min", param.min().item(), step)
                self.logger.report_scalar(f"layer{layer_num}-{name}", "mean", param.mean().item(), step)
                self.logger.report_scalar(f"layer{layer_num}-{name}", "std", param.std().item(), step)
                self.logger.report_histogram(
                    f"layer{layer_num}-{name}",
                    "param",
                    iteration=step,
                    values=param.detach().tolist(),
                )
                if param.grad is not None:
                    self.logger.report_histogram(
                        f"layer{layer_num}-{name}",
                        "grad",
                        iteration=step,
                        values=param.grad.detach().tolist(),
                    )
            layer_num += 1

    def add_scalar(self, title: str, value: float, step: int) -> None:
        """Log a scalar value to ClearML."""
        if torch.is_tensor(value):
            value = value.item()
        self.logger.report_scalar(title, title, value, step)

    def add_text(self, title: str, text: str, step: int) -> None:
        """Log text to ClearML."""
        self.logger.report_text(text, level=0, print_console=False)

    def add_figure(self, title: str, figure: Figure, step: int) -> None:
        """Log a figure to ClearML."""
        self.logger.report_matplotlib_figure(title, title, figure, step)

    def add_audio(self, title: str, audio: Audio, step: int, sample_rate: int) -> None:
        """Log audio to ClearML."""
        if audio.dtype == "float16":
            audio = audio.astype("float32")
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            temp_path = Path(f.name)
            sf.write(f.name, audio, sample_rate)
            self.logger.report_media(title, title, iteration=step, local_path=f.name)
            temp_path.unlink()

    @rank_zero_only
    def add_config(self, config: TrainerConfig) -> None:
        """Upload config file(s) to ClearML."""
        self.add_text("run_config", f"{config.to_json()}", 0)
        self.run.connect_configuration(name="model_config", configuration=config.to_dict())
        self.run.set_comment(config.run_description)
        self.run.upload_artifact("model_config", config.to_dict())
        self.run.upload_artifact("configs", artifact_object=os.path.join(self.local_path, "*.json"))

    @rank_zero_only
    def add_artifact(
        self, file_or_dir: str | os.PathLike[Any], name: str, artifact_type: str, aliases: list[str] | None = None
    ) -> None:
        """Upload artifact to ClearML."""
        self.run.upload_artifact(name, artifact_object=file_or_dir)

    def flush(self) -> None:
        """Flush the logger."""
        self.logger.flush()

    def finish(self) -> None:
        """Close the ClearML task."""
        if self.run:
            self.run.close()

    @staticmethod
    @rank_zero_only
    def save_model(state: Any, path: str) -> None:
        torch.save(state, path)
