import os
from typing import Any

import torch

from trainer._types import Audio, Figure
from trainer.logging.base_dash_logger import BaseDashboardLogger
from trainer.utils.distributed import rank_zero_only

try:
    import aim
    from aim.sdk.run import Run
except ImportError as e:
    msg = "To use the Aim logger you need to install `aim`"
    raise ImportError(msg) from e


class AimLogger(BaseDashboardLogger):
    def __init__(
        self,
        repo: str | os.PathLike[Any],
        model_name: str,
        tags: str | None = None,
    ) -> None:
        self._context: dict[str, str] = {}
        self.model_name = model_name
        self.run = Run(repo=repo, experiment=model_name)
        self.run.set_artifacts_uri(f"file://{repo}/artifacts")

        # query = f"runs.name == '{model_name}'"
        # runs = self.run.repo.query_runs(query=query)

        if tags:
            for tag in tags.split(","):
                self.run.add_tag(tag)

    @property
    def context(self) -> dict[str, str]:
        return self._context

    @context.setter
    def context(self, context: dict[str, str]) -> None:
        self._context = context

    def model_weights(self, model: torch.nn.Module, step: int) -> None:
        layer_num = 1
        for name, param in model.named_parameters():
            if param.numel() == 1:
                self.run.track(param.max(), name=f"layer{layer_num}-{name}/value", step=step)
            else:
                self.run.track(param.max(), name=f"layer{layer_num}-{name}/max", step=step)
                self.run.track(param.min(), name=f"layer{layer_num}-{name}/min", step=step)
                self.run.track(param.mean(), name=f"layer{layer_num}-{name}/mean", step=step)
                self.run.track(param.std(), name=f"layer{layer_num}-{name}/std", step=step)
                self.run.track(aim.Distribution(param.detach()), name=f"layer{layer_num}-{name}/param", step=step)
                if param.grad is not None:
                    self.run.track(aim.Distribution(param.grad), name=f"layer{layer_num}-{name}/grad", step=step)
            layer_num += 1

    def add_scalar(self, title: str, value: float, step: int) -> None:
        if torch.is_tensor(value):
            value = value.item()
        self.run.track(value, name=title, step=step, context=self.context)

    def add_text(self, title: str, text: str, step: int) -> None:
        self.run.track(
            aim.Text(text),  # Pass a string you want to track
            name=title,  # The name of distributions
            step=step,  # Step index (optional)
            context=self.context,
        )

    def add_figure(self, title: str, figure: Figure, step: int) -> None:
        self.run.track(
            aim.Image(figure, f"{title}/{step}.png"),  # Pass image data and/or caption
            name=title,  # The name of image set
            step=step,  # Step index (optional)
            context=self.context,
        )

    def add_artifact(
        self, file_or_dir: str | os.PathLike[Any], name: str, artifact_type: str, aliases: list[str] | None = None
    ) -> None:
        # AIM does not support artifacts
        pass

    def add_audio(self, title: str, audio: Audio, step: int, sample_rate: int) -> None:
        self.run.track(
            aim.Audio(audio),  # Pass audio file or numpy array
            name=f"{title}/{step}.wav",  # The name of distributions
            step=step,  # Step index (optional)
            context=self.context,
        )

    def train_step_stats(self, step: int, stats: dict[str, float]) -> None:
        self.context = {"subset": "train"}
        super().train_step_stats(step, stats)

    def train_epoch_stats(self, step: int, stats: dict[str, float]) -> None:
        self.context = {"subset": "train"}
        super().train_epoch_stats(step, stats)

    def train_figures(self, step: int, figures: dict[str, Figure]) -> None:
        self.context = {"subset": "train"}
        super().train_figures(step, figures)

    def train_audios(self, step: int, audios: dict[str, Audio], sample_rate: int) -> None:
        self.context = {"subset": "train"}
        super().train_audios(step, audios, sample_rate)

    def eval_stats(self, step: int, stats: dict[str, float]) -> None:
        self.context = {"subset": "eval"}
        super().eval_stats(step, stats)

    def eval_figures(self, step: int, figures: dict[str, Figure]) -> None:
        self.context = {"subset": "eval"}
        super().eval_figures(step, figures)

    def eval_audios(self, step: int, audios: dict[str, Audio], sample_rate: int) -> None:
        self.context = {"subset": "eval"}
        super().eval_audios(step, audios, sample_rate)

    def test_audios(self, step: int, audios: dict[str, Audio], sample_rate: int) -> None:
        self.context = {"subset": "test"}
        super().test_audios(step, audios, sample_rate)

    def test_figures(self, step: int, figures: dict[str, Figure]) -> None:
        self.context = {"subset": "test"}
        super().test_figures(step, figures)

    def flush(self) -> None:
        pass

    @rank_zero_only
    def finish(self) -> None:
        self.run.close()
