from collections.abc import Callable
from typing import TYPE_CHECKING, Any, TypeAlias, TypedDict

import torch

if TYPE_CHECKING:
    import matplotlib.figure
    import numpy.typing as npt
    import plotly.graph_objects

    from trainer.trainer import Trainer

Audio: TypeAlias = "npt.NDArray[Any]"
Figure: TypeAlias = "matplotlib.figure.Figure | plotly.graph_objects.Figure"
LRScheduler: TypeAlias = torch.optim.lr_scheduler._LRScheduler

Callback: TypeAlias = Callable[["Trainer"], None]


class LossDict(TypedDict):
    train_loss: float
    eval_loss: float | None
