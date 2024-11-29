import importlib.metadata

from trainer.config import TrainerArgs, TrainerConfig
from trainer.model import TrainerModel
from trainer.trainer import Trainer

__version__ = importlib.metadata.version("coqui-tts-trainer")

__all__ = ["Trainer", "TrainerArgs", "TrainerConfig", "TrainerModel"]
