import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch

from tests.utils.mnist import MnistModel, MnistModelConfig
from trainer.logging import logger_factory

is_cuda = torch.cuda.is_available()

os.environ["CLEARML_OFFLINE_MODE"] = "1"
os.environ["WANDB_MODE"] = "offline"


@pytest.mark.parametrize(
    "dashboard_logger",
    [
        pytest.param(
            "aim", marks=pytest.mark.skipif(sys.version_info >= (3, 13), reason="aim doesn't support Python >= 3.13")
        ),
        "clearml",
        "mlflow",
        "tensorboard",
        "wandb",
    ],
)
def test_logger(dashboard_logger, tmp_path):
    os.environ["CLEARML_CACHE_DIR"] = str(tmp_path)
    os.environ["WANDB_DIR"] = str(tmp_path)

    config = MnistModelConfig(dashboard_logger=dashboard_logger)
    config.save_json(tmp_path / "config.json")
    model = MnistModel()
    logger = logger_factory(config, tmp_path)

    logger.add_config(config)
    logger.model_weights(model, 0)
    logger.add_text("title", "text", 1)
    logger.add_scalars("train", {"a": 0.0, "b": -1.5}, 2)
    logger.add_audios("test", {"foo": np.array([0.0] * 10000)}, 3, sample_rate=16000)
    logger.add_figures("test", {"bar": plt.figure()}, 4)
    logger.add_artifact(tmp_path / "config.json", "config", "file")
    logger.flush()
    logger.finish()
