import logging
import os
from typing import Any

from trainer.config import TrainerConfig
from trainer.logging.base_dash_logger import BaseDashboardLogger

logger = logging.getLogger("trainer")


def logger_factory(config: TrainerConfig, output_path: str | os.PathLike[Any]) -> BaseDashboardLogger:
    run_name = config.run_name
    project_name = config.project_name
    model_name = f"{project_name}@{run_name}" if project_name else run_name
    log_uri = config.logger_uri or output_path
    dashboard_logger: BaseDashboardLogger

    if config.dashboard_logger == "tensorboard":
        from trainer.logging.tensorboard_logger import TensorboardLogger  # ruff: ignore[import-outside-top-level]

        dashboard_logger = TensorboardLogger(log_uri, model_name=model_name)

        logger.info(" > Start Tensorboard: tensorboard --logdir=%s", log_uri)

    elif config.dashboard_logger == "wandb":
        from trainer.logging.wandb_logger import WandbLogger  # ruff: ignore[import-outside-top-level]

        dashboard_logger = WandbLogger(
            project=project_name,
            name=run_name,
            config=config,
            entity=config.wandb_entity,
        )

    elif config.dashboard_logger == "mlflow":
        from trainer.logging.mlflow_logger import MLFlowLogger  # ruff: ignore[import-outside-top-level]

        dashboard_logger = MLFlowLogger(log_uri=log_uri, model_name=model_name)

    elif config.dashboard_logger == "aim":
        from trainer.logging.aim_logger import AimLogger  # ruff: ignore[import-outside-top-level]

        dashboard_logger = AimLogger(repo=log_uri, model_name=model_name)

    elif config.dashboard_logger == "clearml":
        from trainer.logging.clearml_logger import ClearMLLogger  # ruff: ignore[import-outside-top-level]

        dashboard_logger = ClearMLLogger(
            output_uri=log_uri,
            local_path=output_path,
            project_name=project_name,
            task_name=run_name,
        )

    else:
        msg = f"Unknown dashboard logger: {config.dashboard_logger}"
        raise ValueError(msg)

    return dashboard_logger
