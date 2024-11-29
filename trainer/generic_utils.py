import datetime
import os
import subprocess
from collections.abc import ItemsView
from pathlib import Path
from typing import Any

import fsspec
import torch
from packaging.version import Version

from trainer.config import TrainerConfig
from trainer.logger import logger


def is_pytorch_at_least_2_3() -> bool:
    """Check if the installed Pytorch version is 2.3 or higher."""
    return Version(torch.__version__) >= Version("2.3")


def is_pytorch_at_least_2_4() -> bool:
    """Check if the installed Pytorch version is 2.4 or higher."""
    return Version(torch.__version__) >= Version("2.4")


def isimplemented(obj: Any, method_name: str) -> bool:
    """Check if a method is implemented in a class."""
    if method_name in dir(obj) and callable(getattr(obj, method_name)):
        try:
            obj.__getattribute__(method_name)()  # pylint: disable=bad-option-value, unnecessary-dunder-call
        except NotImplementedError:
            return False
        except Exception:
            return True
        return True
    return False


def to_cuda(x: torch.Tensor) -> torch.Tensor:
    if x is None:
        return None
    if torch.is_tensor(x):
        x = x.contiguous()
        if torch.cuda.is_available():
            x = x.cuda(non_blocking=True)
    return x


def get_cuda() -> tuple[bool, torch.device]:
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return use_cuda, device


def get_git_branch() -> str:
    try:
        out = subprocess.check_output(["git", "branch"]).decode("utf8")
        current = next(line for line in out.split("\n") if line.startswith("*"))
        current.replace("* ", "")
    except subprocess.CalledProcessError:
        current = "inside_docker"
    except (FileNotFoundError, StopIteration):
        current = "unknown"
    return current


def get_commit_hash() -> str:
    """Get current git commit hash.

    Source:
    https://stackoverflow.com/questions/14989858/get-the-current-git-hash-in-a-python-script
    """
    try:
        commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).decode().strip()
    # Not copying .git folder into docker container
    except (subprocess.CalledProcessError, FileNotFoundError):
        commit = "0000000"
    return commit


def get_experiment_folder_path(root_path: str | os.PathLike[Any], model_name: str) -> Path:
    """Get an experiment folder path with the current date and time."""
    date_str = datetime.datetime.now().strftime("%B-%d-%Y_%I+%M%p")
    commit_hash = get_commit_hash()
    return Path(root_path) / f"{model_name}-{date_str}-{commit_hash}"


def remove_experiment_folder(experiment_path: str | os.PathLike[Any]) -> None:
    """Check folder if there is a checkpoint, otherwise remove the folder."""
    experiment_path = str(experiment_path)
    fs = fsspec.get_mapper(experiment_path).fs
    checkpoint_files = fs.glob(os.path.join(experiment_path, "*.pth"))
    if not checkpoint_files:
        if fs.exists(experiment_path):
            fs.rm(experiment_path, recursive=True)
            logger.info(" ! Run is removed from %s", experiment_path)
    else:
        logger.info(" ! Run is kept in %s", experiment_path)


def count_parameters(model: torch.nn.Module) -> int:
    r"""Count number of trainable parameters in a network."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def set_partial_state_dict(model_dict: dict, checkpoint_state: dict, c: TrainerConfig) -> dict:
    # Partial initialization: if there is a mismatch with new and old layer, it is skipped.
    for k in checkpoint_state:
        if k not in model_dict:
            logger.info(" | > Layer missing in the model definition: %s", k)
    for k in model_dict:
        if k not in checkpoint_state:
            logger.info(" | > Layer missing in the checkpoint: %s", k)
    for k, v in checkpoint_state.items():
        if k in model_dict and v.numel() != model_dict[k].numel():
            logger.info(" | > Layer dimention missmatch between model definition and checkpoint: %s", k)
    # 1. filter out unnecessary keys
    pretrained_dict = {k: v for k, v in checkpoint_state.items() if k in model_dict}
    # 2. filter out different size layers
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if v.numel() == model_dict[k].numel()}
    # 3. skip reinit layers
    if c.has("reinit_layers") and c.reinit_layers is not None:
        for reinit_layer_name in c.reinit_layers:
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if reinit_layer_name not in k}
    # 4. overwrite entries in the existing state dict
    model_dict.update(pretrained_dict)
    logger.info(" | > %i / %i layers are restored.", len(pretrained_dict), len(model_dict))
    return model_dict


class KeepAverage:
    def __init__(self) -> None:
        self.avg_values: dict[str, float] = {}
        self.iters: dict[str, int] = {}

    def __getitem__(self, key: str) -> Any:
        return self.avg_values[key]

    def items(self) -> ItemsView[str, Any]:
        return self.avg_values.items()

    def add_value(self, name: str, init_val: float = 0, init_iter: int = 0) -> None:
        self.avg_values[name] = init_val
        self.iters[name] = init_iter

    def update_value(self, name: str, value: float, *, weighted_avg: bool = False) -> None:
        if name not in self.avg_values:
            # add value if not exist before
            self.add_value(name, init_val=value)
        # else update existing value
        elif weighted_avg:
            self.avg_values[name] = 0.99 * self.avg_values[name] + 0.01 * value
            self.iters[name] += 1
        else:
            self.avg_values[name] = self.avg_values[name] * self.iters[name] + value
            self.iters[name] += 1
            self.avg_values[name] /= self.iters[name]

    def add_values(self, name_dict: dict[str, float]) -> None:
        for key, value in name_dict.items():
            self.add_value(key, init_val=value)

    def update_values(self, value_dict: dict[str, float]) -> None:
        for key, value in value_dict.items():
            self.update_value(key, value)
