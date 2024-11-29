import datetime
import logging
import os
from dataclasses import dataclass
from typing import Any

from trainer.utils.distributed import rank_zero_only

logger = logging.getLogger("trainer")


@dataclass(frozen=True)
class tcolors:
    OKBLUE: str = "\033[94m"
    HEADER: str = "\033[95m"
    OKGREEN: str = "\033[92m"
    WARNING: str = "\033[93m"
    FAIL: str = "\033[91m"
    ENDC: str = "\033[0m"
    BOLD: str = "\033[1m"
    UNDERLINE: str = "\033[4m"


class ConsoleLogger:
    def __init__(self) -> None:
        # TODO: color code for value changes
        # use these to compare values between iterations
        self.old_train_loss_dict = None
        self.old_epoch_loss_dict = None
        self.old_eval_loss_dict: dict[str, float] = {}

    @staticmethod
    def log_with_flush(msg: str) -> None:
        if logger is not None:
            logger.info(msg)
            for handler in logger.handlers:
                handler.flush()
        else:
            print(msg, flush=True)

    @staticmethod
    def get_time() -> str:
        now = datetime.datetime.now()
        return now.strftime("%Y-%m-%d %H:%M:%S")

    @rank_zero_only
    def print_epoch_start(self, epoch: int, max_epoch: int, output_path: str | os.PathLike[Any] | None = None):
        self.log_with_flush(
            f"\n{tcolors.UNDERLINE}{tcolors.BOLD} > EPOCH: {epoch}/{max_epoch}{tcolors.ENDC}",
        )
        if output_path is not None:
            self.log_with_flush(f" --> {output_path}")

    @rank_zero_only
    def print_train_start(self) -> None:
        self.log_with_flush(f"\n{tcolors.BOLD} > TRAINING ({self.get_time()}) {tcolors.ENDC}")

    @rank_zero_only
    def print_train_step(
        self, batch_steps: int, step: int, global_step: int, loss_dict: dict, avg_loss_dict: dict
    ) -> None:
        indent = "     | > "
        self.log_with_flush("")
        log_text = f"{tcolors.BOLD}   --> TIME: {self.get_time()} -- STEP: {step}/{batch_steps} -- GLOBAL_STEP: {global_step}{tcolors.ENDC}\n"
        for key, value in loss_dict.items():
            # print the avg value if given
            if f"avg_{key}" in avg_loss_dict:
                log_text += "{}{}: {}  ({})\n".format(indent, key, str(value), str(avg_loss_dict[f"avg_{key}"]))
            else:
                log_text += f"{indent}{key}: {value!s} \n"
        self.log_with_flush(log_text)

    # pylint: disable=unused-argument
    @rank_zero_only
    def print_train_epoch_end(self, global_step: int, epoch: int, epoch_time, print_dict: dict) -> None:
        indent = "     | > "
        log_text = f"\n{tcolors.BOLD}   --> TRAIN PERFORMACE -- EPOCH TIME: {epoch_time:.2f} sec -- GLOBAL_STEP: {global_step}{tcolors.ENDC}\n"
        for key, value in print_dict.items():
            log_text += f"{indent}{key}: {value!s}\n"
        self.log_with_flush(log_text)

    @rank_zero_only
    def print_eval_start(self) -> None:
        self.log_with_flush(f"\n{tcolors.BOLD} > EVALUATION {tcolors.ENDC}\n")

    @rank_zero_only
    def print_eval_step(self, step: int, loss_dict: dict, avg_loss_dict: dict) -> None:
        indent = "     | > "
        log_text = f"{tcolors.BOLD}   --> STEP: {step}{tcolors.ENDC}\n"
        for key, value in loss_dict.items():
            # print the avg value if given
            if f"avg_{key}" in avg_loss_dict:
                log_text += "{}{}: {}  ({})\n".format(indent, key, str(value), str(avg_loss_dict[f"avg_{key}"]))
            else:
                log_text += f"{indent}{key}: {value!s} \n"
        self.log_with_flush(log_text)

    @rank_zero_only
    def print_epoch_end(self, epoch: int, avg_loss_dict: dict) -> None:
        indent = "     | > "
        log_text = f"\n  {tcolors.BOLD}--> EVAL PERFORMANCE{tcolors.ENDC}\n"
        for key, value in avg_loss_dict.items():
            # print the avg value if given
            color = ""
            sign = "+"
            diff = 0
            if key in self.old_eval_loss_dict:
                diff = value - self.old_eval_loss_dict[key]
                if diff < 0:
                    color = tcolors.OKGREEN
                    sign = ""
                elif diff > 0:
                    color = tcolors.FAIL
                    sign = "+"
            log_text += f"{indent}{key}:{color} {value!s} {tcolors.ENDC}({sign}{diff!s})\n"
        self.old_eval_loss_dict = avg_loss_dict
        self.log_with_flush(log_text)
