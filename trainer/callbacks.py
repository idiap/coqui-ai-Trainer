from typing import TYPE_CHECKING, Any

from trainer._types import Callback

if TYPE_CHECKING:
    from trainer import Trainer


class TrainerCallback:
    def __init__(self) -> None:
        self.callbacks_on_init_start: list[Callback] = []
        self.callbacks_on_init_end: list[Callback] = []
        self.callbacks_on_epoch_start: list[Callback] = []
        self.callbacks_on_epoch_end: list[Callback] = []
        self.callbacks_on_train_epoch_start: list[Callback] = []
        self.callbacks_on_train_epoch_end: list[Callback] = []
        self.callbacks_on_train_step_start: list[Callback] = []
        self.callbacks_on_train_step_end: list[Callback] = []
        self.callbacks_on_keyboard_interrupt: list[Callback] = []

    def parse_callbacks_dict(self, callbacks_dict: dict[str, Callback]) -> None:
        for key, value in callbacks_dict.items():
            if key == "on_init_start":
                self.callbacks_on_init_start.append(value)
            elif key == "on_init_end":
                self.callbacks_on_init_end.append(value)
            elif key == "on_epoch_start":
                self.callbacks_on_epoch_start.append(value)
            elif key == "on_epoch_end":
                self.callbacks_on_epoch_end.append(value)
            elif key == "on_train_epoch_start":
                self.callbacks_on_train_epoch_start.append(value)
            elif key == "on_train_epoch_end":
                self.callbacks_on_train_epoch_end.append(value)
            elif key == "on_train_step_start":
                self.callbacks_on_train_step_start.append(value)
            elif key == "on_train_step_end":
                self.callbacks_on_train_step_end.append(value)
            elif key == "on_keyboard_interrupt":
                self.callbacks_on_keyboard_interrupt.append(value)
            else:
                msg = f"Invalid callback key: {key}"
                raise ValueError(msg)

    def on_init_start(self, trainer: "Trainer") -> None:
        trainer._get_model().on_init_start(trainer)

        for criterion in trainer.criterion:
            if hasattr(criterion, "on_init_start"):
                criterion.on_init_start(trainer)  # type: ignore[operator]

        for optimizer in trainer.optimizer:
            if hasattr(optimizer, "on_init_start"):
                optimizer.on_init_start(trainer)

        if self.callbacks_on_init_start:
            for callback in self.callbacks_on_init_start:
                callback(trainer)

    def on_init_end(self, trainer: "Trainer") -> None:
        trainer._get_model().on_init_end(trainer)

        for criterion in trainer.criterion:
            if hasattr(criterion, "on_init_end"):
                criterion.on_init_end(trainer)  # type: ignore[operator]

        for optimizer in trainer.optimizer:
            if hasattr(optimizer, "on_init_end"):
                optimizer.on_init_end(trainer)

        if len(self.callbacks_on_init_end) > 0:
            for callback in self.callbacks_on_init_end:
                callback(trainer)

    def on_epoch_start(self, trainer: "Trainer") -> None:
        trainer._get_model().on_epoch_start(trainer)

        for criterion in trainer.criterion:
            if hasattr(criterion, "on_epoch_start"):
                criterion.on_epoch_start(trainer)  # type: ignore[operator]

        for optimizer in trainer.optimizer:
            if hasattr(optimizer, "on_epoch_start"):
                optimizer.on_epoch_start(trainer)

        if self.callbacks_on_epoch_start:
            for callback in self.callbacks_on_epoch_start:
                callback(trainer)

    def on_epoch_end(self, trainer: "Trainer") -> None:
        trainer._get_model().on_epoch_end(trainer)

        for criterion in trainer.criterion:
            if hasattr(criterion, "on_epoch_end"):
                criterion.on_epoch_end(trainer)  # type: ignore[operator]

        for optimizer in trainer.optimizer:
            if hasattr(optimizer, "on_epoch_end"):
                optimizer.on_epoch_end(trainer)

        if self.callbacks_on_epoch_end:
            for callback in self.callbacks_on_epoch_end:
                callback(trainer)

    def on_train_epoch_start(self, trainer: "Trainer") -> None:
        trainer._get_model().on_train_epoch_start(trainer)

        for criterion in trainer.criterion:
            if hasattr(criterion, "on_train_epoch_start"):
                criterion.on_train_epoch_start(trainer)  # type: ignore[operator]

        for optimizer in trainer.optimizer:
            if hasattr(optimizer, "on_train_epoch_start"):
                optimizer.on_train_epoch_start(trainer)

        if self.callbacks_on_train_epoch_start:
            for callback in self.callbacks_on_train_epoch_start:
                callback(trainer)

    def on_train_epoch_end(self, trainer: "Trainer") -> None:
        trainer._get_model().on_train_epoch_end(trainer)

        for criterion in trainer.criterion:
            if hasattr(criterion, "on_train_epoch_end"):
                criterion.on_train_epoch_end(trainer)  # type: ignore[operator]

        for optimizer in trainer.optimizer:
            if hasattr(optimizer, "on_train_epoch_end"):
                optimizer.on_train_epoch_end(trainer)

        if self.callbacks_on_train_epoch_end:
            for callback in self.callbacks_on_train_epoch_end:
                callback(trainer)

    @staticmethod
    def before_backward_pass(trainer: "Trainer", loss_dict: dict[str, Any]) -> None:
        trainer._get_model().before_backward_pass(loss_dict, trainer.optimizer)

    @staticmethod
    def before_gradient_clipping(trainer: "Trainer") -> None:
        trainer._get_model().before_gradient_clipping()

    def on_train_step_start(self, trainer: "Trainer") -> None:
        trainer._get_model().on_train_step_start(trainer)

        for criterion in trainer.criterion:
            if hasattr(criterion, "on_train_step_start"):
                criterion.on_train_step_start(trainer)  # type: ignore[operator]

        for optimizer in trainer.optimizer:
            if hasattr(optimizer, "on_train_step_start"):
                optimizer.on_train_step_start(trainer)

        if self.callbacks_on_train_step_start:
            for callback in self.callbacks_on_train_step_start:
                callback(trainer)

    def on_train_step_end(self, trainer: "Trainer") -> None:
        trainer._get_model().on_train_step_end(trainer)

        for criterion in trainer.criterion:
            if hasattr(criterion, "on_train_step_end"):
                criterion.on_train_step_end(trainer)  # type: ignore[operator]

        for optimizer in trainer.optimizer:
            if hasattr(optimizer, "on_train_step_end"):
                optimizer.on_train_step_end(trainer)

        if self.callbacks_on_train_step_end:
            for callback in self.callbacks_on_train_step_end:
                callback(trainer)

    def on_keyboard_interrupt(self, trainer: "Trainer") -> None:
        trainer._get_model().on_keyboard_interrupt(trainer)

        for criterion in trainer.criterion:
            if hasattr(criterion, "on_keyboard_interrupt"):
                criterion.on_keyboard_interrupt(trainer)  # type: ignore[operator]

        for optimizer in trainer.optimizer:
            if hasattr(optimizer, "on_keyboard_interrupt"):
                optimizer.on_keyboard_interrupt(trainer)

        if self.callbacks_on_keyboard_interrupt:
            for callback in self.callbacks_on_keyboard_interrupt:
                callback(trainer)
