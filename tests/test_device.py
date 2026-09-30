import pytest
import torch

from tests.utils.mnist import MnistModel, MnistModelConfig, create_trainer, run_steps
from trainer.generic_utils import get_device

is_mps = torch.backends.mps.is_available()


def _gpu():
    return 0 if torch.cuda.is_available() else None


def test_trainer_uses_selected_device(tmp_path):
    """The trainer trains on whatever accelerator `get_device()` picked."""
    trainer = create_trainer(MnistModelConfig(), MnistModel(), tmp_path, _gpu())

    assert trainer.device == get_device()
    for name, param in trainer.model.named_parameters():
        assert param.device.type == trainer.device.type, name


@pytest.mark.skipif(not is_mps, reason="requires Apple Silicon (MPS)")
def test_trainer_trains_on_mps(tmp_path):
    """On Apple Silicon the model is on Metal and a real step runs there."""
    trainer = create_trainer(MnistModelConfig(), MnistModel(), tmp_path, None)

    assert trainer.device.type == "mps"
    run_steps(trainer, 0, 1)
    assert trainer.keep_avg_train["avg_loss"] > 0


def test_continue_keeps_state_on_device(tmp_path):
    """Continuing a run must not leave parameters or optimizer state on the CPU.

    `load_fsspec()` maps checkpoints to the CPU, so this guards the hand-off
    back to the training device.
    """
    train_steps = 2
    config = MnistModelConfig(save_step=train_steps - 1)
    trainer = create_trainer(config, MnistModel(), tmp_path, _gpu())
    run_steps(trainer, 0, train_steps)

    continue_path = max(tmp_path.iterdir(), key=lambda p: p.stat().st_mtime)
    continued = create_trainer(config, MnistModel(), tmp_path / "continue", _gpu(), continue_path=str(continue_path))

    device_type = continued.device.type
    for name, param in continued.model.named_parameters():
        assert param.device.type == device_type, name
    for name, buffer in continued.model.named_buffers():
        assert buffer.device.type == device_type, name

    seen_state = False
    for optimizer in continued.optimizer:
        for state in optimizer.state.values():
            for key, value in state.items():
                # `step` is a 0-dim tensor that stays on the CPU by design
                if torch.is_tensor(value) and value.dim() > 0:
                    assert value.device.type == device_type, key
                    seen_state = True
    assert seen_state, "optimizer state was not restored, so nothing was checked"
