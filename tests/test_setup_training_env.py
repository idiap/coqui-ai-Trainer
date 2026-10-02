import torch

from trainer import TrainerArgs, TrainerConfig
from trainer.trainer import Trainer


def _setup(monkeypatch, device, *, autocast_available, mixed_precision=True):
    monkeypatch.setattr("trainer.trainer.setup_torch_training_env", lambda **kwargs: (device, 1))
    monkeypatch.setattr("trainer.trainer.is_autocast_available", lambda device_type: autocast_available)

    config = TrainerConfig(mixed_precision=mixed_precision)
    device, num_gpus = Trainer.setup_training_environment(args=TrainerArgs(), config=config, gpu=None)
    return config, device, num_gpus


def test_mixed_precision_kept_when_autocast_available(monkeypatch):
    config, device, num_gpus = _setup(monkeypatch, torch.device("mps"), autocast_available=True)
    assert device == torch.device("mps")
    assert num_gpus == 1
    assert config.mixed_precision


def test_mixed_precision_disabled_when_autocast_unavailable(monkeypatch):
    config, device, _ = _setup(monkeypatch, torch.device("mps"), autocast_available=False)
    assert device == torch.device("mps")
    assert not config.mixed_precision


def test_mixed_precision_untouched_when_not_requested(monkeypatch):
    config, _, _ = _setup(monkeypatch, torch.device("cpu"), autocast_available=False, mixed_precision=False)
    assert not config.mixed_precision
