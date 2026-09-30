import pytest
import torch

from trainer.generic_utils import (
    empty_cache,
    get_device,
    is_autocast_available,
    is_pytorch_at_least_2_4,
    remove_experiment_folder,
    to_device,
)


@pytest.mark.parametrize(
    ("cuda", "mps", "expected"),
    [
        (True, False, "cuda:0"),
        (True, True, "cuda:0"),  # CUDA takes precedence
        (False, True, "mps"),
        (False, False, "cpu"),
    ],
)
def test_get_device(monkeypatch, cuda, mps, expected):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: mps)
    assert get_device() == torch.device(expected)


def test_to_device():
    device = torch.device("cpu")
    assert to_device(None, device) is None
    assert to_device("not a tensor", device) == "not a tensor"

    x = torch.arange(6).view(2, 3).t()
    assert not x.is_contiguous()
    moved = to_device(x, device)
    assert moved.is_contiguous()
    assert moved.device == device
    assert torch.equal(moved, x)


def test_empty_cache():
    # No-op on CPU, must not raise
    empty_cache(torch.device("cpu"))


def test_empty_cache_dispatches_to_mps(monkeypatch):
    calls = []
    monkeypatch.setattr(torch.mps, "empty_cache", lambda: calls.append("mps"))
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: calls.append("cuda"))

    empty_cache(torch.device("mps"))
    assert calls == ["mps"]

    empty_cache(torch.device("cpu"))
    assert calls == ["mps"]


def test_is_autocast_available():
    assert is_autocast_available("cuda")
    assert is_autocast_available("cpu")

    if is_pytorch_at_least_2_4():
        assert is_autocast_available("mps") == torch.amp.is_autocast_available("mps")
    else:
        # torch.amp.is_autocast_available() only exists from 2.4 on, so the
        # helper falls back to the backends that predate it
        assert not is_autocast_available("mps")


def test_remove_experiment_folder(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir(exist_ok=True, parents=True)

    remove_experiment_folder(run_dir)
    assert not run_dir.is_dir()

    run_dir.mkdir(exist_ok=True, parents=True)
    checkpoint = run_dir / "checkpoint.pth"
    checkpoint.touch(exist_ok=False)
    remove_experiment_folder(run_dir)
    assert checkpoint.is_file()

    remove_experiment_folder(str(run_dir) + "/")
    assert checkpoint.is_file()

    checkpoint.unlink()
    run_dir.rmdir()
