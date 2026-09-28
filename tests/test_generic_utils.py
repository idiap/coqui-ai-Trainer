import torch

from trainer.generic_utils import empty_cache, get_device, remove_experiment_folder, to_device


def test_get_device(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert get_device() == torch.device("cuda:0")

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert get_device() == torch.device("cpu")


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
