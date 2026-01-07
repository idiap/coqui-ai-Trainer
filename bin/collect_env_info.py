"""Get detailed info about the working environment."""

import contextlib
import json
import os
import platform
import sys

import torch

import trainer

sys.path += [os.path.abspath(".."), os.path.abspath(".")]


def system_info():
    return {
        "OS": platform.system(),
        "architecture": platform.architecture(),
        "version": platform.version(),
        "processor": platform.processor(),
        "python": platform.python_version(),
    }


def cuda_info():
    return {
        "GPU": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
        "available": torch.cuda.is_available(),
        "version": torch.version.cuda,
    }


def package_info():
    numpy_version = "N/A"
    with contextlib.suppress(ImportError):
        import numpy as np  # noqa: PLC0415

        numpy_version = np.__version__
    return {
        "numpy": numpy_version,
        "PyTorch_version": torch.__version__,
        "PyTorch_debug": torch.version.debug,
        "Trainer": trainer.__version__,
    }


def main():
    details = {"System": system_info(), "CUDA": cuda_info(), "Packages": package_info()}
    print(json.dumps(details, indent=4, sort_keys=True))


if __name__ == "__main__":
    main()
