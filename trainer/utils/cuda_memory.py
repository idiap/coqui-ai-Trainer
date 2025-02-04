"""Helper to free Torch cuda memory and determine when a Torch exception might be because of OOM conditions.

credit: https://github.com/BlackHC/toma/blob/master/toma/torch_cuda_memory.py
"""

import gc

import torch

from trainer.utils.cpu_memory import is_out_of_cpu_memory


def gc_cuda() -> None:
    """Gargage collect Torch (CUDA) memory."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def get_cuda_total_memory() -> int:
    if torch.cuda.is_available():
        return torch.cuda.get_device_properties(0).total_memory
    return 0


def get_cuda_assumed_available_memory() -> int:
    if torch.cuda.is_available():
        return get_cuda_total_memory() - torch.cuda.memory_reserved()
    return 0


def get_cuda_available_memory() -> int:
    # Always allow for 1 GB overhead.
    if torch.cuda.is_available():
        return get_cuda_assumed_available_memory() - get_cuda_blocked_memory()
    return 0


def get_cuda_blocked_memory() -> int:
    if not torch.cuda.is_available():
        return 0

    available_memory = get_cuda_assumed_available_memory()
    current_block = available_memory - 2**28  # 256 MB steps
    while True:
        try:
            _ = torch.empty((current_block,), dtype=torch.uint8, device="cuda")
            break
        except RuntimeError as exception:
            if is_cuda_out_of_memory(exception):
                current_block -= 2**30
                if current_block <= 0:
                    return available_memory
            else:
                raise
    _ = None
    gc_cuda()
    return available_memory - current_block


def is_cuda_out_of_memory(exception: Exception) -> bool:
    return (
        isinstance(exception, (RuntimeError | torch.cuda.OutOfMemoryError))
        and len(exception.args) == 1
        and "CUDA out of memory." in exception.args[0]
    )


def is_cudnn_snafu(exception: Exception) -> bool:
    # For/because of https://github.com/pytorch/pytorch/issues/4107
    return (
        isinstance(exception, RuntimeError)
        and len(exception.args) == 1
        and "cuDNN error: CUDNN_STATUS_NOT_SUPPORTED." in exception.args[0]
    )


def cuda_meminfo() -> None:
    if not torch.cuda.is_available():
        return

    print("Total:", torch.cuda.memory_allocated() / 2**30, " GB Cached: ", torch.cuda.memory_reserved() / 2**30, "GB")
    print(
        "Max Total:",
        torch.cuda.max_memory_allocated() / 2**30,
        " GB Max Cached: ",
        torch.cuda.max_memory_reserved() / 2**30,
        "GB",
    )


def should_reduce_batch_size(exception: Exception) -> bool:
    return is_cuda_out_of_memory(exception) or is_cudnn_snafu(exception) or is_out_of_cpu_memory(exception)
