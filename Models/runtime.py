import torch


def current_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def best_dtype() -> torch.dtype:
    if torch.cuda.is_available():
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16
    return torch.float32
