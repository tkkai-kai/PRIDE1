"""PyTorch device helpers for NVIDIA CUDA (Sheffield) and Intel XPU (Dawn)."""
import os

import torch


def maybe_import_ipex() -> None:
    """Load Intel Extension for PyTorch on Dawn when available."""
    if os.environ.get("PRIDE_CLUSTER") != "dawn" and os.environ.get("PRIDE_IMPORT_IPEX") != "1":
        return
    try:
        import intel_extension_for_pytorch  # noqa: F401
    except ImportError:
        pass


def xpu_is_available() -> bool:
    xpu = getattr(torch, "xpu", None)
    return xpu is not None and xpu.is_available()


def resolve_torch_device(requested="auto"):
    """Map config/device strings to an available torch.device."""
    maybe_import_ipex()

    if requested in (None, "auto"):
        if torch.cuda.is_available():
            requested = "cuda"
        elif xpu_is_available():
            requested = "xpu"
        else:
            requested = "cpu"

    requested = str(requested)
    if requested == "cuda" and not torch.cuda.is_available():
        if xpu_is_available():
            requested = "xpu"
        else:
            requested = "cpu"
    if requested == "xpu" and not xpu_is_available():
        requested = "cpu"

    return torch.device(requested)


def using_accelerator(device) -> bool:
    dev = torch.device(device)
    if dev.type == "cuda":
        return torch.cuda.is_available()
    if dev.type == "xpu":
        return xpu_is_available()
    return False


def tensor_to_numpy(tensor):
    if tensor.is_cuda or (getattr(tensor, "is_xpu", False) and tensor.is_xpu):
        return tensor.detach().cpu().numpy()
    return tensor.detach().numpy()


def seed_accelerators(seed: int) -> None:
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    xpu = getattr(torch, "xpu", None)
    if xpu is not None and xpu.is_available() and hasattr(xpu, "manual_seed_all"):
        xpu.manual_seed_all(seed)
