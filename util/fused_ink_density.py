import os

import torch
from torch.utils.cpp_extension import load

from params import USE_FUSED_INK_DENSITY

_EXTENSION = None
_LOAD_FAILED = False


def foreground_density(image, lengths, resolution):
    if not _can_use_extension(image):
        return _torch_foreground_density(image, lengths, resolution)
    widths = torch.clamp(lengths.to(image.device).long() * resolution, 1, image.shape[-1])
    return _InkDensity.apply(image.contiguous(), widths)


def _can_use_extension(image):
    return (
        USE_FUSED_INK_DENSITY
        and image.is_cuda
        and image.dtype == torch.float32
        and image.dim() == 4
        and _load_extension() is not None
    )


def _load_extension():
    global _EXTENSION, _LOAD_FAILED
    if _EXTENSION is not None:
        return _EXTENSION
    if _LOAD_FAILED:
        return None
    source = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "cuda_extensions", "ink_density.cu")
    )
    try:
        _EXTENSION = load(
            name="writevit_ink_density",
            sources=[source],
            extra_cuda_cflags=["-O3"],
            verbose=False,
        )
    except Exception:
        _LOAD_FAILED = True
        return None
    return _EXTENSION


class _InkDensity(torch.autograd.Function):
    @staticmethod
    def forward(ctx, image, widths):
        ext = _load_extension()
        density = ext.forward(image, widths)
        widths_clamped = torch.clamp(widths, 1, image.shape[-1])
        denom = int(widths_clamped.sum().item()) * image.shape[1] * image.shape[2]
        ctx.save_for_backward(widths)
        ctx.shape = list(image.shape)
        ctx.device = image.device
        ctx.denom = max(1, denom)
        return density

    @staticmethod
    def backward(ctx, grad_output):
        (widths,) = ctx.saved_tensors
        ext = _load_extension()
        grad_image = ext.backward(
            grad_output.contiguous(),
            widths,
            float(ctx.denom),
            ctx.shape,
            ctx.device,
        )
        return grad_image, None


def _torch_foreground_density(image, lengths, resolution):
    ink = ((1.0 - image) * 0.5).clamp(0.0, 1.0)
    if lengths is None or len(image.shape) != 4:
        return ink.mean()
    mask = torch.zeros_like(ink)
    widths = torch.clamp(lengths.to(image.device).long() * resolution, 1, image.shape[-1])
    for idx, width in enumerate(widths):
        mask[idx, :, :, : int(width.item())] = 1.0
    return (ink * mask).sum() / mask.sum().clamp_min(1.0)
