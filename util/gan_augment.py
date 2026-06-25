import torch
import torch.nn.functional as F

from params import (
    D_AUG_BRIGHTNESS,
    D_AUG_CONTRAST,
    D_AUG_CUTOUT_RATIO,
    D_AUG_PROB,
    D_AUG_TRANSLATE_PIXELS,
)


def maybe_augment_for_discriminator(x):
    if D_AUG_PROB <= 0 or not torch.is_grad_enabled():
        return x
    if torch.rand((), device=x.device) >= D_AUG_PROB:
        return x

    y = x
    if D_AUG_BRIGHTNESS > 0:
        delta = (torch.rand(y.size(0), 1, 1, 1, device=y.device) * 2 - 1) * D_AUG_BRIGHTNESS
        y = y + delta

    if D_AUG_CONTRAST > 0:
        scale = 1.0 + (torch.rand(y.size(0), 1, 1, 1, device=y.device) * 2 - 1) * D_AUG_CONTRAST
        mean = y.mean(dim=(2, 3), keepdim=True)
        y = (y - mean) * scale + mean

    if D_AUG_TRANSLATE_PIXELS > 0:
        y = _random_translate(y, D_AUG_TRANSLATE_PIXELS)

    if D_AUG_CUTOUT_RATIO > 0:
        y = _random_cutout(y, D_AUG_CUTOUT_RATIO)

    return y.clamp(-1.0, 1.0)


def _random_translate(x, max_shift):
    batch, _, height, width = x.shape
    shifts_y = torch.randint(-max_shift, max_shift + 1, (batch,), device=x.device)
    shifts_x = torch.randint(-max_shift, max_shift + 1, (batch,), device=x.device)
    output = torch.empty_like(x)
    for idx in range(batch):
        translated = torch.roll(x[idx], (int(shifts_y[idx]), int(shifts_x[idx])), dims=(1, 2))
        if shifts_y[idx] > 0:
            translated[:, : int(shifts_y[idx]), :] = 1.0
        elif shifts_y[idx] < 0:
            translated[:, int(shifts_y[idx]) :, :] = 1.0
        if shifts_x[idx] > 0:
            translated[:, :, : int(shifts_x[idx])] = 1.0
        elif shifts_x[idx] < 0:
            translated[:, :, int(shifts_x[idx]) :] = 1.0
        output[idx] = translated
    return output


def _random_cutout(x, ratio):
    batch, _, height, width = x.shape
    cut_h = max(1, int(round(height * ratio)))
    cut_w = max(1, int(round(width * ratio)))
    if cut_h >= height or cut_w >= width:
        return x

    mask = torch.ones_like(x)
    tops = torch.randint(0, height - cut_h + 1, (batch,), device=x.device)
    lefts = torch.randint(0, width - cut_w + 1, (batch,), device=x.device)
    for idx in range(batch):
        top = int(tops[idx])
        left = int(lefts[idx])
        mask[idx, :, top : top + cut_h, left : left + cut_w] = 0.0
    return x * mask + (1.0 - mask)


def discriminator_consistency_loss(discriminator, x, lengths):
    if D_AUG_PROB <= 0:
        return x.new_tensor(0.0)
    with torch.no_grad():
        base = discriminator(x.detach())
    aug = maybe_augment_for_discriminator(x.detach())
    pred_aug = discriminator(aug)
    mask = torch.ones_like(pred_aug)
    if lengths is not None and len(pred_aug.shape) > 2:
        for idx in range(len(lengths)):
            mask[idx, :, :, int(lengths[idx]) :] = 0
    return F.mse_loss(pred_aug * mask, base.detach() * mask)
