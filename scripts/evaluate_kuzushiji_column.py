import argparse
import csv
import os
import sys
from itertools import combinations

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.nn import CTCLoss

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from data.dataset import KuzushijiColumnDataset
from models.model import WriteViT
from params import BACKBONE, DEVICE, EXP_NAME, IMG_HEIGHT, resolution
from util.fused_ink_density import foreground_density


DEFAULT_TEXTS = [
    "いろはにほへと",
    "ちりぬるを",
    "わかよたれそ",
    "つねならむ",
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        default=os.path.join("saved_models", EXP_NAME, "model.pth"),
    )
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--grid-limit", type=int, default=8)
    parser.add_argument("--diversity-writers", type=int, default=16)
    parser.add_argument("--num-examples", type=int, default=15)
    parser.add_argument("--output-dir", default="/tmp/writevit_eval")
    parser.add_argument("--fixed-text", action="append", default=None)
    parser.add_argument("--device", default=str(DEVICE))
    return parser.parse_args()


def tensor_to_uint8(image):
    array = image.detach().cpu().squeeze().numpy()
    return ((array + 1.0) * 127.5).clip(0, 255).astype(np.uint8)


def save_image_grid(rows, output_path):
    if not rows:
        return
    gap = 12
    row_gap = 18
    col_widths = [
        max(row[col].shape[1] for row in rows if col < len(row))
        for col in range(max(len(row) for row in rows))
    ]
    row_heights = [max(image.shape[0] for image in row) for row in rows]
    width = sum(col_widths) + gap * (len(col_widths) - 1)
    height = sum(row_heights) + row_gap * (len(rows) - 1)
    canvas = np.full((height, width), 255, dtype=np.uint8)
    y = 0
    for row_idx, row in enumerate(rows):
        x = 0
        for col_idx, image in enumerate(row):
            canvas[y : y + image.shape[0], x : x + image.shape[1]] = image
            x += col_widths[col_idx] + gap
        y += row_heights[row_idx] + row_gap
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    Image.fromarray(canvas, mode="L").save(output_path)


def encode_text(model, texts, device):
    encoded, lengths = model.netconverter.encode([text.encode("utf-8") for text in texts])
    return encoded.to(device), lengths.to(device)


def greedy_decode(model, logits):
    indices = logits.detach().argmax(dim=2).cpu().numpy()
    decoded = []
    for sample in indices:
        chars = []
        prev = None
        for idx in sample:
            idx = int(idx)
            if idx != 0 and idx != prev:
                chars.append(model.netconverter.alphabet[idx - 1])
            prev = idx
        decoded.append("".join(chars))
    return decoded


def edit_distance(source, target):
    dp = list(range(len(target) + 1))
    for i, source_char in enumerate(source, 1):
        prev = dp[0]
        dp[0] = i
        for j, target_char in enumerate(target, 1):
            old = dp[j]
            dp[j] = min(
                dp[j] + 1,
                dp[j - 1] + 1,
                prev + (source_char != target_char),
            )
            prev = old
    return dp[-1]


def cer(prediction, target):
    return edit_distance(prediction, target) / max(1, len(target))


def valid_width(image, label_length):
    return min(image.shape[-1], int(label_length) * resolution)


def summarize(values):
    if not values:
        return {"mean": "", "median": ""}
    array = np.asarray(values, dtype=np.float64)
    return {"mean": float(array.mean()), "median": float(np.median(array))}


def pairwise_l1(images):
    if len(images) < 2:
        return float("nan"), 0
    values = []
    for left, right in combinations(images, 2):
        width = min(left.shape[-1], right.shape[-1])
        values.append(torch.mean(torch.abs(left[..., :width] - right[..., :width])).item())
    return float(np.mean(values)), len(values)


@torch.no_grad()
def evaluate_sample(model, criterion, sample, device, fixed_texts):
    real = sample["img"].unsqueeze(0).to(device)
    writer_id = torch.tensor([sample["wcl"]], device=device)
    label = sample["label"].decode("utf-8")
    text_encode, lengths = encode_text(model, [label], device)
    feat_w = model.netW(real, writer_id, training=False)
    fake = model.netG_ema(feat_w, text_encode)

    _, logits = model.netOCR(fake)
    logits = logits.float()
    preds_size = torch.IntTensor([logits.size(1)]).to(device)
    log_probs = logits.permute(1, 0, 2).log_softmax(2)
    ctc_loss = criterion(log_probs, text_encode.detach(), preds_size, lengths.detach())
    decoded = greedy_decode(model, logits)[0]

    real_density = foreground_density(real, lengths, resolution).item()
    fake_density = foreground_density(fake, lengths, resolution).item()
    _, writer_loss = model.netW(fake, writer_id)

    fixed_images = []
    fixed_encoded, _ = encode_text(model, fixed_texts, device)
    for index in range(fixed_encoded.shape[0]):
        fixed_images.append(model.netG_ema(feat_w, fixed_encoded[index : index + 1]))

    real_width = sample["img"].shape[-1]
    fake_width = valid_width(fake, int(lengths[0].item()))
    return {
        "label": label,
        "decoded": decoded,
        "ctc": float(ctc_loss.mean().item()),
        "cer": cer(decoded, label),
        "real_density": real_density,
        "fake_density": fake_density,
        "density_abs_diff": abs(fake_density - real_density),
        "writer_loss": float(writer_loss.mean().item()),
        "real_width": int(real_width),
        "fake_width": int(fake_width),
        "width_ratio": float(fake_width / max(1, real_width)),
        "fake": fake,
        "fixed_images": fixed_images,
        "real": real,
    }


@torch.no_grad()
def writer_diversity(model, dataset, device, text, limit):
    images = []
    seen_writers = set()
    text_encode, _ = encode_text(model, [text], device)
    for idx in range(len(dataset)):
        sample = dataset[idx]
        if sample["wcl"] in seen_writers:
            continue
        seen_writers.add(sample["wcl"])
        real = sample["img"].unsqueeze(0).to(device)
        writer_id = torch.tensor([sample["wcl"]], device=device)
        feat_w = model.netW(real, writer_id, training=False)
        images.append(model.netG_ema(feat_w, text_encode))
        if len(images) >= limit:
            break
    value, pair_count = pairwise_l1(images)
    return value, pair_count, len(images)


def main():
    args = parse_args()
    device = torch.device(args.device)
    fixed_texts = args.fixed_text or DEFAULT_TEXTS
    os.makedirs(args.output_dir, exist_ok=True)

    dataset = KuzushijiColumnDataset(
        split=args.split,
        num_examples=args.num_examples,
    )
    model = WriteViT(batch_size=1, backbone=BACKBONE, device=device).to(device)
    state = torch.load(args.checkpoint, map_location=device)
    model.load_checkpoint_state(state)
    model.eval()
    criterion = CTCLoss(zero_infinity=True, reduction="none")

    rows = []
    grid_rows = []
    max_items = min(args.limit, len(dataset))
    for idx in range(max_items):
        result = evaluate_sample(model, criterion, dataset[idx], device, fixed_texts)
        rows.append(
            {
                key: result[key]
                for key in (
                    "label",
                    "decoded",
                    "ctc",
                    "cer",
                    "real_density",
                    "fake_density",
                    "density_abs_diff",
                    "writer_loss",
                    "real_width",
                    "fake_width",
                    "width_ratio",
                )
            }
        )
        if len(grid_rows) < args.grid_limit:
            image_row = [
                tensor_to_uint8(result["real"][0, :, :, : result["real_width"]]),
                tensor_to_uint8(result["fake"][0, :, :, : result["fake_width"]]),
            ]
            for fixed_image in result["fixed_images"]:
                image_row.append(tensor_to_uint8(fixed_image[0]))
            grid_rows.append(image_row)

    if not rows:
        raise RuntimeError(f"No samples were evaluated for split={args.split!r}.")

    csv_path = os.path.join(args.output_dir, f"{args.split}_metrics.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    grid_path = os.path.join(args.output_dir, f"{args.split}_generation_reconstruction_grid.png")
    save_image_grid(grid_rows, grid_path)

    fixed_eval = evaluate_sample(model, criterion, dataset[0], device, fixed_texts)
    same_writer_l1, same_writer_pairs = pairwise_l1(fixed_eval["fixed_images"])
    same_text_l1, same_text_pairs, same_text_writer_count = writer_diversity(
        model,
        dataset,
        device,
        fixed_texts[0],
        args.diversity_writers,
    )
    summary = {
        "split": args.split,
        "checkpoint": args.checkpoint,
        "count": len(rows),
        "csv": csv_path,
        "grid": grid_path,
        "same_writer_different_text_l1": same_writer_l1,
        "same_writer_different_text_pairs": same_writer_pairs,
        "same_text_different_writer_l1": same_text_l1,
        "same_text_different_writer_pairs": same_text_pairs,
        "same_text_different_writer_count": same_text_writer_count,
    }
    for key in ("ctc", "cer", "fake_density", "density_abs_diff", "writer_loss", "width_ratio"):
        metric = summarize([row[key] for row in rows])
        summary[f"{key}_mean"] = metric["mean"]
        summary[f"{key}_median"] = metric["median"]

    summary_path = os.path.join(args.output_dir, f"{args.split}_summary.csv")
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary.keys()))
        writer.writeheader()
        writer.writerow(summary)

    print(summary)


if __name__ == "__main__":
    main()
