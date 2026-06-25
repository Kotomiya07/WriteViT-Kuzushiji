import argparse
import os
import sys
import time

import torch
from torch.profiler import ProfilerActivity, profile, record_function

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from models.model import WriteViT
from params import BACKBONE, DEVICE, NUM_EXAMPLES, batch_size
from train import build_dataloader


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--num-examples", type=int, default=NUM_EXAMPLES)
    parser.add_argument("--sort-by", default="self_cpu_time_total")
    parser.add_argument("--limit", type=int, default=20)
    return parser.parse_args()


def sync_if_cuda(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def main():
    args = parse_args()
    device = torch.device(DEVICE)
    print(
        {
            "device": str(device),
            "cuda_available": torch.cuda.is_available(),
            "batch_size": batch_size,
            "num_examples": args.num_examples,
            "steps": args.steps,
            "warmup": args.warmup,
        }
    )

    start = time.perf_counter()
    dataloader, _ = build_dataloader(
        "train",
        args.num_examples,
        is_train=True,
        distributed=False,
    )
    dataset_seconds = time.perf_counter() - start

    model = WriteViT(
        backbone=BACKBONE,
        device=device,
        distributed=False,
    )
    model.train()

    iterator = iter(dataloader)
    timings = []
    activities = [ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(ProfilerActivity.CUDA)

    total_steps = args.warmup + args.steps
    with profile(
        activities=activities,
        record_shapes=True,
        profile_memory=True,
        with_stack=False,
    ) as prof:
        for step in range(total_steps):
            next_start = time.perf_counter()
            with record_function("dataloader_next"):
                try:
                    data = next(iterator)
                except StopIteration:
                    iterator = iter(dataloader)
                    data = next(iterator)
            sync_if_cuda(device)
            next_seconds = time.perf_counter() - next_start

            train_start = time.perf_counter()
            with record_function("train_step"):
                model._set_input(data)
                model.optimize_G_only()
                model.optimize_G_step()
                model._set_input(data)
                model.optimize_D_OCR_W()
                model.optimize_D_OCR_W_step()
            sync_if_cuda(device)
            train_seconds = time.perf_counter() - train_start

            if step >= args.warmup:
                timings.append((next_seconds, train_seconds))
            prof.step()

    loader_total = sum(item[0] for item in timings)
    train_total = sum(item[1] for item in timings)
    total = loader_total + train_total
    print(
        {
            "dataset_init_seconds": round(dataset_seconds, 4),
            "dataloader_next_seconds": round(loader_total, 4),
            "train_step_seconds": round(train_total, 4),
            "dataloader_share": round(loader_total / max(total, 1e-9), 4),
            "train_share": round(train_total / max(total, 1e-9), 4),
            "avg_dataloader_next_seconds": round(loader_total / len(timings), 4),
            "avg_train_step_seconds": round(train_total / len(timings), 4),
        }
    )
    print(prof.key_averages().table(sort_by=args.sort_by, row_limit=args.limit))


if __name__ == "__main__":
    main()
