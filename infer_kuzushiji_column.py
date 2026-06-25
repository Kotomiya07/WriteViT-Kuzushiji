import argparse
import os

import numpy as np
import torch
from PIL import Image

from data.dataset import KuzushijiColumnDataset
from models.model import WriteViT
from params import BACKBONE, DEVICE, EXP_NAME


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=os.path.join("saved_models", EXP_NAME, "model.pth"))
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--text", default=None)
    parser.add_argument("--output", default="outputs/kuzushiji_column.png")
    return parser.parse_args()


def tensor_to_image(tensor):
    array = tensor.detach().cpu().squeeze(0).numpy()
    array = ((array + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
    return Image.fromarray(array, mode="L")


def main():
    args = parse_args()
    dataset = KuzushijiColumnDataset(split=args.split, num_examples=1)
    sample = dataset[args.index]
    text = args.text or sample["label"].decode("utf-8")

    model = WriteViT(batch_size=1, backbone=BACKBONE, device=DEVICE).to(DEVICE)
    if args.checkpoint and os.path.exists(args.checkpoint):
        state = torch.load(args.checkpoint, map_location=DEVICE)
        model.load_checkpoint_state(state)
    elif args.checkpoint:
        print(f"Checkpoint not found, running with random weights: {args.checkpoint}")
    model.eval()

    real = sample["img"].unsqueeze(0).to(DEVICE)
    writer_id = torch.tensor([sample["wcl"]], device=DEVICE)
    encoded, _ = model.netconverter.encode([text.encode("utf-8")])
    encoded = encoded.to(DEVICE)

    with torch.no_grad():
        writer_features = model.netW(real, writer_id, training=False)
        fake = model.netG_ema(writer_features, encoded)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    tensor_to_image(fake[0]).save(args.output)
    print(
        {
            "output": args.output,
            "text": text,
            "reference": sample["img_path"],
            "split": args.split,
        }
    )


if __name__ == "__main__":
    main()
