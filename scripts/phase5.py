import argparse
import os
import shlex
import subprocess
import sys


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

BASE_CHECKPOINT = "saved_models/Kuzushiji-COCO-column-quality-ddp2-wandb-20260620/model.pth"

TRAIN_PRESETS = {
    "run-a-recon": {
        "description": "label-conditioned reconstruction loss ablation",
        "exp": "Kuzushiji-COCO-column-phase5-run-a-recon-20260621",
        "env": {
            "WRITEVIT_RECON_LOSS_WEIGHT": "0.5",
            "WRITEVIT_RECON_FOREGROUND_WEIGHT": "2.0",
            "WRITEVIT_WRITER_FAKE_WEIGHT": "1.5",
            "WRITEVIT_WRITER_EMBED_MATCH_WEIGHT": "0.0",
            "WRITEVIT_D_AUG_PROB": "0.0",
            "WRITEVIT_D_CONSISTENCY_WEIGHT": "0.0",
        },
    },
    "run-b-writer": {
        "description": "writer consistency and writer embedding matching ablation",
        "exp": "Kuzushiji-COCO-column-phase5-run-b-writer-20260621",
        "env": {
            "WRITEVIT_RECON_LOSS_WEIGHT": "0.25",
            "WRITEVIT_RECON_FOREGROUND_WEIGHT": "2.0",
            "WRITEVIT_WRITER_FAKE_WEIGHT": "3.0",
            "WRITEVIT_WRITER_EMBED_MATCH_WEIGHT": "0.25",
            "WRITEVIT_D_LR": "1e-5",
            "WRITEVIT_NUM_CRITIC_DOCR_TRAIN": "2",
            "WRITEVIT_HINGE_MARGIN": "1.0",
            "WRITEVIT_D_LOGIT_REG_WEIGHT": "0.005",
            "WRITEVIT_RESET_D_ON_RESUME": "1",
            "WRITEVIT_D_AUG_PROB": "0.1",
            "WRITEVIT_D_CONSISTENCY_WEIGHT": "0.05",
        },
    },
    "run-c-disc": {
        "description": "discriminator augmentation and consistency ablation",
        "exp": "Kuzushiji-COCO-column-phase5-run-c-disc-20260621",
        "env": {
            "WRITEVIT_RECON_LOSS_WEIGHT": "0.25",
            "WRITEVIT_RECON_FOREGROUND_WEIGHT": "2.0",
            "WRITEVIT_WRITER_FAKE_WEIGHT": "1.5",
            "WRITEVIT_WRITER_EMBED_MATCH_WEIGHT": "0.0",
            "WRITEVIT_D_AUG_PROB": "0.1",
            "WRITEVIT_D_CONSISTENCY_WEIGHT": "0.5",
        },
    },
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Phase 5 ablation launcher for WriteViT Kuzushiji experiments."
    )
    parser.add_argument(
        "preset",
        choices=["run-a-recon", "run-b-writer", "run-c-disc", "run-d-eval", "all-train"],
    )
    parser.add_argument("--execute", action="store_true", help="Run commands instead of printing them.")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Use a one-epoch, one-step, no-save local smoke configuration.",
    )
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--nproc-per-node", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-examples", type=int, default=15)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-mode", default=os.environ.get("WANDB_MODE", "online"))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--no-val-preview", action="store_true")
    parser.add_argument("--no-save", action="store_true")
    parser.add_argument("--checkpoint", default=BASE_CHECKPOINT)
    parser.add_argument("--eval-limit", type=int, default=100)
    parser.add_argument("--output-root", default="/tmp/writevit_phase5")
    return parser.parse_args()


def command_to_string(env, cmd):
    exports = " ".join(f"{key}={shlex.quote(value)}" for key, value in sorted(env.items()))
    return f"{exports} {' '.join(shlex.quote(part) for part in cmd)}"


def train_command(preset_name, args):
    preset = TRAIN_PRESETS[preset_name]
    epochs = args.epochs if args.epochs is not None else 50
    env = {
        "WRITEVIT_EXP_NAME": preset["exp"],
        "WRITEVIT_BATCH_SIZE": str(args.batch_size),
        "WRITEVIT_NUM_EXAMPLES": str(args.num_examples),
        "WRITEVIT_EPOCHS": str(epochs),
        "WRITEVIT_DDP_FIND_UNUSED_PARAMETERS": "1",
        "WRITEVIT_DDP_TIMEOUT_MINUTES": "30",
        "TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC": "3600",
        "WANDB_MODE": args.wandb_mode,
        **preset["env"],
    }
    cmd = [
        "uv",
        "run",
        "torchrun",
        f"--nproc_per_node={args.nproc_per_node}",
        "train.py",
        "--epochs",
        str(epochs),
    ]
    if args.max_steps is not None:
        cmd.extend(["--max-steps", str(args.max_steps)])
    if args.resume:
        cmd.append("--resume")
    if args.no_val_preview:
        cmd.append("--no-val-preview")
    if args.no_save:
        cmd.append("--no-save")
    if args.wandb:
        cmd.append("--wandb")
    return env, cmd


def eval_commands(args):
    commands = []
    for split in ("val", "test"):
        env = {}
        cmd = [
            "uv",
            "run",
            "python",
            "scripts/evaluate_kuzushiji_column.py",
            "--checkpoint",
            args.checkpoint,
            "--split",
            split,
            "--limit",
            str(args.eval_limit),
            "--num-examples",
            str(args.num_examples),
            "--output-dir",
            os.path.join(args.output_root, "run-d-eval", split),
        ]
        commands.append((env, cmd))
    return commands


def selected_commands(args):
    if args.preset == "run-d-eval":
        return eval_commands(args)
    if args.preset == "all-train":
        return [train_command(name, args) for name in TRAIN_PRESETS]
    return [train_command(args.preset, args)]


def main():
    args = parse_args()
    if args.smoke:
        args.epochs = 1
        args.max_steps = 1
        args.nproc_per_node = 1
        args.no_val_preview = True
        args.no_save = True
        args.wandb_mode = "disabled"
    if args.nproc_per_node > 1 and not args.no_val_preview:
        args.no_val_preview = True
    commands = selected_commands(args)
    for env, cmd in commands:
        print(command_to_string(env, cmd))
        if args.execute:
            run_env = os.environ.copy()
            run_env.update(env)
            subprocess.run(cmd, cwd=ROOT, env=run_env, check=True)


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as exc:
        sys.exit(exc.returncode)
