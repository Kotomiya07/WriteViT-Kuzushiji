import os
import time
import argparse
from datetime import timedelta
import torch
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler

try:
    import wandb
except ImportError:
    wandb = None

#os.environ["WANDB_API_KEY"] = ""

from data.dataset import KuzushijiColumnDataset, TextDataset, TextDatasetval
from models.model import WriteViT
from params import *


def is_distributed():
    return dist.is_available() and dist.is_initialized()


def is_main_process():
    return not is_distributed() or dist.get_rank() == 0


def setup_distributed():
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size <= 1:
        return False, 0, 0, torch.device(DEVICE)

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        backend = "nccl"
        device = torch.device("cuda", local_rank)
    else:
        backend = "gloo"
        device = torch.device("cpu")

    if device.type == "cuda":
        dist.init_process_group(
            backend=backend,
            device_id=device,
            timeout=timedelta(minutes=DDP_TIMEOUT_MINUTES),
        )
    else:
        dist.init_process_group(
            backend=backend,
            timeout=timedelta(minutes=DDP_TIMEOUT_MINUTES),
        )
    return True, dist.get_rank(), local_rank, device


def cleanup_distributed():
    if is_distributed():
        dist.destroy_process_group()


def build_dataloader(split, num_examples, is_train, distributed=False):
    if DATASET == "KUZUSHIJI_COCO_COLUMN":
        max_pages = KUZUSHIJI_MAX_PAGES if is_train else 0
        dataset_obj = KuzushijiColumnDataset(
            split=split,
            num_examples=num_examples,
            max_pages=max_pages,
        )
    elif split == "train":
        dataset_obj = TextDataset(num_examples=num_examples)
    else:
        dataset_obj = TextDatasetval(num_examples=num_examples)

    sampler = None
    if distributed:
        sampler = DistributedSampler(
            dataset_obj,
            shuffle=is_train,
            drop_last=is_train,
        )

    loader_kwargs = {
        "batch_size": batch_size,
        "shuffle": is_train and sampler is None,
        "sampler": sampler,
        "num_workers": DATALOADER_NUM_WORKERS,
        "pin_memory": torch.cuda.is_available(),
        "drop_last": is_train,
        "collate_fn": dataset_obj.collate_fn,
    }
    if DATALOADER_NUM_WORKERS > 0:
        loader_kwargs["persistent_workers"] = DATALOADER_PERSISTENT_WORKERS
        loader_kwargs["prefetch_factor"] = DATALOADER_PREFETCH_FACTOR

    return torch.utils.data.DataLoader(
        dataset_obj,
        **loader_kwargs,
    ), sampler


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--resume", action="store_true", default=RESUME)
    parser.add_argument("--no-val-preview", action="store_true")
    parser.add_argument(
        "--ddp-val-preview",
        action="store_true",
        help="Allow rank-0 image previews during DDP. This can make other ranks wait.",
    )
    parser.add_argument("--no-save", action="store_true")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", default=os.environ.get("WANDB_PROJECT", "WriteViT-Kuzushiji"))
    parser.add_argument("--wandb-name", default=os.environ.get("WANDB_NAME", EXP_NAME))
    parser.add_argument("--wandb-mode", default=os.environ.get("WANDB_MODE", "online"))
    return parser.parse_args()


def scalarize_losses(losses):
    output = {}
    for key, value in losses.items():
        if torch.is_tensor(value):
            output[key] = float(value.detach().cpu())
        else:
            output[key] = float(value)
    return output


def main():
    args = parse_args()
    distributed, rank, local_rank, device = setup_distributed()
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

    if args.wandb and wandb is None:
        raise ImportError("wandb is not installed. Run `uv sync` first.")
    if distributed and not args.no_val_preview and not args.ddp_val_preview:
        args.no_val_preview = True
        if is_main_process():
            print("DDP detected: disabling validation image preview. Use --ddp-val-preview to override.")

    if is_main_process():
        init_project()
    if distributed:
        dist.barrier()

    if distributed and rank != 0:
        dist.barrier()
    dataset, train_sampler = build_dataloader(
        "train", NUM_EXAMPLES, is_train=True, distributed=distributed
    )
    if distributed and rank == 0:
        dist.barrier()
    datasetval = None
    if is_main_process() and not args.no_val_preview:
        datasetval, _ = build_dataloader(
            "val", NUM_EXAMPLES, is_train=False, distributed=False
        )
    if distributed:
        dist.barrier()


    model = WriteViT(
        backbone=BACKBONE,
        device=device,
        distributed=distributed,
        local_rank=local_rank,
    )

    os.makedirs('saved_models', exist_ok = True)
    MODEL_PATH = os.path.join('saved_models', EXP_NAME)
    if os.path.isdir(MODEL_PATH) and args.resume:
        model.load_checkpoint_state(
            torch.load(MODEL_PATH+'/model.pth', map_location=device),
            reset_discriminator=RESET_D_ON_RESUME,
        )
        if is_main_process():
            print (MODEL_PATH+' : Model loaded Successfully')
            if RESET_D_ON_RESUME:
                print ("Discriminator reset on resume")
    else:
        if is_main_process() and not os.path.isdir(MODEL_PATH):
            os.mkdir(MODEL_PATH)
    if distributed:
        dist.barrier()

    wandb_run = None
    if args.wandb and is_main_process():
        wandb_run = wandb.init(
            project=args.wandb_project,
            name=args.wandb_name,
            mode=args.wandb_mode,
            config={
                "dataset": DATASET,
                "batch_size_per_process": batch_size,
                "world_size": dist.get_world_size() if distributed else 1,
                "num_examples": NUM_EXAMPLES,
                "backbone": BACKBONE,
                "num_workers": DATALOADER_NUM_WORKERS,
                "fused_optimizer": USE_FUSED_OPTIMIZER,
                "amp": USE_AMP,
                "amp_dtype": AMP_DTYPE,
                "grad_accum_steps": GRAD_ACCUM_STEPS,
                "torch_compile": USE_TORCH_COMPILE,
                "channels_last": USE_CHANNELS_LAST,
                "g_lr": G_LR,
                "d_lr": D_LR,
                "num_critic_gocr_train": NUM_CRITIC_GOCR_TRAIN,
                "num_critic_docr_train": NUM_CRITIC_DOCR_TRAIN,
                "adv_loss_weight": ADV_LOSS_WEIGHT,
                "hinge_margin": HINGE_MARGIN,
                "d_logit_reg_weight": D_LOGIT_REG_WEIGHT,
                "reset_d_on_resume": RESET_D_ON_RESUME,
                "d_aug_prob": D_AUG_PROB,
                "d_consistency_weight": D_CONSISTENCY_WEIGHT,
                "recon_loss_weight": RECON_LOSS_WEIGHT,
                "recon_foreground_weight": RECON_FOREGROUND_WEIGHT,
                "writer_fake_weight": WRITER_FAKE_WEIGHT,
                "writer_embed_match_weight": WRITER_EMBED_MATCH_WEIGHT,
                "preview_reconstruction": PREVIEW_RECONSTRUCTION,
            },
        )


    for epoch in range(args.epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        
        start_time = time.time()
        grad_accum_steps = max(1, GRAD_ACCUM_STEPS)
        g_accum = 0
        d_accum = 0

        for i,data in enumerate(dataset):
            if args.max_steps is not None and i >= args.max_steps:
                break

            if (i % NUM_CRITIC_GOCR_TRAIN) == 0:
                model._set_input(data)
                model.optimize_G_only(
                    zero_grad=g_accum == 0,
                    loss_scale=1.0 / grad_accum_steps,
                )
                g_accum += 1
                if g_accum >= grad_accum_steps:
                    model.optimize_G_step()
                    g_accum = 0

            if (i % NUM_CRITIC_DOCR_TRAIN) == 0:

                model._set_input(data)
                model.optimize_D_OCR_W(
                    zero_grad=d_accum == 0,
                    loss_scale=1.0 / grad_accum_steps,
                )
                d_accum += 1
                if d_accum >= grad_accum_steps:
                    model.optimize_D_OCR_W_step()
                    d_accum = 0

        if g_accum > 0:
            model.optimize_G_step()
        if d_accum > 0:
            model.optimize_D_OCR_W_step()


        end_time = time.time()
        
        losses = model.get_current_losses()
        scalar_losses = scalarize_losses(losses)

        if distributed:
            dist.barrier()

        page_val = None
        page_recon = None
        if is_main_process() and not args.no_val_preview:
            data_val = next(iter(datasetval))
            val_img = data_val['img'].to(device)
            val_wcl = data_val['wcl'].to(device)
            page_val = model._generate_page(
                val_img,
                data_val['simg'].to(device),
                val_wcl,
                data_val['swids'].to(device),
            )
            if PREVIEW_RECONSTRUCTION:
                page_recon = model._generate_reconstruction_page(
                    val_img,
                    val_wcl,
                    data_val['label'],
                )

        
        # wandb.log({'loss-G': losses['G'],
        #             'loss-D': losses['D'], 
        #             'loss-Dfake': losses['Dfake'],
        #             'loss-Dreal': losses['Dreal'],
        #             'loss-OCR_fake': losses['OCR_fake'],
        #             'loss-OCR_real': losses['OCR_real'],
        #             'loss-w_fake': losses['w_fake'],
        #             'loss-w_real': losses['w_real'],
        #             'epoch' : epoch,
        #             'timeperepoch': end_time-start_time,
        #             "result":[wandb.Image(page_val*255, caption="page_val")],
        #             })

                    
 
        if is_main_process():
            log_data = {
                **{f"loss/{key}": value for key, value in scalar_losses.items()},
                "epoch": epoch,
                "time/epoch_seconds": end_time-start_time,
            }
            if wandb_run is not None:
                if page_val is not None:
                    log_data["preview_fixed_text"] = wandb.Image(
                        page_val * 255,
                        caption="fixed text: real writer feature + fixed phrases",
                    )
                if page_recon is not None:
                    log_data["preview_reconstruction"] = wandb.Image(
                        page_recon * 255,
                        caption="reconstruction-like: real writer feature + real label",
                    )
                wandb.log(log_data, step=epoch)
            print ({'EPOCH':epoch, 'TIME':end_time-start_time, 'LOSSES': scalar_losses})

        if is_main_process() and not args.no_save:
            save_paths = []
            if epoch % SAVE_MODEL == 0:
                save_paths.append(MODEL_PATH+ '/model.pth')
            if epoch % SAVE_MODEL_HISTORY == 0:
                save_paths.append(MODEL_PATH+ '/model'+str(epoch)+'.pth')
            if save_paths:
                checkpoint = model.checkpoint_state(cpu=distributed)
                for save_path in save_paths:
                    torch.save(checkpoint, save_path)
                del checkpoint
        if distributed:
            dist.barrier()

    if wandb_run is not None:
        wandb_run.finish()
    cleanup_distributed()


if __name__ == "__main__":
    
    main()
