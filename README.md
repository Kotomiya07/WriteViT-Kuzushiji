 # WriteViT: Handwritten Text Generation with Vision Transformer

<p align="center">
  <img src="./Figures/architecture.png" alt="Model Architecture" width="800"/>
</p>

<p align="center">
  <b>
    <a href="https://arxiv.org/abs/2505.13235">ArXiv</a>
    |
    <a href="https://github.com/DAIR-Group/WriteViT">Code</a>
    |
    <a href="https://colab.research.google.com/drive/15Lswqr-aQwI-fF6yRoGYt-2pxSlC2L-R#scrollTo=abWDlzrTFa_h">
      Demo
    </a>
  </b>
</p>

<p align="center">
  <a href="https://github.com/DAIR-Group/WriteViT">
    <img alt="GitHub" src="https://img.shields.io/badge/GitHub-Repo-181717.svg?logo=github&logoColor=white">
  </a>
  <a href="https://arxiv.org/abs/2505.13235">
    <img alt="arXiv" src="https://img.shields.io/badge/arXiv-2505.13235-b31b1b.svg">
  </a>
  <a href="https://colab.research.google.com/drive/15Lswqr-aQwI-fF6yRoGYt-2pxSlC2L-R#scrollTo=abWDlzrTFa_h">
    <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
  </a>
</p>

</p>

  
## Software environment

- Python 3.10+
- PyTorch 2.7.0 / torchvision 0.22.0

## Setup & Training
Please refer to `INSTALL.md` for installation instructions of required libraries.

To visualize generated handwriting during training, you can modify the settings in `params.py`.



Download Dataset files and model from [dataset and checkpoint](https://drive.google.com/drive/folders/1ZgYS6-6l6fjKY75RJipONBByujIgf-uE?usp=sharing)

Quick setup with terminal:

```bash
git clone https://github.com/hnam-1765/WriteViT.git
cd WriteViT
pip install --upgrade --no-cache-dir gdown
gdown --id 1D_aT7CKEufR87pbfK-fF4wCr3cca6jAg && unzip ckpt.zip && rm ckpt.zip
```

To train the model

```
python train.py
```

## Kuzushiji COCO Column Experiment

This fork can train WriteViT on the Hugging Face dataset
[`Kotomiya07/kuzushiji-dataset-coco`](https://huggingface.co/datasets/Kotomiya07/kuzushiji-dataset-coco).
The dataset contains page images, `book_id`, character boxes, and column boxes. The
new loader uses `columns.bbox` and `columns.char_ids` to crop each vertical column,
rotates it into the horizontal line format expected by WriteViT, and splits data
bookwise so the same `book_id` never appears in multiple splits.

Create the environment with uv:

```bash
uv sync
```

Run a small column-level training job:

```bash
WRITEVIT_DATASET=KUZUSHIJI_COCO_COLUMN \
WRITEVIT_BATCH_SIZE=4 \
WRITEVIT_EPOCHS=5 \
uv run python train.py --no-val-preview
```

Run DDP training on 2 GPUs:

```bash
WRITEVIT_DATASET=KUZUSHIJI_COCO_COLUMN \
WRITEVIT_BATCH_SIZE=2 \
uv run torchrun --nproc_per_node=2 train.py --epochs 5 --no-val-preview
```

Run a short 2-GPU smoke job with Weights & Biases logging disabled locally:

```bash
WRITEVIT_DATASET=KUZUSHIJI_COCO_COLUMN \
WRITEVIT_BATCH_SIZE=1 \
WANDB_MODE=disabled \
uv run torchrun --nproc_per_node=2 train.py \
  --epochs 1 \
  --max-steps 1 \
  --no-val-preview \
  --wandb
```

Enable normal wandb logging by setting `WANDB_API_KEY` and using `--wandb` without
`WANDB_MODE=disabled`. Only rank 0 writes checkpoints, prints progress, and logs
to wandb.

Useful environment variables:

```bash
WRITEVIT_VAL_RATIO=0.1
WRITEVIT_TEST_RATIO=0.1
WRITEVIT_SPLIT_SEED=42
WRITEVIT_MAX_COLUMN_WIDTH=512
WRITEVIT_MAX_LABEL_LENGTH=96
WRITEVIT_HF_CACHE_DIR=./.cache/huggingface
WRITEVIT_INK_LOSS_WEIGHT=0.5
WRITEVIT_G_GRAD_CLIP=5.0
WRITEVIT_D_GRAD_CLIP=5.0
WRITEVIT_G_EMA_DECAY=0.999
WRITEVIT_GENERATOR_NOISE_INIT=0.05
WRITEVIT_D_AUG_PROB=0.35
WRITEVIT_D_CONSISTENCY_WEIGHT=0.1
WRITEVIT_NUM_WORKERS=2
WRITEVIT_FUSED_OPTIMIZER=1
WRITEVIT_AMP=1
WRITEVIT_AMP_DTYPE=bfloat16
WRITEVIT_GRAD_ACCUM_STEPS=1
WRITEVIT_COMPILE=0
WRITEVIT_CHANNELS_LAST=0
WRITEVIT_FUSED_INK_DENSITY=0
```

For unstable runs that produce nearly blank previews, keep EMA enabled, lower
`WRITEVIT_GENERATOR_NOISE_INIT`, and monitor `loss/ink`, `loss/gb_ocr`,
`loss/gb_w`, and `loss/grad_*` in wandb. The defaults above are more conservative
than the original WriteViT settings for the lower-data Kuzushiji column split.
The discriminator path also supports light tensor-space augmentation plus
consistency regularization (`loss/D_consistency`) to reduce discriminator
overfitting without changing the text labels.

Phase 5 ablation presets:

```bash
# Run D: evaluation-only baseline for the previous quality checkpoint.
uv run python scripts/phase5.py run-d-eval --execute

# Run A: reconstruction loss ablation.
uv run python scripts/phase5.py run-a-recon --execute --wandb

# Run B: writer consistency / embedding matching ablation.
uv run python scripts/phase5.py run-b-writer --execute --wandb

# Run C: discriminator augmentation / consistency ablation.
uv run python scripts/phase5.py run-c-disc --execute --wandb

# Safe smoke test for any training preset.
uv run python scripts/phase5.py run-a-recon --execute --smoke
```

The training presets default to 50 epochs, 2 processes, and batch size 2 per
process, then write checkpoints under
`saved_models/Kuzushiji-COCO-column-phase5-*`. Validation image previews are
disabled automatically in DDP so rank 0 does not run long preview/W&B image work
while the other ranks are waiting in collectives. Run `run-d-eval` after training
to generate the fixed reconstruction grids and CSV summaries.
The launcher also sets `TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=3600`; full Kuzushiji
epochs can run long enough to trigger PyTorch's default NCCL heartbeat monitor
even when collectives are still progressing. The process-group collective timeout
remains controlled separately by `WRITEVIT_DDP_TIMEOUT_MINUTES`.
`run-d-eval` writes val/test CSV summaries and reconstruction grids under
`/tmp/writevit_phase5` by default.

Speed-focused options:

```bash
WRITEVIT_BATCH_SIZE=4 \
WRITEVIT_GRAD_ACCUM_STEPS=4 \
WRITEVIT_NUM_WORKERS=4 \
WRITEVIT_AMP=1 \
WRITEVIT_AMP_DTYPE=bfloat16 \
WRITEVIT_FUSED_OPTIMIZER=1 \
WRITEVIT_COMPILE=1 \
WRITEVIT_COMPILE_MODE=reduce-overhead \
uv run python train.py --wandb
```

`WRITEVIT_GRAD_ACCUM_STEPS=4` with `WRITEVIT_BATCH_SIZE=4` gives an effective
batch size of 16 without allocating a full batch-16 backward graph. `WRITEVIT_AMP`
uses CUDA autocast; bfloat16 is the default because it is more stable than float16
for the OCR/CTC and adversarial losses in this project. `WRITEVIT_COMPILE=1`
compiles the generator and discriminator with `torch.compile`; leave it disabled
if dynamic input widths cause graph breaks.
`WRITEVIT_FUSED_INK_DENSITY=1` optionally JIT-builds
`cuda_extensions/ink_density.cu` and uses a fused CUDA kernel for the ink-density
regularizer. The Python fallback remains the default so training does not depend
on a local CUDA compiler.

Generate a column image from a trained checkpoint and a validation reference:

```bash
uv run python infer_kuzushiji_column.py \
  --checkpoint saved_models/Kuzushiji-COCO-column/model.pth \
  --split val \
  --index 0 \
  --text "いろはにほへと" \
  --output outputs/kuzushiji_column.png
```

If you want to use ```wandb``` please install it and change your auth_key in the ```train.py``` file. 

You can also modify different hyperparameters in  ```params.py``` file.

The dataset is organized as a dictionary containing lists of writer samples 

```python
{
'train': [{writer_1:[{'img': <PIL.IMAGE>, 'label':<str_label>},...]}, {writer_2:[{'img': <PIL.IMAGE>, 'label':<str_label>},...]},...], 
'test': [{writer_3:[{'img': <PIL.IMAGE>, 'label':<str_label>},...]}, {writer_4:[{'img': <PIL.IMAGE>, 'label':<str_label>},...]},...], 
}
```
 <!-- ## Run Demo using Docker
```
 docker run -it -p 7860:7860 --platform=linux/amd64 \
	registry.hf.space/ankankbhunia-hwt:latest python app.py
 ``` -->

## Handwriting generation results

 <p align="center">
<img src=Figures/Generation.png width="1000"/>
</p>


## Handwriting reconstruction results
 

 <p align="center">
<img src=Figures/Reconstruction.png width="1000"/>
</p>

## Acknowledgements

A large portion of codes in this repo is based on:[Handwriting-Transformers](https://github.com/ankanbhunia/Handwriting-Transformers) by Ankan Bhunia et al.

We thank the authors for open-sourcing their work, which has been instrumental in developing this project.

 
