import copy
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn import CTCLoss
from torch.nn.parallel import DistributedDataParallel as DDP
import os
import cv2
from tqdm import tqdm
from params import *
from .BigGAN_networks import Discriminator
from util.util import (
    loss_hinge_dis,
    loss_hinge_gen,
    padding,
)
from util.gan_augment import maybe_augment_for_discriminator
from util.fused_ink_density import foreground_density

from data.dataset import TextDataset, TextDatasetval
import shutil
from .recognizer import ViT_OCR
from .Generator import Generator
from .Writer import Writer, strLabelConverter


def _make_adam(parameters, lr):
    kwargs = {
        "lr": lr,
        "betas": (0.0, 0.999),
        "weight_decay": 0,
        "eps": 1e-8,
    }
    if USE_FUSED_OPTIMIZER and torch.cuda.is_available():
        try:
            return torch.optim.Adam(parameters, fused=True, **kwargs)
        except TypeError:
            pass
    return torch.optim.Adam(parameters, **kwargs)


def _amp_dtype():
    if AMP_DTYPE == "bfloat16":
        return torch.bfloat16
    return torch.float16


class WriteViT(nn.Module):

    def __init__(
        self,
        batch_size=batch_size,
        backbone="resnet18",
        device=None,
        distributed=False,
        local_rank=0,
    ):
        super(WriteViT, self).__init__()

        self.batch_size = batch_size
        self.epsilon = 1e-7
        self.device = torch.device(device or DEVICE)
        self.distributed = distributed
        self.local_rank = local_rank
        self.netG = Generator(device=self.device).to(self.device)
        self.netG_ema = copy.deepcopy(self.netG).to(self.device)
        self.netG_ema.eval()
        for param in self.netG_ema.parameters():
            param.requires_grad_(False)
        self.netD = Discriminator().to(self.device)
        self.netW =  Writer().to(self.device)
        self.netconverter = strLabelConverter(ALPHABET)
        self.netOCR = ViT_OCR(backbone=backbone).to(self.device)
        self.OCR_criterion = CTCLoss(zero_infinity=True, reduction="none")
        self.use_amp = USE_AMP and self.device.type == "cuda"
        self.amp_dtype = _amp_dtype()
        self.use_grad_scaler = self.use_amp and self.amp_dtype == torch.float16
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_grad_scaler)

        if USE_CHANNELS_LAST and self.device.type == "cuda":
            self.netG = self.netG.to(memory_format=torch.channels_last)
            self.netG_ema = self.netG_ema.to(memory_format=torch.channels_last)
            self.netD = self.netD.to(memory_format=torch.channels_last)
            self.netW = self.netW.to(memory_format=torch.channels_last)
            self.netOCR = self.netOCR.to(memory_format=torch.channels_last)

        if USE_TORCH_COMPILE and hasattr(torch, "compile"):
            self.netG = torch.compile(self.netG, mode=COMPILE_MODE)
            self.netD = torch.compile(self.netD, mode=COMPILE_MODE)

        if self.distributed:
            ddp_kwargs = {}
            if self.device.type == "cuda":
                ddp_kwargs = {"device_ids": [self.local_rank], "output_device": self.local_rank}
            ddp_kwargs["find_unused_parameters"] = DDP_FIND_UNUSED_PARAMETERS
            self.netG = DDP(self.netG, **ddp_kwargs)
            self.netD = DDP(self.netD, **ddp_kwargs)
            self.netW = DDP(self.netW, **ddp_kwargs)
            self.netOCR = DDP(self.netOCR, **ddp_kwargs)

        self.optimizer_G = _make_adam(self.netG.parameters(), G_LR)
        self.optimizer_OCR = _make_adam(self.netOCR.parameters(), OCR_LR)
        self.optimizer_D = _make_adam(self.netD.parameters(), D_LR)
        self.optimizer_wl = _make_adam(self.netW.parameters(), W_LR)
        self.optimizers = [
            self.optimizer_G,
            self.optimizer_OCR,
            self.optimizer_D,
            self.optimizer_wl,
        ]

        self.optimizer_G.zero_grad()
        self.optimizer_OCR.zero_grad()
        self.optimizer_D.zero_grad()
        self.optimizer_wl.zero_grad()

        self.loss_G = 0
        self.loss_D = 0
        self.loss_Dfake = 0
        self.loss_Dreal = 0
        self.loss_OCR_fake = 0
        self.loss_OCR_real = 0
        self.loss_w_fake = 0
        self.loss_w_real = 0
        self.Lcycle1 = 0
        self.Lcycle2 = 0
        self.lda1 = 0
        self.lda2 = 0
        self.KLD = 0
        self.loss_patch_real = 0
        self.loss_patch_fake = 0
        self.loss_patch = 0
        self.loss_ink = 0
        self.loss_recon = 0
        self.loss_writer_embed = 0
        self.grad_balance_ocr = 1.0
        self.grad_balance_w = 1.0
        self.grad_norm_G = 0.0
        self.grad_norm_D = 0.0
        self.grad_norm_OCR = 0.0
        self.grad_norm_W = 0.0
        self.loss_D_consistency = 0.0
        self.loss_D_logit_reg = 0.0
        self.score_Dreal = 0.0
        self.score_Dfake = 0.0

        if os.path.exists(WORDS_PATH):
            with open(WORDS_PATH, "rb") as f:
                self.lex = f.read().splitlines()
        else:
            self.lex = [word.encode("utf-8") for word in MY_STRING.split(" ") if word]

        lex = []
        lex_upper_number = []
        max_lex_len = (
            KUZUSHIJI_MAX_LABEL_LENGTH
            if DATASET == "KUZUSHIJI_COCO_COLUMN"
            else 19
        )

        for word in self.lex:
            try:
                word = word.decode("utf-8")
            except:
                continue

            if len(word) <= max_lex_len and all(char in self.netconverter.dict for char in word):
                if word.isupper() or word.isdigit():
                    lex_upper_number.append(word)
                else:
                    lex.append(word)

        self.lex = lex or [word.encode("utf-8").decode("utf-8") for word in MY_STRING.split(" ") if word]
        self.lex_upper_number = lex_upper_number

        self.fake_y_dist = torch.distributions.Categorical(
            torch.tensor([1.0 / len(self.lex)] * len(self.lex))
        )
        my_string = MY_STRING
        self.text = [j.encode() for j in my_string.split(" ")]
        self.eval_text_encode, self.eval_len_text = self.netconverter.encode(self.text)
        self.eval_text_encode = self.eval_text_encode.to(self.device).repeat(
            self.batch_size, 1, 1
        )

    def _module(self, module):
        if hasattr(module, "module"):
            module = module.module
        return module._orig_mod if hasattr(module, "_orig_mod") else module

    def autocast(self):
        return torch.amp.autocast(
            device_type=self.device.type,
            dtype=self.amp_dtype,
            enabled=self.use_amp,
        )

    def backward_loss(self, loss):
        if self.use_grad_scaler:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

    def step_optimizer(self, optimizer):
        if self.use_grad_scaler:
            self.scaler.step(optimizer)
        else:
            optimizer.step()

    def update_scaler(self):
        if self.use_grad_scaler:
            self.scaler.update()

    def checkpoint_state(self, cpu=False):
        state = {
            "netG": self._module(self.netG).state_dict(),
            "netG_ema": self.netG_ema.state_dict(),
            "netD": self._module(self.netD).state_dict(),
            "netW": self._module(self.netW).state_dict(),
            "netOCR": self._module(self.netOCR).state_dict(),
        }
        if not cpu:
            return state
        return {
            name: {
                key: value.detach().cpu()
                if torch.is_tensor(value)
                else value
                for key, value in module_state.items()
            }
            for name, module_state in state.items()
        }

    def load_checkpoint_state(self, state, reset_discriminator=False):
        if all(key in state for key in ("netG", "netD", "netW", "netOCR")):
            self._module(self.netG).load_state_dict(state["netG"], strict=False)
            self.netG_ema.load_state_dict(state.get("netG_ema", state["netG"]), strict=False)
            if not reset_discriminator:
                self._module(self.netD).load_state_dict(state["netD"])
            self._module(self.netW).load_state_dict(state["netW"])
            self._module(self.netOCR).load_state_dict(state["netOCR"])
        else:
            self.load_state_dict(state)
            self.netG_ema.load_state_dict(self._module(self.netG).state_dict())

    @torch.no_grad()
    def update_g_ema(self):
        if G_EMA_DECAY <= 0:
            self.netG_ema.load_state_dict(self._module(self.netG).state_dict())
            return
        source = self._module(self.netG).state_dict()
        target = self.netG_ema.state_dict()
        for key, value in source.items():
            if torch.is_floating_point(value):
                target[key].mul_(G_EMA_DECAY).add_(value, alpha=1.0 - G_EMA_DECAY)
            else:
                target[key].copy_(value)

    @staticmethod
    def _pad_preview_height(image, height):
        if image.shape[0] == height:
            return image
        return np.concatenate(
            [image, np.ones([height - image.shape[0], image.shape[1]])],
            0,
        )

    @staticmethod
    def _add_preview_labels(real_page, fake_page):
        real_page = np.rot90(real_page, k=3)
        fake_page = np.rot90(fake_page, k=3)

        height = max(real_page.shape[0], fake_page.shape[0])
        real_page = WriteViT._pad_preview_height(real_page, height)
        fake_page = WriteViT._pad_preview_height(fake_page, height)

        separator_width = 4
        separator = np.zeros([height, separator_width])
        body = np.concatenate([real_page, separator, fake_page], 1)

        label_height = 40
        label_band = np.ones([label_height, body.shape[1]], dtype=np.uint8) * 255
        label_band[:, real_page.shape[1] : real_page.shape[1] + separator_width] = 0

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8
        thickness = 2
        y = 28
        for label, start_x, width in (
            ("Real", 0, real_page.shape[1]),
            ("Fake", real_page.shape[1] + separator_width, fake_page.shape[1]),
        ):
            text_width, _ = cv2.getTextSize(label, font, font_scale, thickness)[0]
            x = start_x + max(0, (width - text_width) // 2)
            cv2.putText(
                label_band,
                label,
                (x, y),
                font,
                font_scale,
                0,
                thickness,
                cv2.LINE_AA,
            )

        labeled = np.concatenate([body, label_band.astype(np.float32) / 255.0], 0)
        return labeled.astype(np.float32)

    def save_images_for_fid_calculation(self, dataloader, epoch, mode="train"):


        self.real_base = os.path.join("saved_images", EXP_NAME, "Real")
        self.fake_base = os.path.join("saved_images", EXP_NAME, "Fake")

        if os.path.isdir(self.real_base):
            shutil.rmtree(self.real_base)
        if os.path.isdir(self.fake_base):
            shutil.rmtree(self.fake_base)

        os.makedirs(self.real_base, exist_ok=True)
        os.makedirs(self.fake_base, exist_ok=True)

        # =========================
        # Save fake images
        # =========================
        with torch.no_grad():
            for step, data in enumerate(tqdm(dataloader)):
                self.sdata = data["img"].to(self.device)
                self.label = data["label"]
                writer_ids = data["wcl"]   # nhãn id người viết, shape thường là [B]

                self.text_encode_fake, self.len_text_fake = self.netconverter.encode(self.label)
                self.text_encode_fake = self.text_encode_fake.to(self.device)

                feat_w, _ = self.netW(self.sdata.detach(), writer_ids.to(self.device))
                self.fakes = self.netG(feat_w, self.text_encode_fake)
                fake_images = self.fakes.detach().cpu().numpy()

                # fake_images: thường là [B, C, H, W] hoặc [B, N, H, W]
                for i in range(fake_images.shape[0]):
                    writer_id = writer_ids[i].item() if torch.is_tensor(writer_ids[i]) else int(writer_ids[i])
                    if mode == "train":
                        writer_fake_dir = os.path.join(self.fake_base, str(writer_id))
                        os.makedirs(writer_fake_dir, exist_ok=True)
                    else:
                        writer_fake_dir = self.fake_base

                    for j in range(fake_images.shape[1]):
                        img = 255 * (((fake_images[i, j]) + 1) / 2)
                        img = padding(img)

                        filename = f"{step}_{i}_{j}.png"
                        cv2.imwrite(
                            os.path.join(writer_fake_dir, filename),
                            img,
                        )

        # =========================
        # Save real images
        # =========================
        for step, data in enumerate(tqdm(dataloader)):
            real_images = data["img"].numpy()
            writer_ids = data["wcl"]

            for i in range(real_images.shape[0]):
                writer_id = writer_ids[i].item() if torch.is_tensor(writer_ids[i]) else int(writer_ids[i])
                if mode == "train":
                    writer_real_dir = os.path.join(self.real_base, str(writer_id))
                    os.makedirs(writer_real_dir, exist_ok=True)
                else:
                    writer_real_dir = self.real_base

                for j in range(real_images.shape[1]):
                    img = 255 * ((real_images[i, j] + 1) / 2)
                    img = padding(img)

                    filename = f"{step}_{i}_{j}.png"
                    cv2.imwrite(
                        os.path.join(writer_real_dir, filename),
                        img,
                    )

        return self.real_base, self.fake_base

    def _generate_page(
        self, img, ST, wcl,SLEN, eval_text_encode=None, eval_len_text=None
    ):

        if eval_text_encode == None:
            eval_text_encode = self.eval_text_encode
        if eval_len_text == None:
            eval_len_text = self.eval_len_text

        feat_w = self._module(self.netW)(img.detach(), wcl, training=False)

        self.fakes = self.netG_ema.Eval(feat_w, eval_text_encode)

        page1s = []
        page2s = []

        for batch_idx in range(self.batch_size):

            word_t = []
            word_l = []

            gap = np.ones([IMG_HEIGHT, 16])

            line_wids = []

            for idx, fake_ in enumerate(self.fakes):

                word_t.append(
                    (
                        fake_[batch_idx, 0, :, : eval_len_text[idx] * resolution]
                        .cpu()
                        .numpy()
                        + 1
                    )
                    / 2
                )

                word_t.append(gap)

                if len(word_t) == 16 or idx == len(self.fakes) - 1:

                    line_ = np.concatenate(word_t, -1)

                    word_l.append(line_)
                    line_wids.append(line_.shape[1])

                    word_t = []

            gap_h = np.ones([16, max(line_wids)])

            page_ = []

            for l in word_l:

                pad_ = np.ones([IMG_HEIGHT, max(line_wids) - l.shape[1]])

                page_.append(np.concatenate([l, pad_], 1))
                page_.append(gap_h)

            page1 = np.concatenate(page_, 0)

            word_t = []
            word_l = []

            gap = np.ones([IMG_HEIGHT, 16])

            line_wids = []

            sdata_ = [i.unsqueeze(1) for i in torch.unbind(ST, 1)]

            for idx, st in enumerate((sdata_)):

                word_t.append(
                    (
                        st[batch_idx, 0, :, : int(SLEN.cpu().numpy()[batch_idx][idx])]
                        .cpu()
                        .numpy()
                        + 1
                    )
                    / 2
                )

                word_t.append(gap)

                if len(word_t) == 16 or idx == len(sdata_) - 1:

                    line_ = np.concatenate(word_t, -1)

                    word_l.append(line_)
                    line_wids.append(line_.shape[1])

                    word_t = []

            gap_h = np.ones([16, max(line_wids)])

            page_ = []

            for l in word_l:

                pad_ = np.ones([IMG_HEIGHT, max(line_wids) - l.shape[1]])

                page_.append(np.concatenate([l, pad_], 1))
                page_.append(gap_h)

            page2 = np.concatenate(page_, 0)

            merge_w_size = max(page1.shape[0], page2.shape[0])

            if page1.shape[0] != merge_w_size:

                page1 = np.concatenate(
                    [page1, np.ones([merge_w_size - page1.shape[0], page1.shape[1]])], 0
                )

            if page2.shape[0] != merge_w_size:

                page2 = np.concatenate(
                    [page2, np.ones([merge_w_size - page2.shape[0], page2.shape[1]])], 0
                )

            page1s.append(page1)
            page2s.append(page2)

            # page = np.concatenate([page2, page1], 1)

        page1s_ = np.concatenate(page1s, 0)
        max_wid = max([i.shape[1] for i in page2s])
        padded_page2s = []

        for para in page2s:
            padded_page2s.append(
                np.concatenate(
                    [para, np.ones([para.shape[0], max_wid - para.shape[1]])], 1
                )
            )

        padded_page2s_ = np.concatenate(padded_page2s, 0)

        return self._add_preview_labels(padded_page2s_, page1s_)

    def get_current_losses(self):

        losses = {}

        losses["G"] = self.loss_G
        losses["D"] = self.loss_D
        losses["Dfake"] = self.loss_Dfake
        losses["Dreal"] = self.loss_Dreal
        losses["OCR_fake"] = self.loss_OCR_fake
        losses["OCR_real"] = self.loss_OCR_real
        losses["w_fake"] = self.loss_w_fake
        losses["w_real"] = self.loss_w_real
        losses["cycle1"] = self.Lcycle1
        losses["cycle2"] = self.Lcycle2
        losses["lda1"] = self.lda1
        losses["lda2"] = self.lda2
        losses["KLD"] = self.KLD
        losses["patch_real"] = self.loss_patch_real
        losses["patch_fake"] = self.loss_patch_fake
        losses["patch"] = self.loss_patch
        losses["ink"] = self.loss_ink
        losses["recon"] = self.loss_recon
        losses["writer_embed"] = self.loss_writer_embed
        losses["gb_ocr"] = self.grad_balance_ocr
        losses["gb_w"] = self.grad_balance_w
        losses["grad_G"] = self.grad_norm_G
        losses["grad_D"] = self.grad_norm_D
        losses["grad_OCR"] = self.grad_norm_OCR
        losses["grad_W"] = self.grad_norm_W
        losses["D_consistency"] = self.loss_D_consistency
        losses["D_logit_reg"] = self.loss_D_logit_reg
        losses["Dreal_score"] = self.score_Dreal
        losses["Dfake_score"] = self.score_Dfake

        return losses

    def load_networks(self, epoch):
        BaseModel.load_networks(self, epoch)
        if self.opt.single_writer:
            load_filename = "%s_z.pkl" % (epoch)
            load_path = os.path.join(self.save_dir, load_filename)
            self.z = torch.load(load_path)

    def _set_input(self, input):
        self.input = input

    def set_requires_grad(self, nets, requires_grad=False):
        """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    @staticmethod
    def _clip_grad_norm(parameters, max_norm):
        if max_norm <= 0:
            return torch.tensor(0.0)
        params = [param for param in parameters if param.grad is not None]
        if not params:
            return torch.tensor(0.0)
        return torch.nn.utils.clip_grad_norm_(params, max_norm)

    def _foreground_density(self, image, lengths=None):
        return foreground_density(image, lengths, resolution)

    def _forward_module(self, module, *args, **kwargs):
        return self._module(module)(*args, **kwargs)

    def _valid_width_mask(self, image, lengths):
        batch, _, height, width = image.shape
        mask = image.new_zeros(batch, 1, height, width)
        valid_widths = torch.clamp(lengths.to(image.device).long() * resolution, min=1, max=width)
        for idx, valid_width in enumerate(valid_widths.tolist()):
            mask[idx, :, :, :valid_width] = 1.0
        return mask

    def _reconstruction_losses(self, feat_w):
        if RECON_LOSS_WEIGHT <= 0 and WRITER_EMBED_MATCH_WEIGHT <= 0:
            zero = self.real.new_tensor(0.0)
            return zero, zero

        recon_fake = self.netG(feat_w, self.text_encode)
        real_resized = F.interpolate(
            self.real.detach(),
            size=recon_fake.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        mask = self._valid_width_mask(recon_fake, self.len_text.detach())
        foreground = (real_resized < 0.6).to(recon_fake.dtype)
        recon_weight = mask * (1.0 + RECON_FOREGROUND_WEIGHT * foreground)
        recon_loss = torch.sum(torch.abs(recon_fake - real_resized) * recon_weight) / recon_weight.sum().clamp_min(1.0)

        writer_embed_loss = recon_fake.new_tensor(0.0)
        if WRITER_EMBED_MATCH_WEIGHT > 0:
            fake_feat = self._forward_module(self.netW, recon_fake, training=False)
            writer_embed_loss = F.l1_loss(fake_feat.mean(dim=1), feat_w.detach().mean(dim=1)) * WRITER_EMBED_MATCH_WEIGHT
        return recon_loss, writer_embed_loss

    @torch.no_grad()
    def _generate_reconstruction_page(self, img, wcl, labels):
        text_encode, len_text = self.netconverter.encode(labels)
        text_encode = text_encode.to(self.device)
        len_text = len_text.to(self.device)
        feat_w = self._module(self.netW)(img.detach(), wcl, training=False)
        fakes = self.netG_ema(feat_w, text_encode)

        real_rows = []
        fake_rows = []
        for batch_idx in range(img.shape[0]):
            real_w = min(img.shape[-1], int(len_text[batch_idx].item()) * resolution)
            fake_w = min(fakes.shape[-1], int(len_text[batch_idx].item()) * resolution)
            real_rows.append(((img[batch_idx, 0, :, :real_w].cpu().numpy() + 1) / 2))
            fake_rows.append(((fakes[batch_idx, 0, :, :fake_w].cpu().numpy() + 1) / 2))

        max_w = max(max(row.shape[1] for row in real_rows), max(row.shape[1] for row in fake_rows))
        gap_h = np.ones([16, max_w])

        def stack_rows(rows):
            out = []
            for row in rows:
                out.append(np.concatenate([row, np.ones([IMG_HEIGHT, max_w - row.shape[1]])], 1))
                out.append(gap_h)
            return np.concatenate(out, 0)

        return self._add_preview_labels(stack_rows(real_rows), stack_rows(fake_rows))

    def forward(self):

        self.real = self.input["img"].to(self.device)
        if USE_CHANNELS_LAST and self.real.dim() == 4:
            self.real = self.real.contiguous(memory_format=torch.channels_last)
        self.label = self.input["label"]
        self.sdata = self.input["img"].to(self.device)
        self.ST_LEN = self.input["swids"]
        self.text_encode, self.len_text = self.netconverter.encode(self.label)

        self.text_encode = self.text_encode.to(self.device).detach()
        self.len_text = self.len_text.detach()

        sample_lex_idx = self.fake_y_dist.sample([self.batch_size])
        fake_y = [self.lex[i].encode("utf-8") for i in sample_lex_idx]

        self.text_encode_fake, self.len_text_fake = self.netconverter.encode(fake_y)
        self.text_encode_fake = self.text_encode_fake.to(self.device)

    def backward_D_OCR_W(self, loss_scale=1.0):
        feat_w, self.loss_w_real = self.netW(
            self.real.detach(), self.input["wcl"].to(self.device)
        )
        _, self.pred_real_OCR = self.netOCR(self.real.detach())
        self.loss_w_real = self.loss_w_real.mean()
        with torch.no_grad():
            self.fake = self._forward_module(self.netG, feat_w.detach(), self.text_encode_fake)
        real_for_d = maybe_augment_for_discriminator(self.real.detach())
        fake_for_d = maybe_augment_for_discriminator(self.fake.detach())
        pred_real = self.netD(real_for_d)
        pred_fake = self.netD(**{"x": fake_for_d})

        self.score_Dreal = self._masked_discriminator_score(
            pred_real.detach(), self.len_text.detach()
        )
        self.score_Dfake = self._masked_discriminator_score(
            pred_fake.detach(), self.len_text_fake.detach()
        )

        self.loss_Dreal, self.loss_Dfake = loss_hinge_dis(
            pred_fake,
            pred_real,
            self.len_text_fake.detach(),
            self.len_text.detach(),
            True,
        )


        self.loss_D_consistency = pred_real.new_tensor(0.0)
        if D_CONSISTENCY_WEIGHT > 0:
            with torch.no_grad():
                pred_real_base = self._forward_module(self.netD, self.real.detach())
                pred_fake_base = self._forward_module(self.netD, x=self.fake.detach())
            self.loss_D_consistency = self._disc_consistency(
                pred_real,
                pred_real_base.detach(),
                self.len_text.detach(),
            ) + self._disc_consistency(
                pred_fake,
                pred_fake_base.detach(),
                self.len_text_fake.detach(),
            )
        self.loss_D_logit_reg = pred_real.new_tensor(0.0)
        if D_LOGIT_REG_WEIGHT > 0:
            self.loss_D_logit_reg = self._masked_discriminator_square(
                pred_real,
                self.len_text.detach(),
            ) + self._masked_discriminator_square(
                pred_fake,
                self.len_text_fake.detach(),
            )
        self.loss_D = (
            self.loss_Dreal
            + self.loss_Dfake
            + self.loss_D_consistency * D_CONSISTENCY_WEIGHT
            + self.loss_D_logit_reg * D_LOGIT_REG_WEIGHT
        )
        self.pred_real_OCR = self.pred_real_OCR.float()
        preds_size = torch.IntTensor(
            [self.pred_real_OCR.size(1)] * self.batch_size
        ).detach()
        self.pred_real_OCR = self.pred_real_OCR.permute(1, 0, 2).log_softmax(2)
        loss_OCR_real = self.OCR_criterion(
            self.pred_real_OCR,
            self.text_encode.detach(),
            preds_size,
            self.len_text.detach(),
        )
        self.loss_OCR_real = torch.mean(loss_OCR_real[~torch.isnan(loss_OCR_real)])

        loss_total = self.loss_D * ADV_LOSS_WEIGHT + self.loss_OCR_real + self.loss_w_real
        # backward
        self.backward_loss(loss_total * loss_scale)
        return loss_total

    def _masked_discriminator_score(self, pred, lengths):
        mask = torch.ones_like(pred)
        if lengths is not None and len(pred.shape) > 2:
            for idx in range(len(lengths)):
                mask[idx, :, :, int(lengths[idx]) :] = 0
        return torch.sum(pred * mask) / mask.sum().clamp_min(1.0)

    def _masked_discriminator_square(self, pred, lengths):
        mask = torch.ones_like(pred)
        if lengths is not None and len(pred.shape) > 2:
            for idx in range(len(lengths)):
                mask[idx, :, :, int(lengths[idx]) :] = 0
        return torch.sum((pred * mask) ** 2) / mask.sum().clamp_min(1.0)

    def _disc_consistency(self, pred_aug, pred_base, lengths):
        if D_CONSISTENCY_WEIGHT <= 0:
            return pred_aug.new_tensor(0.0)
        mask = torch.ones_like(pred_aug)
        if lengths is not None and len(pred_aug.shape) > 2:
            for idx in range(len(lengths)):
                mask[idx, :, :, int(lengths[idx]) :] = 0
        denom = mask.sum().clamp_min(1.0)
        return torch.sum(((pred_aug - pred_base) * mask) ** 2) / denom

    def backward_G_only(self, loss_scale=1.0):

        feat_w = self._forward_module(
            self.netW,
            self.real.detach(),
            self.input["wcl"].to(self.device),
            training=False,
        )
        self.fake = self.netG(feat_w, self.text_encode_fake)
        pred_fake = self._forward_module(self.netD, x=self.fake)
        self.loss_G = loss_hinge_gen(
            pred_fake, self.len_text_fake.detach(), True
        ).mean()

        _, pred_fake_OCR = self._forward_module(self.netOCR, self.fake)
        pred_fake_OCR = pred_fake_OCR.float()
        preds_size = torch.IntTensor([pred_fake_OCR.size(1)] * self.batch_size).detach()
        pred_fake_OCR = pred_fake_OCR.permute(1, 0, 2).log_softmax(2)
        loss_OCR_fake = self.OCR_criterion(
            pred_fake_OCR,
            self.text_encode_fake.detach(),
            preds_size,
            self.len_text_fake.detach(),
        )
        self.loss_OCR_fake = torch.mean(loss_OCR_fake[~torch.isnan(loss_OCR_fake)])
        self.loss_recon, self.loss_writer_embed = self._reconstruction_losses(feat_w)

        _, self.loss_w_fake = self._forward_module(
            self.netW,
            self.fake,
            self.input["wcl"].to(self.device),
        )
        self.loss_w_fake = self.loss_w_fake.mean()

        real_density = self._foreground_density(self.real.detach(), self.len_text.detach())
        fake_density = self._foreground_density(self.fake, self.len_text_fake.detach())
        self.loss_ink = torch.abs(fake_density - real_density)

        if USE_GRAD_BALANCE:
            grad_fake_adv = torch.autograd.grad(
                self.loss_G, self.fake, retain_graph=True, create_graph=False
            )[0]
            grad_fake_OCR = torch.autograd.grad(
                self.loss_OCR_fake, self.fake, retain_graph=True, create_graph=False
            )[0]
            grad_fake_WL = torch.autograd.grad(
                self.loss_w_fake, self.fake, retain_graph=True, create_graph=False
            )[0]
            adv_std = torch.std(grad_fake_adv.detach())
            gp_ocr = GRAD_BALANCE_ALPHA * adv_std / (
                self.epsilon + torch.std(grad_fake_OCR.detach())
            )
            gp_wl = GRAD_BALANCE_BETA * adv_std / (
                self.epsilon + torch.std(grad_fake_WL.detach())
            )
            gp_ocr = torch.clamp(gp_ocr, 0.0, GRAD_BALANCE_MAX)
            gp_wl = torch.clamp(gp_wl, 0.0, GRAD_BALANCE_MAX)
        else:
            gp_ocr = self.loss_G.new_tensor(GRAD_BALANCE_ALPHA)
            gp_wl = self.loss_G.new_tensor(GRAD_BALANCE_BETA)
        self.grad_balance_ocr = gp_ocr.detach()
        self.grad_balance_w = gp_wl.detach()
        self.loss_OCR_fake = gp_ocr.detach() * self.loss_OCR_fake
        self.loss_w_fake = gp_wl.detach() * self.loss_w_fake
        self.loss_T = (
            self.loss_G * ADV_LOSS_WEIGHT
            + self.loss_OCR_fake
            + self.loss_w_fake * WRITER_FAKE_WEIGHT
            + self.loss_ink * INK_LOSS_WEIGHT
            + self.loss_recon * RECON_LOSS_WEIGHT
            + self.loss_writer_embed
        )
        self.backward_loss(self.loss_T * loss_scale)

    def optimize_D_OCR_W(self, zero_grad=True, loss_scale=1.0):
        self.forward()
        self.set_requires_grad([self.netG], False)
        self.set_requires_grad([self.netD], True)
        self.set_requires_grad([self.netOCR], True)
        self.set_requires_grad([self.netW], True)
        if zero_grad:
            self.optimizer_D.zero_grad()
            self.optimizer_OCR.zero_grad()
            self.optimizer_wl.zero_grad()
        with self.autocast():
            self.backward_D_OCR_W(loss_scale=loss_scale)

    def optimize_D_OCR_W_step(self):

        if self.use_grad_scaler:
            self.scaler.unscale_(self.optimizer_D)
            self.scaler.unscale_(self.optimizer_wl)
            self.scaler.unscale_(self.optimizer_OCR)
        self.grad_norm_D = self._clip_grad_norm(self.netD.parameters(), D_GRAD_CLIP)
        self.grad_norm_W = self._clip_grad_norm(self.netW.parameters(), W_GRAD_CLIP)
        self.grad_norm_OCR = self._clip_grad_norm(self.netOCR.parameters(), OCR_GRAD_CLIP)
        self.step_optimizer(self.optimizer_D)
        self.step_optimizer(self.optimizer_wl)
        self.step_optimizer(self.optimizer_OCR)
        self.update_scaler()
        self.optimizer_D.zero_grad()
        self.optimizer_OCR.zero_grad()
        self.optimizer_wl.zero_grad()

    def optimize_G_only(self, zero_grad=True, loss_scale=1.0):
        self.forward()
        self.set_requires_grad([self.netG], True)
        self.set_requires_grad([self.netD], False)
        self.set_requires_grad([self.netOCR], False)
        self.set_requires_grad([self.netW], False)
        if zero_grad:
            self.optimizer_G.zero_grad()
        with self.autocast():
            self.backward_G_only(loss_scale=loss_scale)

    def optimize_G_step(self):

        if self.use_grad_scaler:
            self.scaler.unscale_(self.optimizer_G)
        self.grad_norm_G = self._clip_grad_norm(self.netG.parameters(), G_GRAD_CLIP)
        self.step_optimizer(self.optimizer_G)
        self.update_scaler()
        self.update_g_ema()
        self.optimizer_G.zero_grad()
