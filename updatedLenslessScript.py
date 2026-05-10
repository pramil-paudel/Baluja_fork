#!/usr/bin/env python3
# encoding: utf-8
"""
main_lensless_npy.py
====================

Adapter for the original deep-steganography HidingUNet / RevealNet code,
but using Pramil's dataset structure where the secret is a precomputed
lensless diffraction pattern saved as .npy.

Expected dataset structure:

DATA_ROOT/
    train_cover/class_0/
    train_secret_lensless/class_0/
    train_secret/class_0/                  # optional original RGB secret reference

    validation_cover/class_0/
    validation_secret_lensless/class_0/
    validation_secret/class_0/             # optional original RGB secret reference

    test_cover/class_0/
    test_secret_lensless/class_0/
    test_secret/class_0/                   # optional original RGB secret reference

Model behavior:
    cover image          : RGB image tensor [0,1]
    secret image         : lensless .npy tensor [0,1]
    Hnet input           : concat(cover, lensless_secret) -> 6 channels
    Hnet output          : container image [0,1]
    Rnet input           : container image
    Rnet output          : recovered lensless secret [0,1]

This script keeps the original model idea:
    Hnet = UnetGenerator(input_nc=6, output_nc=3, output_function=nn.Sigmoid)
    Rnet = RevealNet(output_function=nn.Sigmoid)

but replaces the original MyImageFolder half-batch split with an explicit
paired cover/lensless-secret dataset.
"""

import argparse
import os
import shutil
import socket
import time
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torchvision.utils as vutils
from PIL import Image
from tensorboardX import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as tvT
from torchvision.transforms import InterpolationMode

from models.HidingUNet import UnetGenerator
from models.RevealNet import RevealNet

# Same lensless reconstruction module used in the cleaned stego_run.py validation grid.
# It is used only for qualitative validation/test grids, not for training.
import lensless.lenslessThree as lenslessConverter


# ──────────────────────────────────────────────────────────
#  Defaults
# ──────────────────────────────────────────────────────────
DEFAULT_DATA_DIR = "/scratch/p522p287/DATA/STEN_DATA_LENSLESS/DIV2K_STEN/"
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


# ──────────────────────────────────────────────────────────
#  Args
# ──────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Original HidingUNet/RevealNet on lensless .npy secrets")
parser.add_argument("--data", default=DEFAULT_DATA_DIR, help="dataset root")
parser.add_argument("--workers", type=int, default=0)
parser.add_argument("--batchSize", type=int, default=4)
parser.add_argument("--imageSize", type=int, default=128, help="cover/secret tensor size")
parser.add_argument("--niter", type=int, default=100)
parser.add_argument("--lr", type=float, default=0.001)
parser.add_argument("--beta1", type=float, default=0.5)
parser.add_argument("--beta", type=float, default=0.75, help="weight for reveal loss")
parser.add_argument("--cuda", type=bool, default=True)
parser.add_argument("--ngpu", type=int, default=1)
parser.add_argument("--Hnet", default="", help="path to HidingNet checkpoint")
parser.add_argument("--Rnet", default="", help="path to RevealNet checkpoint")
parser.add_argument("--test_only", action="store_true")
parser.add_argument("--debug", action="store_true")
parser.add_argument("--remark", default="_lensless_npy")
parser.add_argument("--hostname", default=socket.gethostname())
parser.add_argument("--logFrequency", type=int, default=10)
parser.add_argument("--resultPicFrequency", type=int, default=100)
parser.add_argument("--outroot", default="./training_lensless_original_model")
parser.add_argument("--strict_load", action="store_true", help="strict checkpoint loading")


# ──────────────────────────────────────────────────────────
#  Utilities
# ──────────────────────────────────────────────────────────
class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val, n=1):
        self.val = float(val)
        self.sum += float(val) * n
        self.count += n
        self.avg = self.sum / max(self.count, 1)


def make_dirs(*paths):
    for p in paths:
        os.makedirs(p, exist_ok=True)


def print_log(log_info, log_path=None, console=True, debug=False):
    if console:
        print(log_info)
    if debug or log_path is None:
        return
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "a+", encoding="utf-8") as f:
        f.write(log_info + "\n")


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        if hasattr(m, "weight") and m.weight is not None:
            m.weight.data.normal_(0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        if hasattr(m, "weight") and m.weight is not None:
            m.weight.data.normal_(1.0, 0.02)
        if hasattr(m, "bias") and m.bias is not None:
            m.bias.data.fill_(0)


def print_network(net, log_path=None, debug=False):
    num_params = sum(p.numel() for p in net.parameters())
    print_log(str(net), log_path, debug=debug)
    print_log(f"Total number of parameters: {num_params}", log_path, debug=debug)


def list_images_recursive(root):
    root = Path(root)
    return sorted([p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS])


def list_npy_recursive(root):
    root = Path(root)
    return sorted([p for p in root.rglob("*.npy") if p.is_file()])


def tensor_to_display_grid(x):
    """Ensure tensor for save_image is [0,1]."""
    if x.min() < -0.05:
        x = x.clamp(-1, 1).add(1).mul(0.5)
    return x.clamp(0, 1)


def batch_psnr(pred, target):
    pred = pred.clamp(0, 1)
    target = target.clamp(0, 1)
    mse = (pred - target).pow(2).flatten(1).mean(1)
    psnr = -10.0 * torch.log10(mse + 1e-10)
    return psnr.mean().item()


# ──────────────────────────────────────────────────────────
#  Dataset
# ──────────────────────────────────────────────────────────
class CoverLenslessSecretDataset(Dataset):
    """
    Paired dataset:
        cover image: split_cover/class_0/*.png/jpg
        secret:      split_secret_lensless/class_0/*.npy
        optional original secret reference: split_secret/class_0/*.png/jpg

    Pairing is by sorted order, preserving class_0 folder layout.
    If your filenames match exactly, sorted order should already align.
    """

    def __init__(self, root, split, image_size=128, return_original=True):
        self.root = Path(root)
        self.split = split
        self.image_size = image_size
        self.return_original = return_original

        self.cover_root = self.root / f"{split}_cover"
        self.lensless_root = self.root / f"{split}_secret_lensless"
        self.original_root = self.root / f"{split}_secret"

        self.cover_files = list_images_recursive(self.cover_root)
        self.secret_files = list_npy_recursive(self.lensless_root)
        self.original_files = list_images_recursive(self.original_root) if self.original_root.exists() else []

        if len(self.cover_files) == 0:
            raise RuntimeError(f"No cover images found in {self.cover_root}")
        if len(self.secret_files) == 0:
            raise RuntimeError(f"No lensless .npy secrets found in {self.lensless_root}")

        self.n = min(len(self.cover_files), len(self.secret_files))
        if self.return_original and len(self.original_files) > 0:
            self.n = min(self.n, len(self.original_files))

        self.cover_files = self.cover_files[:self.n]
        self.secret_files = self.secret_files[:self.n]
        self.original_files = self.original_files[:self.n] if len(self.original_files) > 0 else []

        self.img_transform = tvT.Compose([
            tvT.Resize((image_size, image_size), interpolation=InterpolationMode.BICUBIC),
            tvT.ToTensor(),
        ])

    def __len__(self):
        return self.n

    def _load_lensless_npy(self, path):
        arr = np.load(path, allow_pickle=False)
        t = torch.from_numpy(arr).float()

        # Expected CHW. If HWC, convert to CHW.
        if t.ndim == 3 and t.shape[-1] == 3 and t.shape[0] != 3:
            t = t.permute(2, 0, 1).contiguous()

        if t.ndim != 3 or t.shape[0] != 3:
            raise ValueError(f"Expected 3-channel lensless array, got shape {tuple(t.shape)} from {path}")

        # Resize if needed.
        if t.shape[-2:] != (self.image_size, self.image_size):
            t = torch.nn.functional.interpolate(
                t.unsqueeze(0),
                size=(self.image_size, self.image_size),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)

        return t.clamp(0, 1)

    def __getitem__(self, idx):
        cover = Image.open(self.cover_files[idx]).convert("RGB")
        cover_t = self.img_transform(cover)

        secret_t = self._load_lensless_npy(self.secret_files[idx])

        if self.return_original and len(self.original_files) > 0:
            original = Image.open(self.original_files[idx]).convert("RGB")
            original_t = self.img_transform(original)
        else:
            original_t = torch.zeros_like(secret_t)

        return {
            "cover": cover_t,
            "secret": secret_t,
            "original_secret": original_t,
            "cover_path": str(self.cover_files[idx]),
            "secret_path": str(self.secret_files[idx]),
        }


# ──────────────────────────────────────────────────────────
#  Output setup
# ──────────────────────────────────────────────────────────
def setup_outputs(opt):
    cur_time = time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime())
    experiment_dir = f"{opt.hostname}_{cur_time}{opt.remark}"
    root = os.path.join(opt.outroot, experiment_dir)

    paths = {
        "root": root,
        "ckpt": os.path.join(root, "checkPoints"),
        "trainpics": os.path.join(root, "trainPics"),
        "valpics": os.path.join(root, "validationPics"),
        "testpics": os.path.join(root, "testPics"),
        "logs": os.path.join(root, "trainingLogs"),
        "codes": os.path.join(root, "codes"),
    }
    if not opt.debug:
        for p in paths.values():
            make_dirs(p)
    return paths


def save_current_code(dst_dir, debug=False):
    if debug:
        return
    try:
        os.makedirs(dst_dir, exist_ok=True)
        shutil.copyfile(os.path.realpath(__file__), os.path.join(dst_dir, os.path.basename(__file__)))
    except Exception as e:
        print(f"[WARN] Could not save current code: {e}")


# ──────────────────────────────────────────────────────────
#  Visualization
# ──────────────────────────────────────────────────────────
def save_result_pic(cover, container, secret_lensless, rev_secret, original_secret,
                    epoch, batch_i, save_path, prefix="ResultPics"):
    """
    Saves quick tensor grid rows:
        cover
        container
        original RGB secret reference
        lensless secret target
        revealed lensless secret
        abs difference
    """
    os.makedirs(save_path, exist_ok=True)

    cover = tensor_to_display_grid(cover.detach().cpu())
    container = tensor_to_display_grid(container.detach().cpu())
    original_secret = tensor_to_display_grid(original_secret.detach().cpu())
    secret_lensless = tensor_to_display_grid(secret_lensless.detach().cpu())
    rev_secret = tensor_to_display_grid(rev_secret.detach().cpu())
    diff = (rev_secret - secret_lensless).abs()

    b = cover.size(0)
    grid = torch.cat([cover, container, original_secret, secret_lensless, rev_secret, diff], dim=0)
    out_name = os.path.join(save_path, f"{prefix}_epoch{epoch:03d}_batch{batch_i:04d}.png")
    vutils.save_image(grid, out_name, nrow=b, padding=1, normalize=False)
    return out_name


def _to_grid_img(t, auto_norm=False):
    """Convert CHW tensor to HWC numpy [0,1] for matplotlib."""
    t = t.detach().cpu()
    if t.ndim == 4 and t.size(1) in (1, 3):
        t = t[0]
    if t.ndim == 3 and t.size(0) in (1, 3):
        t = t.permute(1, 2, 0)

    a = t.numpy().astype(np.float32)

    if auto_norm:
        a = (a - a.min()) / (a.max() - a.min() + 1e-8)
    elif a.min() < -0.05:
        a = a * 0.5 + 0.5

    return np.clip(a, 0, 1)


def save_val_reconstruction_grid(
    cover,
    container,
    original_secret,
    secret_lensless,
    rev_secret,
    epoch,
    batch_i,
    save_path,
    prefix="ValRecon",
    n=5,
):
    """
    Qualitative validation/test grid using the same lensless reconstruction
    module used in stego_run.py.

    Purpose:
      Show that the hidden/recovered secret is a lensless diffraction pattern,
      and that direct reconstruction from this pattern is limited/degraded.

    Rows:
      Cover
      Container
      Original Secret RGB
      Secret Lensless .npy
      Recovered Lensless
      Recon from GT Lensless
      Recon from Recovered Lensless
      Diff(Container - Cover)
    """
    os.makedirs(save_path, exist_ok=True)

    n = min(n, cover.size(0))

    cover_01 = tensor_to_display_grid(cover[:n]).detach()
    container_01 = tensor_to_display_grid(container[:n]).detach()
    original_01 = tensor_to_display_grid(original_secret[:n]).detach()
    secret_lensless_01 = tensor_to_display_grid(secret_lensless[:n]).detach()
    rev_secret_01 = tensor_to_display_grid(rev_secret[:n]).detach()
    diff_01 = (container_01 - cover_01).abs()

    device = secret_lensless.device

    try:
        recon_gt = lenslessConverter.partial_reconstruct_tensor_rev_V7_128(
            secret_lensless[:n].detach().to(device)
        ).detach().cpu().clamp(0, 1)
    except Exception as e:
        print(f"[WARN] GT lensless reconstruction failed: {e}")
        recon_gt = None

    try:
        recon_rev = lenslessConverter.partial_reconstruct_tensor_rev_V7_128(
            rev_secret[:n].detach().to(device)
        ).detach().cpu().clamp(0, 1)
    except Exception as e:
        print(f"[WARN] recovered lensless reconstruction failed: {e}")
        recon_rev = None

    rows = []
    titles = []
    auto_norms = []

    def add_row(title, tensor, auto_norm=False):
        if tensor is not None:
            rows.append(tensor)
            titles.append(title)
            auto_norms.append(auto_norm)

    add_row("Cover", cover_01, False)
    add_row("Container", container_01, False)
    add_row("Original Secret RGB", original_01, False)
    add_row("Secret Lensless (.npy)", secret_lensless_01, False)
    add_row("Recovered Lensless", rev_secret_01, False)
    add_row("Recon from GT Lensless", recon_gt, False)
    add_row("Recon from Recovered", recon_rev, False)
    add_row("Diff(Container-Cover)", diff_01, True)

    fig, axes = plt.subplots(len(rows), n, figsize=(4 * n, 3 * len(rows)), dpi=120)
    axes = np.asarray(axes).reshape(len(rows), n)

    for i, (title, stack) in enumerate(zip(titles, rows)):
        for j in range(n):
            axes[i, j].axis("off")
            axes[i, j].imshow(_to_grid_img(stack[j], auto_norm=auto_norms[i]))
            if i == 0:
                axes[i, j].set_title(f"#{j + 1}", fontsize=11)

        axes[i, 0].annotate(
            title,
            xy=(0, 0.5),
            xytext=(-38, 0),
            xycoords="axes fraction",
            textcoords="offset points",
            ha="right",
            va="center",
            fontsize=10,
            fontweight="bold",
        )

    fig.suptitle(
        "Validation Grid: lensless secret recovery and limited direct reconstruction",
        fontsize=14,
        fontweight="bold",
        y=0.995,
    )
    plt.tight_layout(pad=1.4, h_pad=1.0)

    out_name = os.path.join(save_path, f"{prefix}_epoch{epoch:03d}_batch{batch_i:04d}.png")
    plt.savefig(out_name, bbox_inches="tight")
    plt.close(fig)
    return out_name


# ──────────────────────────────────────────────────────────
#  Train / Eval
# ──────────────────────────────────────────────────────────
def train_epoch(train_loader, epoch, Hnet, Rnet, criterion, optimizerH, optimizerR,
                opt, device, writer, log_path, trainpics):
    Hnet.train()
    Rnet.train()

    Hlosses = AverageMeter()
    Rlosses = AverageMeter()
    SumLosses = AverageMeter()
    PSNRs = AverageMeter()

    start_time = time.time()

    for i, batch in enumerate(train_loader):
        cover_img = batch["cover"].to(device, non_blocking=True)
        secret_img = batch["secret"].to(device, non_blocking=True)
        original_secret = batch["original_secret"].to(device, non_blocking=True)

        Hnet.zero_grad(set_to_none=True)
        Rnet.zero_grad(set_to_none=True)

        concat_img = torch.cat([cover_img, secret_img], dim=1)

        container_img = Hnet(concat_img)
        errH = criterion(container_img, cover_img)

        rev_secret_img = Rnet(container_img)
        errR = criterion(rev_secret_img, secret_img)

        err_sum = errH + opt.beta * errR
        err_sum.backward()

        optimizerH.step()
        optimizerR.step()

        bs = cover_img.size(0)
        Hlosses.update(errH.item(), bs)
        Rlosses.update(errR.item(), bs)
        SumLosses.update(err_sum.item(), bs)
        PSNRs.update(batch_psnr(rev_secret_img, secret_img), bs)

        if i % opt.logFrequency == 0:
            elapsed = time.time() - start_time
            log = (
                f"[{epoch}/{opt.niter}][{i}/{len(train_loader)}] "
                f"Loss_H: {Hlosses.val:.6f} Loss_R: {Rlosses.val:.6f} "
                f"Loss_sum: {SumLosses.val:.6f} SecretPSNR: {PSNRs.val:.2f} dB "
                f"time: {elapsed:.2f}s"
            )
            print_log(log, log_path, debug=opt.debug)

        if i % opt.resultPicFrequency == 0:
            save_result_pic(
                cover_img,
                container_img,
                secret_img,
                rev_secret_img,
                original_secret,
                epoch,
                i,
                trainpics,
                prefix="Train",
            )

    if writer is not None:
        writer.add_scalar("train/H_loss", Hlosses.avg, epoch)
        writer.add_scalar("train/R_loss", Rlosses.avg, epoch)
        writer.add_scalar("train/sum_loss", SumLosses.avg, epoch)
        writer.add_scalar("train/secret_psnr", PSNRs.avg, epoch)

    epoch_log = (
        f"Epoch {epoch:03d} | "
        f"Hloss={Hlosses.avg:.6f} Rloss={Rlosses.avg:.6f} "
        f"Sum={SumLosses.avg:.6f} SecretPSNR={PSNRs.avg:.2f} dB"
    )
    print_log(epoch_log, log_path, debug=opt.debug)

    return Hlosses.avg, Rlosses.avg, SumLosses.avg, PSNRs.avg


@torch.no_grad()
def eval_epoch(kind, loader, epoch, Hnet, Rnet, criterion, opt, device, writer, log_path, outpics):
    Hnet.eval()
    Rnet.eval()

    Hlosses = AverageMeter()
    Rlosses = AverageMeter()
    SumLosses = AverageMeter()
    PSNRs = AverageMeter()

    for i, batch in enumerate(loader):
        cover_img = batch["cover"].to(device, non_blocking=True)
        secret_img = batch["secret"].to(device, non_blocking=True)
        original_secret = batch["original_secret"].to(device, non_blocking=True)

        concat_img = torch.cat([cover_img, secret_img], dim=1)
        container_img = Hnet(concat_img)
        rev_secret_img = Rnet(container_img)

        errH = criterion(container_img, cover_img)
        errR = criterion(rev_secret_img, secret_img)
        err_sum = errH + opt.beta * errR

        bs = cover_img.size(0)
        Hlosses.update(errH.item(), bs)
        Rlosses.update(errR.item(), bs)
        SumLosses.update(err_sum.item(), bs)
        PSNRs.update(batch_psnr(rev_secret_img, secret_img), bs)

        # Always save first batch for validation/test.
        if i == 0:
            save_result_pic(
                cover_img,
                container_img,
                secret_img,
                rev_secret_img,
                original_secret,
                epoch,
                i,
                outpics,
                prefix=kind,
            )

            save_val_reconstruction_grid(
                cover=cover_img,
                container=container_img,
                original_secret=original_secret,
                secret_lensless=secret_img,
                rev_secret=rev_secret_img,
                epoch=epoch,
                batch_i=i,
                save_path=outpics,
                prefix=f"{kind}_ReconGrid",
                n=min(5, cover_img.size(0)),
            )

    if writer is not None:
        writer.add_scalar(f"{kind}/H_loss", Hlosses.avg, epoch)
        writer.add_scalar(f"{kind}/R_loss", Rlosses.avg, epoch)
        writer.add_scalar(f"{kind}/sum_loss", SumLosses.avg, epoch)
        writer.add_scalar(f"{kind}/secret_psnr", PSNRs.avg, epoch)

    log = (
        f"[{kind}] Epoch {epoch:03d} | "
        f"Hloss={Hlosses.avg:.6f} Rloss={Rlosses.avg:.6f} "
        f"Sum={SumLosses.avg:.6f} SecretPSNR={PSNRs.avg:.2f} dB"
    )
    print_log(log, log_path, debug=opt.debug)

    return Hlosses.avg, Rlosses.avg, SumLosses.avg, PSNRs.avg


# ──────────────────────────────────────────────────────────
#  Main
# ──────────────────────────────────────────────────────────
def main():
    opt = parser.parse_args()
    cudnn.benchmark = True

    device = torch.device("cuda:0" if torch.cuda.is_available() and opt.cuda else "cpu")
    print(f"Using device: {device}")

    paths = setup_outputs(opt)
    log_path = os.path.join(paths["logs"], f"lensless_npy_bs{opt.batchSize}_log.txt")
    save_current_code(paths["codes"], debug=opt.debug)

    writer = SummaryWriter(log_dir=os.path.join(paths["root"], "runs")) if not opt.debug else None

    print_log(str(opt), log_path, debug=opt.debug)
    print_log(f"Dataset root: {opt.data}", log_path, debug=opt.debug)

    # Datasets/loaders
    train_dataset = CoverLenslessSecretDataset(opt.data, "train", image_size=opt.imageSize, return_original=True)
    val_dataset = CoverLenslessSecretDataset(opt.data, "validation", image_size=opt.imageSize, return_original=True)
    test_dataset = CoverLenslessSecretDataset(opt.data, "test", image_size=opt.imageSize, return_original=True)

    train_loader = DataLoader(
        train_dataset,
        batch_size=opt.batchSize,
        shuffle=True,
        num_workers=opt.workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=opt.batchSize,
        shuffle=False,
        num_workers=opt.workers,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=opt.batchSize,
        shuffle=False,
        num_workers=opt.workers,
        pin_memory=torch.cuda.is_available(),
    )

    print_log(f"Train samples: {len(train_dataset)}", log_path, debug=opt.debug)
    print_log(f"Val samples:   {len(val_dataset)}", log_path, debug=opt.debug)
    print_log(f"Test samples:  {len(test_dataset)}", log_path, debug=opt.debug)

    # Models from original implementation.
    Hnet = UnetGenerator(input_nc=6, output_nc=3, num_downs=7, output_function=nn.Sigmoid).to(device)
    Rnet = RevealNet(output_function=nn.Sigmoid).to(device)

    Hnet.apply(weights_init)
    Rnet.apply(weights_init)

    if opt.Hnet:
        Hnet.load_state_dict(torch.load(opt.Hnet, map_location="cpu"), strict=opt.strict_load)
        print_log(f"Loaded Hnet: {opt.Hnet}", log_path, debug=opt.debug)
    if opt.Rnet:
        Rnet.load_state_dict(torch.load(opt.Rnet, map_location="cpu"), strict=opt.strict_load)
        print_log(f"Loaded Rnet: {opt.Rnet}", log_path, debug=opt.debug)

    if opt.ngpu > 1 and torch.cuda.device_count() > 1:
        Hnet = torch.nn.DataParallel(Hnet).to(device)
        Rnet = torch.nn.DataParallel(Rnet).to(device)

    print_network(Hnet, log_path, debug=opt.debug)
    print_network(Rnet, log_path, debug=opt.debug)

    criterion = nn.MSELoss().to(device)

    optimizerH = optim.Adam(Hnet.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
    optimizerR = optim.Adam(Rnet.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
    schedulerH = ReduceLROnPlateau(optimizerH, mode="min", factor=0.2, patience=5, verbose=True)
    schedulerR = ReduceLROnPlateau(optimizerR, mode="min", factor=0.2, patience=8, verbose=True)

    if opt.test_only:
        eval_epoch("test", test_loader, 0, Hnet, Rnet, criterion, opt, device, writer, log_path, paths["testpics"])
        if writer is not None:
            writer.close()
        return

    smallest_loss = float("inf")

    print_log("Training is beginning...", log_path, debug=opt.debug)
    for epoch in range(opt.niter):
        train_epoch(
            train_loader,
            epoch,
            Hnet,
            Rnet,
            criterion,
            optimizerH,
            optimizerR,
            opt,
            device,
            writer,
            log_path,
            paths["trainpics"],
        )

        val_hloss, val_rloss, val_sumloss, val_psnr = eval_epoch(
            "validation",
            val_loader,
            epoch,
            Hnet,
            Rnet,
            criterion,
            opt,
            device,
            writer,
            log_path,
            paths["valpics"],
        )

        schedulerH.step(val_sumloss)
        schedulerR.step(val_rloss)

        if val_sumloss < smallest_loss:
            smallest_loss = val_sumloss
            h_path = os.path.join(paths["ckpt"], f"netH_epoch_{epoch:03d}_sumloss={val_sumloss:.6f}_Hloss={val_hloss:.6f}.pth")
            r_path = os.path.join(paths["ckpt"], f"netR_epoch_{epoch:03d}_sumloss={val_sumloss:.6f}_Rloss={val_rloss:.6f}.pth")
            torch.save(Hnet.state_dict(), h_path)
            torch.save(Rnet.state_dict(), r_path)
            print_log(f"Saved best checkpoints:\n  {h_path}\n  {r_path}", log_path, debug=opt.debug)

    eval_epoch("test", test_loader, opt.niter, Hnet, Rnet, criterion, opt, device, writer, log_path, paths["testpics"])

    if writer is not None:
        writer.close()

    print_log("Training complete.", log_path, debug=opt.debug)


if __name__ == "__main__":
    main()