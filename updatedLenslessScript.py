#!/usr/bin/env python3
# encoding: utf-8
"""
main_lensless_npy.py
====================

Adapter for the original deep-steganography HidingUNet / RevealNet code,
with two secret modes:

1) --secret_mode lensless
   Secret is a precomputed lensless diffraction pattern saved as .npy.

2) --secret_mode rgb
   Secret is the normal RGB image from *_secret/class_0/.

Expected dataset structure:

DATA_ROOT/
    train_cover/class_0/
    train_secret_lensless/class_0/
    train_secret/class_0/

    validation_cover/class_0/
    validation_secret_lensless/class_0/
    validation_secret/class_0/

    test_cover/class_0/
    test_secret_lensless/class_0/
    test_secret/class_0/

Model behavior:
    cover image  : RGB image tensor [0,1]
    secret image : either RGB image [0,1] or lensless .npy tensor [0,1]
    Hnet input   : concat(cover, secret) -> 6 channels
    Hnet output  : container image [0,1]
    Rnet input   : container image
    Rnet output  : recovered secret [0,1]

Purpose:
    Run original Baluja-style HidingUNet/RevealNet on both normal RGB secrets
    and lensless diffraction-pattern secrets. Compare container PSNR and secret
    PSNR/SSIM curves to show why the original architecture is insufficient for
    lensless/diffraction secrets and why a lensless-aware architecture is needed.
"""

import argparse
import csv
import os
import shutil
import socket
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
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

try:
    from pytorch_msssim import ssim as msssim_ssim
    SSIM_BACKEND = "pytorch_msssim"
except Exception:
    msssim_ssim = None
    SSIM_BACKEND = "simple"


# ──────────────────────────────────────────────────────────
#  Defaults
# ──────────────────────────────────────────────────────────
DEFAULT_DATA_DIR = "/scratch/p522p287/DATA/STEN_DATA_LENSLESS/DIV2K_STEN/"
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


# ──────────────────────────────────────────────────────────
#  Args
# ──────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Original HidingUNet/RevealNet on RGB or lensless .npy secrets")
parser.add_argument("--data", default=DEFAULT_DATA_DIR, help="dataset root")
parser.add_argument("--secret_mode", default="lensless", choices=["lensless", "rgb"],
                    help="lensless = use *_secret_lensless/*.npy as secret; rgb = use *_secret images as secret")
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
parser.add_argument("--remark", default="", help="extra suffix. If empty, auto uses secret_mode")
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
    """Ensure tensor for save_image/matplotlib is [0,1]."""
    if x is None:
        return None
    if x.min() < -0.05:
        x = x.clamp(-1, 1).add(1).mul(0.5)
    return x.clamp(0, 1)


def batch_psnr(pred, target):
    pred = pred.clamp(0, 1)
    target = target.clamp(0, 1)
    mse = (pred - target).pow(2).flatten(1).mean(1)
    psnr = -10.0 * torch.log10(mse + 1e-10)
    return psnr.mean().item()


def batch_ssim(pred, target):
    pred = pred.clamp(0, 1)
    target = target.clamp(0, 1)

    if msssim_ssim is not None:
        try:
            return float(msssim_ssim(pred, target, data_range=1.0, size_average=True))
        except Exception:
            pass

    # Lightweight fallback: not a full MS-SSIM implementation, but stable for trend curves.
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    mu_x = pred.mean(dim=(2, 3), keepdim=True)
    mu_y = target.mean(dim=(2, 3), keepdim=True)
    sigma_x = ((pred - mu_x) ** 2).mean(dim=(2, 3), keepdim=True)
    sigma_y = ((target - mu_y) ** 2).mean(dim=(2, 3), keepdim=True)
    sigma_xy = ((pred - mu_x) * (target - mu_y)).mean(dim=(2, 3), keepdim=True)
    ssim = ((2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)) / (
        (mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2) + 1e-10
    )
    return float(ssim.mean())


# ──────────────────────────────────────────────────────────
#  Dataset
# ──────────────────────────────────────────────────────────
class CoverSecretDataset(Dataset):
    """
    Paired dataset supporting two secret modes:

    secret_mode='lensless':
        cover:  split_cover/class_0/*.png/jpg
        secret: split_secret_lensless/class_0/*.npy
        reference original RGB secret: split_secret/class_0/*.png/jpg

    secret_mode='rgb':
        cover:  split_cover/class_0/*.png/jpg
        secret: split_secret/class_0/*.png/jpg
        reference original RGB secret: same as secret
    """

    def __init__(self, root, split, image_size=128, secret_mode="lensless", return_original=True):
        self.root = Path(root)
        self.split = split
        self.image_size = image_size
        self.secret_mode = secret_mode
        self.return_original = return_original

        self.cover_root = self.root / f"{split}_cover"
        self.lensless_root = self.root / f"{split}_secret_lensless"
        self.original_root = self.root / f"{split}_secret"

        self.cover_files = list_images_recursive(self.cover_root)
        self.original_files = list_images_recursive(self.original_root) if self.original_root.exists() else []

        if self.secret_mode == "lensless":
            self.secret_files = list_npy_recursive(self.lensless_root)
        elif self.secret_mode == "rgb":
            self.secret_files = list_images_recursive(self.original_root)
        else:
            raise ValueError(f"Unknown secret_mode: {self.secret_mode}")

        if len(self.cover_files) == 0:
            raise RuntimeError(f"No cover images found in {self.cover_root}")
        if len(self.secret_files) == 0:
            raise RuntimeError(f"No secret files found for mode={self.secret_mode}")

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

    def _load_image(self, path):
        img = Image.open(path).convert("RGB")
        return self.img_transform(img)

    def _load_npy(self, path):
        arr = np.load(path, allow_pickle=False)
        t = torch.from_numpy(arr).float()

        # Supports HWC RGB or CHW RGB. Also supports single-channel by repeating to 3 channels.
        if t.ndim == 2:
            t = t.unsqueeze(0).repeat(3, 1, 1)
        elif t.ndim == 3 and t.shape[0] == 1:
            t = t.repeat(3, 1, 1)
        elif t.ndim == 3 and t.shape[-1] == 1 and t.shape[0] != 1:
            t = t.permute(2, 0, 1).repeat(3, 1, 1).contiguous()
        elif t.ndim == 3 and t.shape[-1] == 3 and t.shape[0] != 3:
            t = t.permute(2, 0, 1).contiguous()

        if t.ndim != 3 or t.shape[0] != 3:
            raise ValueError(f"Expected 1- or 3-channel npy, got shape {tuple(t.shape)} from {path}")

        if t.shape[-2:] != (self.image_size, self.image_size):
            t = torch.nn.functional.interpolate(
                t.unsqueeze(0),
                size=(self.image_size, self.image_size),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)

        return t.clamp(0, 1)

    def __getitem__(self, idx):
        cover_t = self._load_image(self.cover_files[idx])

        if self.secret_mode == "lensless":
            secret_t = self._load_npy(self.secret_files[idx])
        else:
            secret_t = self._load_image(self.secret_files[idx])

        if self.return_original and len(self.original_files) > 0:
            original_t = self._load_image(self.original_files[idx])
        else:
            original_t = secret_t.clone()

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
    if opt.remark == "":
        opt.remark = f"_{opt.secret_mode}_secret"

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
        "plots": os.path.join(root, "plots"),
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
#  History / plots
# ──────────────────────────────────────────────────────────
class MetricHistory:
    def __init__(self):
        self.rows = []

    def append(self, epoch, split, h_loss, r_loss, sum_loss, cover_psnr, cover_ssim, secret_psnr, secret_ssim):
        self.rows.append({
            "epoch": int(epoch),
            "split": split,
            "h_loss": float(h_loss),
            "r_loss": float(r_loss),
            "sum_loss": float(sum_loss),
            "cover_psnr": float(cover_psnr),
            "cover_ssim": float(cover_ssim),
            "secret_psnr": float(secret_psnr),
            "secret_ssim": float(secret_ssim),
        })

    def write_csv(self, path):
        if not self.rows:
            return
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(self.rows[0].keys()))
            writer.writeheader()
            writer.writerows(self.rows)


def plot_metric_history(history, save_dir, secret_mode):
    os.makedirs(save_dir, exist_ok=True)
    csv_path = os.path.join(save_dir, f"metrics_{secret_mode}.csv")
    png_path = os.path.join(save_dir, f"psnr_ssim_curves_{secret_mode}.png")
    history.write_csv(csv_path)

    rows = [r for r in history.rows if r["split"] == "validation"]
    if len(rows) == 0:
        return csv_path, None

    epochs = [r["epoch"] for r in rows]
    cover_psnr = [r["cover_psnr"] for r in rows]
    secret_psnr = [r["secret_psnr"] for r in rows]
    cover_ssim = [r["cover_ssim"] for r in rows]
    secret_ssim = [r["secret_ssim"] for r in rows]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), dpi=130)
    fig.suptitle(f"Original HidingUNet/RevealNet — {secret_mode.upper()} secret", fontsize=14, fontweight="bold")

    axes[0].plot(epochs, cover_psnr, marker="o", label="Container PSNR: container vs cover")
    axes[0].plot(epochs, secret_psnr, marker="o", label="Secret PSNR: recovered secret vs target secret")
    axes[0].set_title("PSNR Curves")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("PSNR (dB)")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(epochs, cover_ssim, marker="o", label="Container SSIM")
    axes[1].plot(epochs, secret_ssim, marker="o", label="Secret SSIM")
    axes[1].set_title("SSIM Curves")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("SSIM")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(png_path, bbox_inches="tight")
    plt.close(fig)
    return csv_path, png_path


# ──────────────────────────────────────────────────────────
#  Visualization
# ──────────────────────────────────────────────────────────
def save_result_pic(cover, container, secret, rev_secret, original_secret,
                    epoch, batch_i, save_path, prefix="ResultPics"):
    """
    Saves quick tensor grid rows:
        cover
        container
        original RGB secret reference
        target secret used for training
        revealed/recovered secret
        abs difference
    """
    os.makedirs(save_path, exist_ok=True)

    cover = tensor_to_display_grid(cover.detach().cpu())
    container = tensor_to_display_grid(container.detach().cpu())
    original_secret = tensor_to_display_grid(original_secret.detach().cpu())
    secret = tensor_to_display_grid(secret.detach().cpu())
    rev_secret = tensor_to_display_grid(rev_secret.detach().cpu())
    diff = (rev_secret - secret).abs()

    b = cover.size(0)
    grid = torch.cat([cover, container, original_secret, secret, rev_secret, diff], dim=0)
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


def _metric_text_rows(cover, container, secret, rev_secret, recon_gt=None, recon_rev=None):
    lines = []
    lines.append(f"Container\nPSNR {batch_psnr(container, cover):.2f} dB\nSSIM {batch_ssim(container, cover):.3f}")
    lines.append(f"Secret recovery\nPSNR {batch_psnr(rev_secret, secret):.2f} dB\nSSIM {batch_ssim(rev_secret, secret):.3f}")
    if recon_gt is not None and recon_rev is not None:
        lines.append(f"Recon recovery\nPSNR {batch_psnr(recon_rev, recon_gt):.2f} dB\nSSIM {batch_ssim(recon_rev, recon_gt):.3f}")
    return "\n\n".join(lines)


def save_val_reconstruction_grid(
    cover,
    container,
    original_secret,
    secret,
    rev_secret,
    epoch,
    batch_i,
    save_path,
    secret_mode="lensless",
    prefix="ValRecon",
    n=5,
):
    """
    Validation/test grid with PSNR/SSIM text.

    For lensless mode, also shows reconstruction rows using lenslessConverter:
      Recon from GT Lensless
      Recon from Recovered Lensless

    For rgb mode, reconstruction rows are skipped because the secret is already a normal RGB image.
    """
    os.makedirs(save_path, exist_ok=True)

    n = min(n, cover.size(0))

    cover_01 = tensor_to_display_grid(cover[:n]).detach()
    container_01 = tensor_to_display_grid(container[:n]).detach()
    original_01 = tensor_to_display_grid(original_secret[:n]).detach()
    secret_01 = tensor_to_display_grid(secret[:n]).detach()
    rev_secret_01 = tensor_to_display_grid(rev_secret[:n]).detach()
    diff_01 = (container_01 - cover_01).abs()

    recon_gt = None
    recon_rev = None

    if secret_mode == "lensless":
        device = secret.device
        try:
            recon_gt = lenslessConverter.partial_reconstruct_tensor_rev_V7_128(
                secret[:n].detach().to(device)
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
    add_row("Target Secret" if secret_mode == "rgb" else "Secret Lensless (.npy)", secret_01, False)
    add_row("Recovered Secret" if secret_mode == "rgb" else "Recovered Lensless", rev_secret_01, False)
    add_row("Recon from GT Lensless", recon_gt, False)
    add_row("Recon from Recovered", recon_rev, False)
    add_row("Diff(Container-Cover)", diff_01, True)

    metric_text = _metric_text_rows(
        cover_01,
        container_01,
        secret_01,
        rev_secret_01,
        recon_gt=recon_gt,
        recon_rev=recon_rev,
    )

    rows_for_fig = len(rows) + 1
    fig, axes = plt.subplots(rows_for_fig, n, figsize=(4 * n, 3 * rows_for_fig), dpi=120)
    axes = np.asarray(axes).reshape(rows_for_fig, n)

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

    # Metric row.
    metric_row_idx = rows_for_fig - 1
    for j in range(n):
        axes[metric_row_idx, j].axis("off")
        if j == 0:
            axes[metric_row_idx, j].text(
                0.02,
                0.95,
                metric_text,
                va="top",
                ha="left",
                fontsize=10,
                family="monospace",
                bbox=dict(boxstyle="round,pad=0.35", facecolor="#ffffff", edgecolor="#cccccc"),
            )
    axes[metric_row_idx, 0].annotate(
        "PSNR / SSIM",
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
        f"Validation Grid — original HidingUNet/RevealNet using {secret_mode.upper()} secret",
        fontsize=14,
        fontweight="bold",
        y=0.995,
    )
    plt.tight_layout(pad=1.4, h_pad=1.0)

    out_name = os.path.join(save_path, f"{prefix}_{secret_mode}_epoch{epoch:03d}_batch{batch_i:04d}.png")
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
    CoverPSNRs = AverageMeter()
    CoverSSIMs = AverageMeter()
    SecretPSNRs = AverageMeter()
    SecretSSIMs = AverageMeter()

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
        CoverPSNRs.update(batch_psnr(container_img, cover_img), bs)
        CoverSSIMs.update(batch_ssim(container_img, cover_img), bs)
        SecretPSNRs.update(batch_psnr(rev_secret_img, secret_img), bs)
        SecretSSIMs.update(batch_ssim(rev_secret_img, secret_img), bs)

        if i % opt.logFrequency == 0:
            elapsed = time.time() - start_time
            log = (
                f"[{epoch}/{opt.niter}][{i}/{len(train_loader)}] "
                f"Loss_H: {Hlosses.val:.6f} Loss_R: {Rlosses.val:.6f} "
                f"Loss_sum: {SumLosses.val:.6f} "
                f"CovPSNR: {CoverPSNRs.val:.2f} SecPSNR: {SecretPSNRs.val:.2f} "
                f"SecSSIM: {SecretSSIMs.val:.3f} time: {elapsed:.2f}s"
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
                prefix=f"Train_{opt.secret_mode}",
            )

    if writer is not None:
        writer.add_scalar("train/H_loss", Hlosses.avg, epoch)
        writer.add_scalar("train/R_loss", Rlosses.avg, epoch)
        writer.add_scalar("train/sum_loss", SumLosses.avg, epoch)
        writer.add_scalar("train/cover_psnr", CoverPSNRs.avg, epoch)
        writer.add_scalar("train/cover_ssim", CoverSSIMs.avg, epoch)
        writer.add_scalar("train/secret_psnr", SecretPSNRs.avg, epoch)
        writer.add_scalar("train/secret_ssim", SecretSSIMs.avg, epoch)

    epoch_log = (
        f"Epoch {epoch:03d} | "
        f"Hloss={Hlosses.avg:.6f} Rloss={Rlosses.avg:.6f} Sum={SumLosses.avg:.6f} | "
        f"CoverPSNR={CoverPSNRs.avg:.2f} CoverSSIM={CoverSSIMs.avg:.3f} | "
        f"SecretPSNR={SecretPSNRs.avg:.2f} SecretSSIM={SecretSSIMs.avg:.3f}"
    )
    print_log(epoch_log, log_path, debug=opt.debug)

    return {
        "h_loss": Hlosses.avg,
        "r_loss": Rlosses.avg,
        "sum_loss": SumLosses.avg,
        "cover_psnr": CoverPSNRs.avg,
        "cover_ssim": CoverSSIMs.avg,
        "secret_psnr": SecretPSNRs.avg,
        "secret_ssim": SecretSSIMs.avg,
    }


@torch.no_grad()
def eval_epoch(kind, loader, epoch, Hnet, Rnet, criterion, opt, device, writer, log_path, outpics):
    Hnet.eval()
    Rnet.eval()

    Hlosses = AverageMeter()
    Rlosses = AverageMeter()
    SumLosses = AverageMeter()
    CoverPSNRs = AverageMeter()
    CoverSSIMs = AverageMeter()
    SecretPSNRs = AverageMeter()
    SecretSSIMs = AverageMeter()

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
        CoverPSNRs.update(batch_psnr(container_img, cover_img), bs)
        CoverSSIMs.update(batch_ssim(container_img, cover_img), bs)
        SecretPSNRs.update(batch_psnr(rev_secret_img, secret_img), bs)
        SecretSSIMs.update(batch_ssim(rev_secret_img, secret_img), bs)

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
                prefix=f"{kind}_{opt.secret_mode}",
            )

            save_val_reconstruction_grid(
                cover=cover_img,
                container=container_img,
                original_secret=original_secret,
                secret=secret_img,
                rev_secret=rev_secret_img,
                epoch=epoch,
                batch_i=i,
                save_path=outpics,
                secret_mode=opt.secret_mode,
                prefix=f"{kind}_MetricGrid",
                n=min(5, cover_img.size(0)),
            )

    if writer is not None:
        writer.add_scalar(f"{kind}/H_loss", Hlosses.avg, epoch)
        writer.add_scalar(f"{kind}/R_loss", Rlosses.avg, epoch)
        writer.add_scalar(f"{kind}/sum_loss", SumLosses.avg, epoch)
        writer.add_scalar(f"{kind}/cover_psnr", CoverPSNRs.avg, epoch)
        writer.add_scalar(f"{kind}/cover_ssim", CoverSSIMs.avg, epoch)
        writer.add_scalar(f"{kind}/secret_psnr", SecretPSNRs.avg, epoch)
        writer.add_scalar(f"{kind}/secret_ssim", SecretSSIMs.avg, epoch)

    log = (
        f"[{kind}] Epoch {epoch:03d} | "
        f"Hloss={Hlosses.avg:.6f} Rloss={Rlosses.avg:.6f} Sum={SumLosses.avg:.6f} | "
        f"CoverPSNR={CoverPSNRs.avg:.2f} CoverSSIM={CoverSSIMs.avg:.3f} | "
        f"SecretPSNR={SecretPSNRs.avg:.2f} SecretSSIM={SecretSSIMs.avg:.3f}"
    )
    print_log(log, log_path, debug=opt.debug)

    return {
        "h_loss": Hlosses.avg,
        "r_loss": Rlosses.avg,
        "sum_loss": SumLosses.avg,
        "cover_psnr": CoverPSNRs.avg,
        "cover_ssim": CoverSSIMs.avg,
        "secret_psnr": SecretPSNRs.avg,
        "secret_ssim": SecretSSIMs.avg,
    }


# ──────────────────────────────────────────────────────────
#  Main
# ──────────────────────────────────────────────────────────
def main():
    opt = parser.parse_args()
    cudnn.benchmark = True

    device = torch.device("cuda:0" if torch.cuda.is_available() and opt.cuda else "cpu")
    print(f"Using device: {device}")
    print(f"Secret mode: {opt.secret_mode}")
    print(f"SSIM backend: {SSIM_BACKEND}")

    paths = setup_outputs(opt)
    log_path = os.path.join(paths["logs"], f"{opt.secret_mode}_bs{opt.batchSize}_log.txt")
    save_current_code(paths["codes"], debug=opt.debug)

    writer = SummaryWriter(log_dir=os.path.join(paths["root"], "runs")) if not opt.debug else None
    history = MetricHistory()

    print_log(str(opt), log_path, debug=opt.debug)
    print_log(f"Dataset root: {opt.data}", log_path, debug=opt.debug)

    # Datasets/loaders
    train_dataset = CoverSecretDataset(opt.data, "train", image_size=opt.imageSize,
                                       secret_mode=opt.secret_mode, return_original=True)
    val_dataset = CoverSecretDataset(opt.data, "validation", image_size=opt.imageSize,
                                     secret_mode=opt.secret_mode, return_original=True)
    test_dataset = CoverSecretDataset(opt.data, "test", image_size=opt.imageSize,
                                      secret_mode=opt.secret_mode, return_original=True)

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
        test_metrics = eval_epoch("test", test_loader, 0, Hnet, Rnet, criterion, opt, device, writer, log_path, paths["testpics"])
        history.append(0, "test", **test_metrics)
        csv_path, png_path = plot_metric_history(history, paths["plots"], opt.secret_mode)
        print_log(f"Metrics CSV: {csv_path}", log_path, debug=opt.debug)
        if png_path:
            print_log(f"Metrics plot: {png_path}", log_path, debug=opt.debug)
        if writer is not None:
            writer.close()
        return

    smallest_loss = float("inf")

    print_log("Training is beginning...", log_path, debug=opt.debug)
    for epoch in range(opt.niter):
        train_metrics = train_epoch(
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
        history.append(epoch, "train", **train_metrics)

        val_metrics = eval_epoch(
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
        history.append(epoch, "validation", **val_metrics)

        schedulerH.step(val_metrics["sum_loss"])
        schedulerR.step(val_metrics["r_loss"])

        if val_metrics["sum_loss"] < smallest_loss:
            smallest_loss = val_metrics["sum_loss"]
            h_path = os.path.join(
                paths["ckpt"],
                f"netH_{opt.secret_mode}_epoch_{epoch:03d}_sumloss={val_metrics['sum_loss']:.6f}_Hloss={val_metrics['h_loss']:.6f}.pth",
            )
            r_path = os.path.join(
                paths["ckpt"],
                f"netR_{opt.secret_mode}_epoch_{epoch:03d}_sumloss={val_metrics['sum_loss']:.6f}_Rloss={val_metrics['r_loss']:.6f}.pth",
            )
            torch.save(Hnet.state_dict(), h_path)
            torch.save(Rnet.state_dict(), r_path)
            print_log(f"Saved best checkpoints:\n  {h_path}\n  {r_path}", log_path, debug=opt.debug)

        csv_path, png_path = plot_metric_history(history, paths["plots"], opt.secret_mode)
        if png_path:
            print_log(f"Curves saved: {png_path}", log_path, console=((epoch + 1) % 10 == 0), debug=opt.debug)

    test_metrics = eval_epoch("test", test_loader, opt.niter, Hnet, Rnet, criterion, opt, device, writer, log_path, paths["testpics"])
    history.append(opt.niter, "test", **test_metrics)
    csv_path, png_path = plot_metric_history(history, paths["plots"], opt.secret_mode)
    print_log(f"Final metrics CSV: {csv_path}", log_path, debug=opt.debug)
    if png_path:
        print_log(f"Final metrics plot: {png_path}", log_path, debug=opt.debug)

    if writer is not None:
        writer.close()

    print_log("Training complete.", log_path, debug=opt.debug)


if __name__ == "__main__":
    main()