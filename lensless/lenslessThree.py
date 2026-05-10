#!/usr/bin/env python3
"""
LenslessConverterThree.py
=========================
Drop-in replacement for lenslessConverterTwo.py with adaptive Wiener
regularization that works correctly at ANY unitSensor value.

Key fix: The reconstruction uses adaptive lambda based on PSF power spectrum
instead of a fixed sqrt(std), which was over/under-regularized depending on
unitSensor.

Scene: 128×128 RGB
Sensor (full lensless measurement): 256×256 RGB
Network I/O: central 128×128 crop from the 256×256 lensless measurement.

Public API (same as lenslessConverterTwo):
  - convert_image_to_lensless_full(pil_image)  -> (256,256,3) uint8
  - convert_into_lensless(tensor)              -> (B,3,128,128) float [0,1]
  - convert_into_lensless_center(tensor)       -> (B,3,128,128) float [0,1]
  - reconstruct_an_image(image)                -> (128,128,3) uint8
  - partial_reconstruct_tensor_rev_V7_128(...)  -> (B,3,128,128) float [0,1]
"""

import numpy as np
import torch
from PIL import Image
from scipy.signal import max_len_seq


# ═══════════════════════════════════════════════════════════
#  Camera Parameters
# ═══════════════════════════════════════════════════════════
numScenePix = 128
numSensorPix = 256
numMaskPix = 127
unitMask = 2
unitSensor = 20
d = 40000
std = 0.02

# Adaptive Wiener regularization parameter.
# Controls reconstruction quality: higher = smoother but blurrier,
# lower = sharper but noisier. Range: 0.0005 - 0.01
WIENER_FRACTION = 0.003

z2a = lambda z: (1 - d / z)
a2z = lambda a: d / (1 - a)
depthList = z2a(np.array([50e4], dtype=np.float64))


# ═══════════════════════════════════════════════════════════
#  Grid Locations
# ═══════════════════════════════════════════════════════════
def make_locations():
    locSensor = np.linspace(
        -numSensorPix * unitSensor / 2 + unitSensor / 2,
        numSensorPix * unitSensor / 2 - unitSensor / 2,
        numSensorPix, dtype=np.float64,
    )
    locMask = np.linspace(
        -numMaskPix * unitMask / 2 + unitMask / 2,
        numMaskPix * unitMask / 2 - unitMask / 2,
        numMaskPix, dtype=np.float64,
    )
    return locSensor, locMask


locSensor, locMask = make_locations()


# ═══════════════════════════════════════════════════════════
#  Center Crop / Paste Helpers
# ═══════════════════════════════════════════════════════════
def center_crop_hwc(img_hwc, crop_hw):
    H, W = img_hwc.shape[:2]
    assert H >= crop_hw and W >= crop_hw
    ys = (H - crop_hw) // 2
    xs = (W - crop_hw) // 2
    return img_hwc[ys:ys + crop_hw, xs:xs + crop_hw, :]


def paste_center_hwc(dst_hwc, patch_hwc):
    H, W = dst_hwc.shape[:2]
    h, w = patch_hwc.shape[:2]
    ys = (H - h) // 2
    xs = (W - w) // 2
    dst_hwc[ys:ys + h, xs:xs + w, :] = patch_hwc
    return dst_hwc


def paste_center_hwc_feather(dst_u8, patch_u8, feather=16):
    H, W, _ = dst_u8.shape
    h, w, _ = patch_u8.shape
    ys = (H - h) // 2
    xs = (W - w) // 2

    out = dst_u8.astype(np.float32).copy()
    patch = patch_u8.astype(np.float32)

    yy = np.linspace(0, 1, h, dtype=np.float32)
    xx = np.linspace(0, 1, w, dtype=np.float32)
    Y, X = np.meshgrid(yy, xx, indexing="ij")

    d_edge = np.minimum.reduce([Y, 1 - Y, X, 1 - X]) * min(h, w)
    m = np.clip(d_edge / max(feather, 1), 0.0, 1.0)
    m = m * m * (3 - 2 * m)  # smoothstep
    m = m[..., None]

    region = out[ys:ys + h, xs:xs + w, :]
    out[ys:ys + h, xs:xs + w, :] = m * patch + (1 - m) * region
    return np.clip(out, 0, 255).astype(np.uint8)


def make_center_cropper(target_hw, full_hw):
    start = (full_hw - target_hw) // 2
    end = start + target_hw

    def _crop(x):
        return x[:, start:end, start:end]

    return _crop


# ═══════════════════════════════════════════════════════════
#  PSF / Mask Generation
# ═══════════════════════════════════════════════════════════
def generate_interp_matrix(depth_list, loc_sensor, loc_mask,
                           unit_mask, num_sensor_pix, num_mask_pix):
    num_depth = depth_list.size
    interp = np.zeros((num_sensor_pix, num_mask_pix, num_depth), dtype=np.float64)

    for di in range(num_depth):
        loc_interp = depth_list[di] * loc_sensor
        hi = np.minimum(len(loc_mask) - 1,
                        np.searchsorted(loc_mask, loc_interp, side="right"))
        lo = np.maximum(0, hi - 1)
        rows = np.arange(num_sensor_pix)

        w_lo = 1.0 - (loc_interp - loc_mask[lo]) / unit_mask
        w_hi = 1.0 - (loc_mask[hi] - loc_interp) / unit_mask
        w_lo = np.clip(w_lo, 0.0, 1.0)
        w_hi = np.clip(w_hi, 0.0, 1.0)

        interp[rows, lo, di] = w_lo
        interp[rows, hi, di] = w_hi

        oob = np.logical_or(loc_interp < loc_mask[0],
                            loc_interp > loc_mask[-1])
        interp[oob, :, di] = 0.0

    return interp


def create_mask_for_lensless():
    m = int(np.log2(numMaskPix + 1))
    mask_vec = max_len_seq(m)[0].reshape(numMaskPix, 1).astype(np.float64)
    mask_pattern = mask_vec @ mask_vec.T

    interp3 = generate_interp_matrix(
        depthList, locSensor, locMask, unitMask, numSensorPix, numMaskPix,
    )
    interp = interp3[:, :, 0]
    psf = interp @ mask_pattern @ interp.T
    return psf.astype(np.float64)


# Cache PSF and its FFT (computed once on first use)
_psf_cache = {}


def _get_psf_fft():
    """Compute and cache PSF FFT for reuse across calls."""
    if "psf_fft" not in _psf_cache:
        psf = create_mask_for_lensless()
        full_len = numSensorPix + numScenePix - 1
        psf_fft = np.fft.fft2(psf, s=(full_len, full_len))
        _psf_cache["psf"] = psf
        _psf_cache["psf_fft"] = psf_fft
        _psf_cache["full_len"] = full_len

        # Precompute adaptive Wiener denominator
        psf_power = np.abs(psf_fft) ** 2
        lambda_adaptive = WIENER_FRACTION * psf_power.max()
        _psf_cache["wiener_denom"] = psf_power + lambda_adaptive
        _psf_cache["lambda"] = lambda_adaptive
    return _psf_cache


# ═══════════════════════════════════════════════════════════
#  Forward: Scene → Lensless (full sensor)
# ═══════════════════════════════════════════════════════════
def convert_image_to_lensless_full(pil_image):
    """
    Scene PIL image → full 256×256 lensless measurement (uint8 HWC).
    """
    img = pil_image.convert("RGB").resize(
        (numScenePix, numScenePix), resample=Image.BICUBIC,
    )
    img = np.asarray(img, dtype=np.float32) / 255.0
    img = img.transpose(2, 0, 1).astype(np.float64)

    cache = _get_psf_fft()
    psf_fft = cache["psf_fft"]
    full_len = cache["full_len"]
    crop_sensor = make_center_cropper(numSensorPix, full_len)

    y = crop_sensor(
        np.fft.ifft2(
            psf_fft * np.fft.fft2(img, s=(full_len, full_len), axes=(1, 2)),
            axes=(1, 2),
        ).real
    )

    eps = 1e-8
    y = (y - y.min()) / max(y.max() - y.min(), eps)
    return (y * 255.0).clip(0, 255).astype(np.uint8).transpose(1, 2, 0)


# ═══════════════════════════════════════════════════════════
#  Forward: Batch tensor → Lensless center crop
# ═══════════════════════════════════════════════════════════
def convert_into_lensless(input_tensor, center_size=(128, 128)):
    """
    (B,3,128,128) tensor [0,1] or [-1,1] → (B,3,128,128) lensless center crop [0,1].
    """
    image_array = []
    with torch.no_grad():
        for img in input_tensor.detach().cpu():
            if img.min().item() < -0.05:
                img01 = img.clamp(-1.0, 1.0) * 0.5 + 0.5
            else:
                img01 = img.clamp(0.0, 1.0)

            img_np = (img01.numpy().transpose(1, 2, 0) * 255.0).astype(np.uint8)
            pil_img = Image.fromarray(img_np, "RGB")

            lensless_u8 = convert_image_to_lensless_full(pil_img)
            lensless_f32 = lensless_u8.astype(np.float32) / 255.0

            H, W, _ = lensless_f32.shape
            ch, cw = center_size
            sh, sw = (H - ch) // 2, (W - cw) // 2
            center = lensless_f32[sh:sh + ch, sw:sw + cw, :]
            image_array.append(center.transpose(2, 0, 1))

    return torch.from_numpy(np.stack(image_array)).float().to(input_tensor.device)


def convert_into_lensless_center(input_tensor, center_hw=128):
    """
    (B,3,128,128) tensor [0,1] → (B,3,128,128) center crop of lensless [0,1].
    """
    assert input_tensor.dim() == 4 and input_tensor.size(1) == 3
    outs = []
    with torch.no_grad():
        x = input_tensor.detach().cpu().float().clamp(0.0, 1.0).numpy()
        for i in range(x.shape[0]):
            rgb = (x[i].transpose(1, 2, 0) * 255.0).round().astype(np.uint8)
            y_full = convert_image_to_lensless_full(Image.fromarray(rgb, "RGB"))
            y_center = center_crop_hwc(y_full, center_hw)
            y_t = torch.from_numpy(y_center).permute(2, 0, 1).float() / 255.0
            outs.append(y_t)
    return torch.stack(outs, dim=0).to(input_tensor.device)


# ═══════════════════════════════════════════════════════════
#  Reconstruction: Lensless → Scene (ADAPTIVE WIENER)
# ═══════════════════════════════════════════════════════════
def reconstruct_an_image(image):
    """
    Full 256×256 lensless (HWC uint8 or float) → 128×128 reconstructed scene (HWC uint8).

    Uses adaptive Wiener regularization: lambda = WIENER_FRACTION * max(|H|²)
    This works correctly regardless of unitSensor.
    """
    image = np.array(image)
    image = image.transpose(2, 0, 1).astype(np.float32)

    cache = _get_psf_fft()
    psf_fft = cache["psf_fft"]
    full_len = cache["full_len"]
    wiener_denom = cache["wiener_denom"]
    crop_img = make_center_cropper(numScenePix, full_len)

    # Wiener deconvolution with adaptive regularization
    xhat_fft = (
        np.conjugate(psf_fft)
        * np.fft.fft2(image, s=(full_len, full_len), axes=(1, 2))
    ) / wiener_denom

    xhat = crop_img(
        np.fft.fftshift(
            np.fft.ifft2(xhat_fft, axes=(1, 2)).real,
            axes=(1, 2),
        )
    )

    # Normalize
    eps = 1e-8
    if np.any(xhat):
        xhat = (xhat - xhat.min()) / max(xhat.max() - xhat.min(), eps)
    xhat = np.nan_to_num(xhat, nan=0.0, posinf=1.0, neginf=0.0)
    return (xhat * 255.0).clip(0, 255).transpose(1, 2, 0).astype(np.uint8)


# ═══════════════════════════════════════════════════════════
#  V7 Postprocess: Center 128 → Reconstruct
# ═══════════════════════════════════════════════════════════
def partial_reconstruct_tensor_rev_V7_128(
    lensless_center_batch,
    renorm_after_paste=False,
):
    """
    (B,3,128,128) lensless center crop → (B,3,128,128) reconstructed scene [0,1].

    Pipeline per image:
      1. Use center 128 patch as scene, forward to get full 256 lensless estimate
      2. Paste original patch into center (feathered blend)
      3. Reconstruct using adaptive Wiener filter
      4. Return 128×128 scene
    """
    assert lensless_center_batch.dim() == 4 and lensless_center_batch.size(1) == 3
    device = lensless_center_batch.device
    B = lensless_center_batch.size(0)
    outs = []
    eps = 1e-8

    with torch.no_grad():
        for b in range(B):
            patch = lensless_center_batch[b]

            # Support [-1,1] or [0,1]
            if patch.min().item() < -0.05 or patch.max().item() > 1.05:
                patch01 = patch.clamp(-1, 1).mul(0.5).add(0.5)
            else:
                patch01 = patch.clamp(0, 1)

            # To CPU numpy
            patch_u8 = (patch01.detach().cpu().numpy() * 255.0).astype(np.uint8)
            patch_hwc_u8 = np.transpose(patch_u8, (1, 2, 0))

            # Full lensless estimate from patch
            y_est_full_u8 = convert_image_to_lensless_full(
                Image.fromarray(patch_hwc_u8, "RGB")
            )

            # Paste original patch into center (feathered)
            y_fused_u8 = paste_center_hwc_feather(
                y_est_full_u8.copy(), patch_hwc_u8,
            )

            if renorm_after_paste:
                y01 = y_fused_u8.astype(np.float32) / 255.0
                cmin = y01.min(axis=(0, 1), keepdims=True)
                cmax = y01.max(axis=(0, 1), keepdims=True)
                y01 = (y01 - cmin) / np.maximum(cmax - cmin, eps)
                y_fused_u8 = (y01 * 255.0).round().clip(0, 255).astype(np.uint8)

            # Reconstruct (now uses adaptive Wiener)
            rec_u8 = reconstruct_an_image(y_fused_u8)
            rec01 = rec_u8.astype(np.float32) / 255.0
            rec_chw = torch.from_numpy(
                np.transpose(rec01, (2, 0, 1))
            ).contiguous()
            outs.append(rec_chw)

    return torch.stack(outs, dim=0).to(device)