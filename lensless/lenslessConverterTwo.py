#!/usr/bin/env python3
"""
Scene: 128×128 RGB
Sensor (full lensless measurement): 256×256 RGB
Network I/O: central 128×128 crop from the 256×256 lensless measurement.

This script provides:
1) convert_image_to_lensless_full(): scene -> full 256×256 lensless (uint8)
2) convert_into_lensless_center():  (B,3,128,128) -> (B,3,128,128) center crop of lensless
3) reconstruct_an_image_fullsensor(): full 256×256 lensless -> 128×128 reconstructed scene (uint8)
4) partial_reconstruct_tensor_rev_V7_128(): consumes (B,3,128,128) lensless center patch and does V7-style postprocessing:
     - build a full 256×256 lensless estimate from the 128×128 patch
     - paste original patch into the center
     - reconstruct using full-sensor model
     - return (B,3,128,128) float in [0,1]
"""
import numpy as np
import torch
from PIL import Image
from scipy.signal import max_len_seq
import numpy as np
import torch
from PIL import Image

# =========================
#   GLOBAL CAMERA PARAMS
# =========================
numScenePix = 128
numSensorPix = 256
numMaskPix = 127
unitMask = 2
unitSensor = 5
d = 40000
std = 0.02
z2a = lambda z: (1 - d / z)
a2z = lambda a: d / (1 - a)
depthList = z2a(np.array([50e4], dtype=np.float64))


# =========================
#   GRID LOCATIONS
# =========================
def make_locations():
    locSensor = np.linspace(
        -numSensorPix * unitSensor / 2 + unitSensor / 2,
        numSensorPix * unitSensor / 2 - unitSensor / 2,
        numSensorPix,
        dtype=np.float64
    )
    locMask = np.linspace(
        -numMaskPix * unitMask / 2 + unitMask / 2,
        numMaskPix * unitMask / 2 - unitMask / 2,
        numMaskPix,
        dtype=np.float64
    )
    return locSensor, locMask


locSensor, locMask = make_locations()


# =========================
#   CENTER CROP HELPERS
# =========================
def center_crop_hwc(img_hwc: np.ndarray, crop_hw: int) -> np.ndarray:
    """Center crop HWC image to crop_hw×crop_hw."""
    H, W = img_hwc.shape[:2]
    assert H >= crop_hw and W >= crop_hw, f"Cannot crop {crop_hw} from {H}x{W}"
    ys = (H - crop_hw) // 2
    xs = (W - crop_hw) // 2
    return img_hwc[ys:ys + crop_hw, xs:xs + crop_hw, :]


import numpy as np

def paste_center_hwc_feather(dst_u8: np.ndarray,
                             patch_u8: np.ndarray,
                             feather: int = 16) -> np.ndarray:
    """
    dst_u8:   (256,256,3) uint8
    patch_u8: (128,128,3) uint8
    feather:  pixels of smooth transition at patch border (try 8, 16, 24)
    """
    H, W, _ = dst_u8.shape
    h, w, _ = patch_u8.shape
    ys = (H - h) // 2
    xs = (W - w) // 2

    out = dst_u8.astype(np.float32).copy()
    patch = patch_u8.astype(np.float32)

    # Build 2D smooth mask in [0,1], 1 in patch centre, ramps down near edges
    yy = np.linspace(0, 1, h, dtype=np.float32)
    xx = np.linspace(0, 1, w, dtype=np.float32)
    Y, X = np.meshgrid(yy, xx, indexing="ij")

    # distance to nearest edge in pixels
    d = np.minimum.reduce([Y, 1 - Y, X, 1 - X]) * min(h, w)
    m = np.clip(d / max(feather, 1), 0.0, 1.0)  # 0 at border, 1 inside
    m = (m * m * (3 - 2 * m))  # smoothstep
    m = m[..., None]  # (h,w,1)

    region = out[ys:ys+h, xs:xs+w, :]
    out[ys:ys+h, xs:xs+w, :] = m * patch + (1 - m) * region

    return np.clip(out, 0, 255).astype(np.uint8)


def paste_center_hwc(dst_hwc: np.ndarray, patch_hwc: np.ndarray) -> np.ndarray:
    """Paste patch into the center of dst (both HWC)."""
    H, W = dst_hwc.shape[:2]
    h, w = patch_hwc.shape[:2]
    ys = (H - h) // 2
    xs = (W - w) // 2
    dst_hwc[ys:ys + h, xs:xs + w, :] = patch_hwc
    return dst_hwc


def make_center_cropper(target_hw: int, full_hw: int):
    """Center cropper for CHW arrays (C,full_hw,full_hw) -> (C,target_hw,target_hw)."""
    start = (full_hw - target_hw) // 2
    end = start + target_hw

    def _crop(x: np.ndarray) -> np.ndarray:
        return x[:, start:end, start:end]

    return _crop


# =========================
#   PSF / MASK
# =========================
def generate_interp_matrix(depth_list, loc_sensor, loc_mask, unit_mask, num_sensor_pix, num_mask_pix):
    num_depth = depth_list.size
    interp = np.zeros((num_sensor_pix, num_mask_pix, num_depth), dtype=np.float64)

    for di in range(num_depth):
        loc_interp = depth_list[di] * loc_sensor

        hi = np.minimum(len(loc_mask) - 1, np.searchsorted(loc_mask, loc_interp, side="right"))
        lo = np.maximum(0, hi - 1)

        rows = np.arange(num_sensor_pix)

        w_lo = 1.0 - (loc_interp - loc_mask[lo]) / unit_mask
        w_hi = 1.0 - (loc_mask[hi] - loc_interp) / unit_mask

        w_lo = np.clip(w_lo, 0.0, 1.0)
        w_hi = np.clip(w_hi, 0.0, 1.0)

        interp[rows, lo, di] = w_lo
        interp[rows, hi, di] = w_hi

        oob = np.logical_or(loc_interp < loc_mask[0], loc_interp > loc_mask[-1])
        interp[oob, :, di] = 0.0

    return interp


def create_mask_for_lensless() -> np.ndarray:
    """
    Returns psf: (numSensorPix, numSensorPix) float64
    """
    m = int(np.log2(numMaskPix + 1))
    mask_vec = max_len_seq(m)[0].reshape(numMaskPix, 1).astype(np.float64)
    mask_pattern = mask_vec @ mask_vec.T  # (numMaskPix,numMaskPix)

    interp3 = generate_interp_matrix(
        depth_list=depthList,
        loc_sensor=locSensor,
        loc_mask=locMask,
        unit_mask=unitMask,
        num_sensor_pix=numSensorPix,
        num_mask_pix=numMaskPix
    )
    interp = interp3[:, :, 0]

    psf = interp @ mask_pattern @ interp.T
    return psf.astype(np.float64)


# =========================
#   FORWARD (scene -> lensless full sensor)
# =========================
def convert_image_to_lensless_full(pil_image: Image.Image) -> np.ndarray:
    img = pil_image.convert("RGB").resize((numScenePix, numScenePix), resample=Image.BICUBIC)
    img = np.asarray(img, dtype=np.float32) / 255.0
    img = img.transpose(2, 0, 1).astype(np.float64)

    psf = create_mask_for_lensless()
    full_len = numSensorPix + numScenePix - 1
    crop_sensor = make_center_cropper(numSensorPix, full_len)

    psf_fft = np.fft.fft2(psf, s=(full_len, full_len))
    y = crop_sensor(np.fft.ifft2(psf_fft * np.fft.fft2(img, s=(full_len, full_len), axes=(1,2)), axes=(1,2)).real)

    eps = 1e-8
    global_min = y.min()
    global_max = y.max()
    y = (y - global_min) / max(global_max - global_min, eps)

    # keep uint8 return — callers expect this
    return (y * 255.0).clip(0, 255).astype(np.uint8).transpose(1, 2, 0)


def convert_into_lensless(input_tensor, center_size=(128, 128)):
    image_array = []
    with torch.no_grad():
        for img in input_tensor.detach().cpu():

            # handle both [-1,1] and [0,1] input
            if img.min().item() < -0.05:
                img01 = img.clamp(-1.0, 1.0) * 0.5 + 0.5  # [-1,1] -> [0,1]
            else:
                img01 = img.clamp(0.0, 1.0)

            img_np = (img01.numpy().transpose(1, 2, 0) * 255.0).astype(np.uint8)
            pil_img = Image.fromarray(img_np, "RGB")

            lensless_u8 = convert_image_to_lensless_full(pil_img)
            lensless_f32 = lensless_u8.astype(np.float32) / 255.0

            H, W, _ = lensless_f32.shape
            ch, cw = center_size
            sh, sw = (H - ch) // 2, (W - cw) // 2
            center = lensless_f32[sh:sh+ch, sw:sw+cw, :]
            image_array.append(center.transpose(2, 0, 1))

    return torch.from_numpy(np.stack(image_array)).float().to(input_tensor.device)

# def convert_into_lensless(input_tensor: torch.Tensor, center_size=(128, 128)) -> torch.Tensor:
#     """
#     input_tensor: (B,3,128,128) in [0,1] or [-1,1]
#     returns     : (B,3,128,128) lensless center crop in [0,1]
#
#     Matches the 64×64 version behaviour:
#       - relies on convert_image_to_lensless_full() to do normalisation
#       - NO extra min/max normalisation here
#     """
#     image_array = []
#
#     with torch.no_grad():
#         for img in input_tensor.detach().cpu():
#             img01 = img
#             # tensor -> uint8 HWC
#             img_np = (img01.numpy().transpose(1, 2, 0) * 255.0).astype(np.uint8)
#             pil_img = Image.fromarray(img_np, "RGB")
#
#             # full lensless 256×256 (already normalised inside)
#             lensless_u8 = convert_image_to_lensless_full(pil_img)  # (256,256,3) uint8
#             lensless01 = lensless_u8.astype(np.float32) / 255.0  # [0,1]
#
#             # center crop to 128×128
#             H, W, _ = lensless01.shape
#             ch, cw = center_size
#             sh = (H - ch) // 2
#             sw = (W - cw) // 2
#             center = lensless01[sh:sh + ch, sw:sw + cw, :]  # (128,128,3)
#
#             image_array.append(center.transpose(2, 0, 1))  # CHW
#
#     return torch.from_numpy(np.stack(image_array)).float().to(input_tensor.device)


def convert_into_lensless_center(input_tensor: torch.Tensor, center_hw: int = 128) -> torch.Tensor:
    """
    input_tensor: (B,3,128,128) in [0,1]
    returns     : (B,3,128,128) in [0,1] (CENTER CROP from full 256×256 lensless)
    """
    assert input_tensor.dim() == 4 and input_tensor.size(1) == 3
    outs = []
    with torch.no_grad():
        x = input_tensor.detach().cpu().float().clamp(0.0, 1.0).numpy()
        for i in range(x.shape[0]):
            rgb = (x[i].transpose(1, 2, 0) * 255.0).round().astype(np.uint8)  # HWC 128
            y_full = convert_image_to_lensless_full(Image.fromarray(rgb, "RGB"))  # HWC 256
            y_center = center_crop_hwc(y_full, center_hw)  # HWC 128
            y_t = torch.from_numpy(y_center).permute(2, 0, 1).float() / 255.0
            outs.append(y_t)
    return torch.stack(outs, dim=0).to(input_tensor.device)


# =========================
#   RECON (full lensless sensor -> scene)
# =========================
def reconstruct_an_image(image):
    image = np.array(image)
    image = image.transpose(2, 0, 1).astype(np.float32)

    psf = create_mask_for_lensless()   # <-- only psf now

    fullLength = numSensorPix + numScenePix - 1
    crop_img = make_center_cropper(numScenePix, fullLength)

    psf_fft = np.fft.fft2(psf, s=[fullLength, fullLength])

    lambda_snr = float(np.sqrt(std))
    eps = 1e-8
    denom = (np.abs(psf_fft) ** 2) + max(lambda_snr, eps)

    xhat_fft = (np.conjugate(psf_fft) * np.fft.fft2(image, s=[fullLength, fullLength], axes=[1, 2])) / denom
    xhat = crop_img(np.fft.fftshift(np.fft.ifft2(xhat_fft, axes=[1, 2]).real, axes=[1, 2]))

    all_zero = not np.any(xhat)
    if not all_zero:
        global_min = xhat.min()
        global_max = xhat.max()
        span = max(global_max - global_min, eps)
        xhat = (xhat - global_min) / span
    xhat = np.nan_to_num(xhat, nan=0.0, posinf=1.0, neginf=0.0)
    xhat = (xhat * 255.0).clip(0, 255).transpose(1, 2, 0).astype(np.uint8)
    return xhat


# =========================
#   V7 POSTPROCESS (CENTER 128 CONSUMER)
# =========================
def partial_reconstruct_tensor_rev_V7_128(
        lensless_center_batch: torch.Tensor,
        renorm_after_paste: bool = False,
) -> torch.Tensor:
    """
    Returns: (B,3,128,128) float in [0,1] on SAME device as input.

    Note:
        are NumPy FFT pipelines -> they run on CPU internally.
      - We only move per-sample tensors to CPU right before PIL/NumPy,
        then bring the final reconstructed tensor back to the original device.
    """
    assert lensless_center_batch.dim() == 4 and lensless_center_batch.size(1) == 3
    device = lensless_center_batch.device
    B = lensless_center_batch.size(0)
    outs = []
    eps = 1e-8

    with torch.no_grad():
        for b in range(B):
            patch = lensless_center_batch[b]  # (3,128,128) on GPU

            # support [-1,1] or [0,1]
            if patch.min().item() < -0.05 or patch.max().item() > 1.05:
                patch01 = patch.clamp(-1, 1).mul(0.5).add(0.5)
            else:
                patch01 = patch.clamp(0, 1)

            # --- move ONLY this sample to CPU for PIL/NumPy ---
            patch_u8 = (patch01.detach().cpu().numpy() * 255.0).astype(np.uint8)  # (3,128,128)
            patch_hwc_u8 = np.transpose(patch_u8, (1, 2, 0))  # (128,128,3)

            # full 256 lensless estimate
            y_est_full_u8 = convert_image_to_lensless_full(Image.fromarray(patch_hwc_u8, "RGB"))  # (256,256,3) uint8

            # paste original patch into center
            y_fused_u8 = paste_center_hwc_feather(y_est_full_u8.copy(), patch_hwc_u8)  # (256,256,3) uint8

            if renorm_after_paste:
                y01 = y_fused_u8.astype(np.float32) / 255.0
                cmin = y01.min(axis=(0, 1), keepdims=True)
                cmax = y01.max(axis=(0, 1), keepdims=True)
                y01 = (y01 - cmin) / np.maximum(cmax - cmin, eps)
                y_fused_u8 = (y01 * 255.0).round().clip(0, 255).astype(np.uint8)

            # reconstruct scene (128)
            rec_u8 = reconstruct_an_image(y_fused_u8)  # (128,128,3)
            rec01 = rec_u8.astype(np.float32) / 255.0
            rec_chw = torch.from_numpy(np.transpose(rec01, (2, 0, 1))).contiguous()  # CPU tensor
            outs.append(rec_chw)

    # stack on CPU then send back to GPU (same device as input)
    return torch.stack(outs, dim=0).to(device)


def partial_reconstruct_tensor_rev_V8_128(
        lensless_center_batch: torch.Tensor,
        renorm_after_paste: bool = False,
) -> torch.Tensor:
    """
    Returns: (B,3,128,128) float in [-1,1] on SAME device as input.

    Stays in float32 throughout — never casts to uint8 — so chroma
    information in the revealed secret is preserved across the FFT pipeline.

    Expects input in [-1,1] (tanh output from RevealNet).
    Also accepts [0,1] input gracefully.
    """
    assert lensless_center_batch.dim() == 4 and lensless_center_batch.size(1) == 3
    device = lensless_center_batch.device
    B = lensless_center_batch.size(0)
    outs = []
    eps = 1e-8

    with torch.no_grad():
        for b in range(B):
            patch = lensless_center_batch[b]  # (3,128,128)

            # ── normalise input to [0,1] float32, never uint8 ──────────────
            if patch.min().item() < -0.05 or patch.max().item() > 1.05:
                # tanh range [-1,1] -> [0,1]
                patch01 = patch.clamp(-1.0, 1.0).mul(0.5).add(0.5)
            else:
                patch01 = patch.clamp(0.0, 1.0)

            # (3,H,W) -> (H,W,3) float32 numpy, range [0,1]
            patch_hwc_f32 = patch01.detach().cpu().numpy().transpose(1, 2, 0).astype(np.float32)

            # ── full lensless estimate (float32 in, float32 out) ───────────
            # convert_image_to_lensless_full_f32 should accept (H,W,3) float32
            # [0,1] and return (256,256,3) float32 [0,1].
            # If your function only accepts PIL, convert minimally:
            patch_pil = Image.fromarray(
                (patch_hwc_f32 * 255.0).round().clip(0, 255).astype(np.uint8), "RGB"
            )
            y_est_full_f32 = np.array(
                convert_image_to_lensless_full(patch_pil), dtype=np.float32
            ) / 255.0                                          # (256,256,3) float32 [0,1]

            # ── paste center with feathering, stay float32 ────────────────
            y_fused_f32 = paste_center_hwc_feather_f32(
                y_est_full_f32.copy(), patch_hwc_f32
            )                                                  # (256,256,3) float32 [0,1]

            # ── optional renorm (operates on float, no uint8 round-trip) ──
            if renorm_after_paste:
                cmin = y_fused_f32.min(axis=(0, 1), keepdims=True)
                cmax = y_fused_f32.max(axis=(0, 1), keepdims=True)
                y_fused_f32 = (y_fused_f32 - cmin) / np.maximum(cmax - cmin, eps)

            # ── reconstruct scene, stay float32 ───────────────────────────
            # reconstruct_an_image_f32 should accept (256,256,3) float32 [0,1]
            # and return (128,128,3) float32 [0,1].
            # If your function only accepts uint8, wrap minimally:
            y_fused_u8 = (y_fused_f32 * 255.0).round().clip(0, 255).astype(np.uint8)
            rec_f32 = np.array(
                reconstruct_an_image(y_fused_u8), dtype=np.float32
            ) / 255.0                                          # (128,128,3) float32 [0,1]

            # ── back to tensor, convert [0,1] -> [-1,1] to match tanh range
            rec_chw = torch.from_numpy(
                rec_f32.transpose(2, 0, 1)
            ).contiguous()                                     # (3,128,128) float32 [0,1]

            # restore to [-1,1] so output range matches RevealNet output
            rec_chw = rec_chw.mul(2.0).sub(1.0)               # (3,128,128) float32 [-1,1]

            outs.append(rec_chw)

    return torch.stack(outs, dim=0).to(device)


# =============================================================================
#   Helper: paste_center_hwc_feather operating entirely in float32
#   Drop-in replacement for paste_center_hwc_feather if it internally
#   casts to uint8.  Copy your original feather mask logic here.
# =============================================================================
def paste_center_hwc_feather_f32(
        canvas_f32: np.ndarray,    # (H,W,3) float32 [0,1]  -- modified in place
        center_f32: np.ndarray,    # (h,w,3) float32 [0,1]
) -> np.ndarray:
    """
    Paste center_f32 into the middle of canvas_f32 with a soft feather blend.
    Operates entirely in float32 — no uint8 cast.
    Returns (H,W,3) float32 [0,1].
    """
    H, W = canvas_f32.shape[:2]
    h, w = center_f32.shape[:2]

    top  = (H - h) // 2
    left = (W - w) // 2

    # build a 2-D feather weight for the center patch (1=center patch, 0=canvas)
    feather_px = max(1, min(h, w) // 8)   # feather width in pixels
    weight = np.ones((h, w), dtype=np.float32)
    for i in range(feather_px):
        alpha = (i + 1) / (feather_px + 1)   # ramps 0->1 from edge inward
        weight[i,  :]  = np.minimum(weight[i,  :],  alpha)
        weight[-1-i,:] = np.minimum(weight[-1-i,:], alpha)
        weight[:,  i]  = np.minimum(weight[:,  i],  alpha)
        weight[:,-1-i] = np.minimum(weight[:,-1-i], alpha)

    weight = weight[:, :, np.newaxis]   # (h,w,1) for broadcast over channels

    roi = canvas_f32[top:top+h, left:left+w, :]
    canvas_f32[top:top+h, left:left+w, :] = (
        weight * center_f32 + (1.0 - weight) * roi
    )
    return canvas_f32





# =========================
#   QUICK TEST
# =========================
if __name__ == "__main__":
    in_path = "img.png"  # change

    raw = Image.open(in_path).convert("RGB").resize((numScenePix, numScenePix), Image.BICUBIC)
    raw.save("scene_128.png")

    # full lensless 256
    lensless_128 = convert_into_lensless(raw)
    Image.fromarray(lensless_128, "RGB").save("lensless_full_128.png")

    # center lensless 128 (what your model will use)

    # reconstruct from full sensor (baseline)
    recon = partial_reconstruct_tensor_rev_V7_128(lensless_128, apply_soft_lowpass=True)
    Image.fromarray(recon, "RGB").save("recon_from_128.png")

    print("Saved: scene_128.png, lensless_full_256.png, lensless_center_128.png, recon_from_full_256.png")
