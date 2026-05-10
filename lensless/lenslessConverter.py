import numpy as np
import torch
from PIL import Image
from scipy.signal import max_len_seq

numScenePix = 64
numMaskPix = 63
unitMask = 2
numSensorPix = 128
unitSensor = 25
d = 40000  # sensor to mask distance
midScenePix = int(np.floor(numScenePix / 2))
midSensorPix = int(np.floor(numSensorPix / 2)) - 1
# helper functions
z2a = lambda z: (1 - d / z)
a2z = lambda a: d / (1 - a)
crop_sensor = lambda x: x[:, midScenePix: midScenePix + numSensorPix, midScenePix:midScenePix + numSensorPix]
crop_img = lambda x: x[:, midSensorPix: midSensorPix + numScenePix, midSensorPix: midSensorPix + numScenePix]
# create a list of predefined depth planes
depthList = z2a(np.array([50e4]))  # 50cm away   (100000:1km, 50000:500m, 100:1m)
# define functions for interpolation
locSensor = np.linspace(-numSensorPix * unitSensor / 2 + unitSensor / 2, numSensorPix * unitSensor / 2 - unitSensor / 2,
                        numSensorPix)
locMask = np.linspace(-numMaskPix * unitMask / 2 + unitMask / 2, numMaskPix * unitMask / 2 - unitMask / 2, numMaskPix)
std = 0.02


def create_mask_for_lensless():
    def generateInterpMatrix(depthList, locSensor, locMask):
        numDepth = depthList.size
        interpMatrix = np.zeros([numSensorPix, numMaskPix, numDepth])
        for di in np.arange(numDepth):
            locInterp = depthList[di] * locSensor
            hi = np.minimum(len(locMask) - 1, np.searchsorted(locMask, locInterp, 'right'))
            lo = np.maximum(0, hi - 1)
            interpMatrix[np.arange(len(lo)), lo, di] = 1 - (locInterp - locMask[lo]) / unitMask
            interpMatrix[np.arange(len(hi)), hi, di] = 1 - (locMask[hi] - locInterp) / unitMask
            rowsCast = np.where(np.logical_or(locInterp < locMask[0], locInterp > locMask[-1]))
            interpMatrix[rowsCast, :, di] = 0
        return interpMatrix

    # ------------------------------------------------------------------------------
    # create mask pattern
    maskVec = max_len_seq(int(np.log2(numMaskPix + 1)))[0].reshape(numMaskPix, 1)
    maskPattern = maskVec @ maskVec.T
    # print(maskPattern)
    # generate interpolation matrices for multiple depths
    interpMatrix = generateInterpMatrix(depthList, locSensor, locMask)[:, :, 0]
    psf = interpMatrix @ maskPattern @ interpMatrix.T
    return crop_img, crop_sensor, psf


def create_non_binary_mask_for_lensless():
    # linear interpolation matrix
    def generateInterpMatrix(depthList, locSensor, locMask):
        numDepth = depthList.size
        interpMatrix = np.zeros([numSensorPix, numMaskPix, numDepth])
        for di in np.arange(numDepth):
            locInterp = depthList[di] * locSensor
            hi = np.minimum(len(locMask) - 1, np.searchsorted(locMask, locInterp, 'right'))
            lo = np.maximum(0, hi - 1)
            # Here we modify the weights to be non-binary (smooth interpolation)
            interpMatrix[np.arange(len(lo)), lo, di] = 1 - (locInterp - locMask[lo]) / unitMask
            interpMatrix[np.arange(len(hi)), hi, di] = 1 - (locMask[hi] - locInterp) / unitMask

            # Smoothen the edges for the interpolation
            interpMatrix[np.arange(len(lo)), lo, di] = np.clip(interpMatrix[np.arange(len(lo)), lo, di], 0, 1)
            interpMatrix[np.arange(len(hi)), hi, di] = np.clip(interpMatrix[np.arange(len(hi)), hi, di], 0, 1)
            # Handling values outside the mask range
            rowsCast = np.where(np.logical_or(locInterp < locMask[0], locInterp > locMask[-1]))
            interpMatrix[rowsCast, :, di] = 0
        return interpMatrix

    # ------------------------------------------------------------------------------
    # Create mask pattern
    maskVec = max_len_seq(int(np.log2(numMaskPix + 1)))[0].reshape(numMaskPix, 1)
    maskPattern = maskVec @ maskVec.T

    # Generate interpolation matrices for multiple depths
    interpMatrix = generateInterpMatrix(depthList, locSensor, locMask)[:, :, 0]

    # Now the PSF will be non-binary
    psf = interpMatrix @ maskPattern @ interpMatrix.T

    return crop_img, crop_sensor, psf


def create_gaussian_weighted_psf():
    numSensorPix = 128
    numMaskPix = 128

    # Gaussian-weighted interpolation matrix
    def generateGaussianInterpMatrix(depthList, locSensor, locMask, numSensorPix, numMaskPix, sigma=0.5):
        numDepth = depthList.size
        interpMatrix = np.zeros([numSensorPix, numMaskPix, numDepth])

        for di in np.arange(numDepth):
            locInterp = depthList[di] * locSensor
            hi = np.minimum(len(locMask) - 1, np.searchsorted(locMask, locInterp, 'right'))
            lo = np.maximum(0, hi - 1)

            # Ensure indices stay within bounds
            lo = np.clip(lo, 0, len(locMask) - 1)
            hi = np.clip(hi, 0, len(locMask) - 1)

            valid_indices = np.arange(min(len(lo), numSensorPix))

            interpMatrix[valid_indices, lo[valid_indices], di] = 1 - (
                    locInterp[valid_indices] - locMask[lo[valid_indices]]) / unitMask
            interpMatrix[valid_indices, hi[valid_indices], di] = 1 - (
                    locMask[hi[valid_indices]] - locInterp[valid_indices]) / unitMask

            # Smoothen the edges for the interpolation
            interpMatrix[valid_indices, lo[valid_indices], di] = np.clip(
                interpMatrix[valid_indices, lo[valid_indices], di], 0, 1)
            interpMatrix[valid_indices, hi[valid_indices], di] = np.clip(
                interpMatrix[valid_indices, hi[valid_indices], di], 0, 1)

            # Handling values outside the mask range
            rowsCast = np.where(np.logical_or(locInterp < locMask[0], locInterp > locMask[-1]))
            interpMatrix[rowsCast, :, di] = 0

        return interpMatrix

    # Create a fixed but different mask pattern using a deterministic random seed
    np.random.seed(42)  # Ensures the pattern is different but fixed for each run
    maskPattern = np.random.randint(0, 2, (numMaskPix, numMaskPix))

    # Introduce variations in at least 900 locations
    random_indices = np.random.choice(numMaskPix * numMaskPix, 900, replace=False)
    maskPattern.flat[random_indices] = 1 - maskPattern.flat[random_indices]

    # Generate interpolation matrices for multiple depths
    interpMatrix = generateGaussianInterpMatrix(depthList, locSensor, locMask, numSensorPix, numMaskPix)[:, :, 0]

    # Apply transformation to create the new PSF
    psf = interpMatrix @ maskPattern @ interpMatrix.T

    return crop_img, crop_sensor, psf


def create_mask_for_lensless_second_version():
    unitMask = 20

    # Gaussian-weighted interpolation matrix
    def generateGaussianInterpMatrix(depthList, locSensor, locMask, numSensorPix, numMaskPix, sigma=0.5):
        numDepth = depthList.size
        interpMatrix = np.zeros([numSensorPix, numMaskPix, numDepth])

        for di in np.arange(numDepth):
            locInterp = depthList[di] * locSensor
            hi = np.minimum(len(locMask) - 1, np.searchsorted(locMask, locInterp, 'right'))
            lo = np.maximum(0, hi - 1)

            # Ensure indices stay within bounds
            lo = np.clip(lo, 0, len(locMask) - 1)
            hi = np.clip(hi, 0, len(locMask) - 1)

            valid_indices = np.arange(min(len(lo), numSensorPix))

            interpMatrix[valid_indices, lo[valid_indices], di] = 1 - (
                    locInterp[valid_indices] - locMask[lo[valid_indices]]) / unitMask
            interpMatrix[valid_indices, hi[valid_indices], di] = 1 - (
                    locMask[hi[valid_indices]] - locInterp[valid_indices]) / unitMask

            # Smoothen the edges for the interpolation
            interpMatrix[valid_indices, lo[valid_indices], di] = np.clip(
                interpMatrix[valid_indices, lo[valid_indices], di], 0, 1)
            interpMatrix[valid_indices, hi[valid_indices], di] = np.clip(
                interpMatrix[valid_indices, hi[valid_indices], di], 0, 1)

            # Handling values outside the mask range
            rowsCast = np.where(np.logical_or(locInterp < locMask[0], locInterp > locMask[-1]))
            interpMatrix[rowsCast, :, di] = 0

        return interpMatrix

    # Create a fixed but different mask pattern using a deterministic random seed
    np.random.seed(42)  # Ensures the pattern is different but fixed for each run
    maskPattern = np.random.randint(0, 2, (numMaskPix, numMaskPix))

    # Introduce variations in at least 900 locations
    random_indices = np.random.choice(numMaskPix * numMaskPix, 900, replace=False)
    maskPattern.flat[random_indices] = 1 - maskPattern.flat[random_indices]

    # Generate interpolation matrices for multiple depths
    interpMatrix = generateGaussianInterpMatrix(depthList, locSensor, locMask, numSensorPix, numMaskPix)[:, :, 0]

    # Apply transformation to create the new PSF
    psf = interpMatrix @ maskPattern @ interpMatrix.T

    return crop_img, crop_sensor, psf


def convert_image_to_lensless(image):
    # This returns an image with noise and lensless transformation
    crop_img, crop_sensor, psf = create_mask_for_lensless()
    fullLength = numSensorPix + numScenePix - 1
    psf_fft = np.fft.fft2(psf, s=[fullLength, fullLength])
    image = image.resize((numScenePix, numScenePix)).convert("RGB")
    image = np.array(image)
    image = image.transpose(2, 0, 1).astype(np.double) / 255
    img_fft = np.fft.fft2(image, s=[fullLength, fullLength], axes=[1, 2])
    y_fft = psf_fft * img_fft
    y = crop_sensor(np.fft.ifft2(y_fft, axes=[1, 2]).real)
    y_im = y
    # Normalizing the image so that image disk writing makes little difference
    y_im -= np.amin(y_im, axis=(1, 2))[:, None, None]
    y_im /= np.amax(y_im, axis=(1, 2))[:, None, None]
    y_im = 255 * y_im
    y_im = y_im.transpose(1, 2, 0).astype(np.uint8)
    return y_im


def reconstruct_an_image(image):
    image = np.array(image)
    # --- minimal change: use float32 for FFT math ---
    image = image.transpose(2, 0, 1).astype(np.float32)  # was uint8

    crop_img, crop_sensor, psf = create_mask_for_lensless()
    fullLength = numSensorPix + numScenePix - 1

    psf_fft = np.fft.fft2(psf, s=[fullLength, fullLength])

    # --- minimal change: epsilon in denom to avoid /0 ---
    lambda_snr = float(np.sqrt(std))
    eps = 1e-8
    denom = (np.abs(psf_fft) ** 2) + max(lambda_snr, eps)

    xhat_fft = (np.conjugate(psf_fft) * np.fft.fft2(image, s=[fullLength, fullLength], axes=[1, 2])) / denom
    xhat = crop_img(np.fft.fftshift(np.fft.ifft2(xhat_fft, axes=[1, 2]).real, axes=[1, 2]))

    all_zero = not np.any(xhat)
    if not all_zero:
        # per-channel min/max with keepdims
        ch_min = np.amin(xhat, axis=(1, 2), keepdims=True)
        ch_max = np.amax(xhat, axis=(1, 2), keepdims=True)
        span = np.maximum(ch_max - ch_min, eps)
        xhat = (xhat - ch_min) / span

    # --- minimal change: scrub NaN/Inf just in case ---
    xhat = np.nan_to_num(xhat, nan=0.0, posinf=1.0, neginf=0.0)

    xhat = xhat * 255.0
    xhat = xhat.transpose(1, 2, 0).astype(np.uint8)
    return xhat


def convert_into_lensless(input_tensor, center_size=(64, 64)):
    image_array = []
    with torch.no_grad():
        for images in input_tensor.detach().cpu().numpy():
            # Convert tensor to HxWxC uint8 image
            image = images.transpose(1, 2, 0)
            image = np.clip(image * 255.0, 0, 255).astype(np.uint8)
            pil_image = Image.fromarray(image)

            # Get lensless version (NO resizing)
            lensless_np = convert_image_to_lensless(pil_image)  # HxWxC, uint8
            lensless_np = lensless_np.astype(np.float32) / 255.0  # Normalize to [0,1]

            # ✅ Extract ONLY the center 64×64 region
            lensless_h, lensless_w, _ = lensless_np.shape
            ch, cw = center_size
            start_h = (lensless_h - ch) // 2
            start_w = (lensless_w - cw) // 2
            center_patch = lensless_np[start_h:start_h + ch, start_w:start_w + cw, :]  # [64, 64, 3]
            image_array.append(center_patch.transpose(2, 0, 1))
    return torch.from_numpy(np.stack(image_array)).float()


def _to01_from_m11(t: torch.Tensor) -> torch.Tensor:
    # Map [-1,1] -> [0,1] safely
    return t.clamp(-1.0, 1.0).mul(0.5).add(0.5)


def _down_to_64_hwc01(x_chw01: torch.Tensor) -> np.ndarray:
    # x_chw01: (3,H,W) torch float in [0,1]
    if x_chw01.shape[-2:] != (64, 64):
        x_chw01 = torch.nn.functional.interpolate(
            x_chw01.unsqueeze(0), size=(64, 64),
            mode="bicubic", align_corners=False
        ).squeeze(0)
    return x_chw01.detach().cpu().float().numpy().transpose(1, 2, 0)


def _gaussian_prefilter_hwc01(img_hwc01: np.ndarray, sigma_space: float = 0.8) -> np.ndarray:
    # Super-light, separable 3×3-ish smoothing via box-gauss approximation.
    # Keep tiny to avoid blur. Operates on HWC float [0,1].
    k = 3
    pad = k // 2
    H, W, C = img_hwc01.shape
    out = np.empty_like(img_hwc01)
    # Fixed tiny kernel (normalized)
    w = np.array([1, 2, 1], dtype=np.float32)
    w = w / w.sum()
    # Horizontal
    tmp = np.pad(img_hwc01, ((0, 0), (pad, pad), (0, 0)), mode='edge')
    tmp = (w[0] * tmp[:, 0:W, :] + w[1] * tmp[:, 1:W + 1, :] + w[2] * tmp[:, 2:W + 2, :])
    # Vertical
    tmp2 = np.pad(tmp, ((pad, pad), (0, 0), (0, 0)), mode='edge')
    out = (w[0] * tmp2[0:H, :, :] + w[1] * tmp2[1:H + 1, :, :] + w[2] * tmp2[2:H + 2, :, :])
    return out


def _joint_bilateral_hwc01(src_hwc01: np.ndarray,
                           guide_hwc01: np.ndarray,
                           sigma_space: float = 0.8,
                           sigma_color: float = 0.06,
                           radius: int = 1) -> np.ndarray:
    """
    Minimal joint bilateral (guided bilateral) on small 64×64.
    - src_hwc01: what we denoise (rev_partial, HWC float [0,1])
    - guide_hwc01: guidance (container downscaled to 64×64, HWC float [0,1])
    """
    H, W, C = src_hwc01.shape
    R = radius
    # Precompute spatial gaussian weights for offsets
    # For tiny radius=1, this is quick
    out = np.zeros_like(src_hwc01, dtype=np.float32)
    # Normalization constant per pixel
    norm = np.zeros((H, W, 1), dtype=np.float32)

    # Spatial kernel for offsets (dy, dx)
    # Using simple exp(- (dx^2+dy^2) / (2*sigma_s^2))
    s2 = 2.0 * (sigma_space ** 2) + 1e-12

    # Pad arrays for easy neighborhood access
    pad = R
    src_pad = np.pad(src_hwc01, ((pad, pad), (pad, pad), (0, 0)), mode='edge')
    guide_pad = np.pad(guide_hwc01, ((pad, pad), (pad, pad), (0, 0)), mode='edge')

    # Iterate over neighborhood offsets
    for dy in range(-R, R + 1):
        for dx in range(-R, R + 1):
            # spatial weight (scalar)
            w_spatial = np.exp(-(dx * dx + dy * dy) / s2).astype(np.float32)

            # gather neighbor slices
            nb_src = src_pad[pad + dy:pad + dy + H, pad + dx:pad + dx + W, :]  # H×W×C
            nb_guide = guide_pad[pad + dy:pad + dy + H, pad + dx:pad + dx + W, :]  # H×W×C

            # range weight from guide (per-pixel, per-channel → reduce to per-pixel)
            diff = nb_guide - guide_hwc01
            # L2 color distance per pixel
            d2 = np.sum(diff * diff, axis=2, keepdims=True)  # H×W×1
            w_range = np.exp(- d2 / (2.0 * (sigma_color ** 2) + 1e-12)).astype(np.float32)

            w = (w_spatial * w_range).astype(np.float32)  # H×W×1
            out += nb_src * w
            norm += w

    out /= np.maximum(norm, 1e-8)
    return np.clip(out, 0.0, 1.0).astype(np.float32)

def partial_reconstruct_tensor_rev_V7(
    lensless_center_batch: torch.Tensor,   # (B,3,64,64) lensless center patches in [-1,1] or [0,1]
) -> torch.Tensor:
    """
    Steps:
      1) Keep the original 64×64 *lensless* center patch.
      2) Treat that patch as a scene; call convert_image_to_lensless -> full 128×128 lensless estimate.
      3) Paste the original 64×64 lensless patch into the center of that 128×128 estimate.
      4) Send the 128×128 lensless image to reconstruct_an_image (your existing recon).
      5) Return the reconstructed *space-domain* image as a tensor in [0,1].

    Output: (B,3,H,W) float in [0,1]
    """
    assert lensless_center_batch.dim() == 4 and lensless_center_batch.size(1) == 3, \
        "Expect (B,3,64,64) lensless center patches"
    B = lensless_center_batch.size(0)
    outs = []

    with torch.no_grad():
        for b in range(B):
            # 1) original 64×64 lensless patch -> uint8 HWC
            patch = lensless_center_batch[b]
            if patch.min() < 0:  # support [-1,1] or [0,1]
                patch01 = patch.clamp(-1, 1).mul(0.5).add(0.5)
            else:
                patch01 = patch.clamp(0, 1)
            patch_u8 = (patch01.cpu().numpy() * 255.0).round().astype(np.uint8)   # (3,64,64)
            patch_hwc_u8 = np.transpose(patch_u8, (1, 2, 0))                      # (64,64,3)

            # 2) “pretend as scene” → lensless full frame (expects 128×128 from your pipeline)
            y_est_u8 = convert_image_to_lensless(Image.fromarray(patch_hwc_u8))    # (Hs,Ws,3) uint8
            Hs, Ws, _ = y_est_u8.shape

            # 3) paste original 64×64 lensless patch into center
            ys = (Hs - 64) // 2
            xs = (Ws - 64) // 2
            y_est_u8[ys:ys+64, xs:xs+64, :] = patch_hwc_u8

            # 4) send to reconstruction (lensless 128×128 -> space domain)
            rec_u8 = reconstruct_an_image(y_est_u8)                                # (H,W,3) uint8 (0..255)

            # 5) to torch CHW in [0,1]
            rec = rec_u8.astype(np.float32) / 255.0
            rec_chw = torch.from_numpy(np.transpose(rec, (2, 0, 1))).contiguous()
            outs.append(rec_chw)

    return torch.stack(outs)

if __name__ == "__main__":
    # test
    input_path = "../img.png"
    output_path = "input_lensless.png"
    # 1) Read image
    img = Image.open(input_path).convert("RGB")
    # 2) Resize to 64×64
    img = img.resize((64, 64), Image.BICUBIC)
    # 3) Convert to lensless
    lensless_img = convert_image_to_lensless(img)
    # 4) Save output
    Image.fromarray(lensless_img).save(output_path)
    print(f"Saved lensless image to: {output_path}")