import cv2
import numpy as np
import scipy.fftpack as fft
import pywt
import matplotlib.pyplot as plt

def gaussian_smooth(image, kernel_size=5, sigma=1.0):
    """Apply Gaussian blur to reduce high-frequency noise."""
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)

def bilateral_filter(image, d=9, sigma_color=75, sigma_space=75):
    """Apply Bilateral Filter for edge-preserving noise reduction."""
    return cv2.bilateralFilter(image, d, sigma_color, sigma_space)

def dct_denoise(image, threshold=20):
    """Apply DCT denoising with higher threshold."""
    dct = fft.dct(fft.dct(image.T, norm='ortho').T, norm='ortho')
    dct[np.abs(dct) < threshold] = 0  # Remove small coefficients (noise)
    return fft.idct(fft.idct(dct.T, norm='ortho').T, norm='ortho')

def wavelet_denoise(image, wavelet='db1', level=1, threshold=20):
    """Apply Wavelet denoising."""
    coeffs = pywt.wavedec2(image, wavelet, level=level)
    coeffs[1:] = [(pywt.threshold(c, value=threshold, mode='soft') for c in level) for level in coeffs[1:]]
    return pywt.waverec2(coeffs, wavelet)

def hybrid_denoise(image):
    """Apply a hybrid of multiple denoising techniques."""
    smoothed = bilateral_filter(image)  # Step 1: Reduce high-frequency noise
    wavelet_cleaned = wavelet_denoise(smoothed)  # Step 2: Remove structured noise
    final_cleaned = dct_denoise(wavelet_cleaned, threshold=30)  # Step 3: DCT denoising
    return final_cleaned.astype(np.uint8)

def apply_denoising(image, method="hybrid"):
    """Process RGB image using the selected denoising method."""
    denoised_channels = [hybrid_denoise(image[:, :, i]) for i in range(3)]
    return np.stack(denoised_channels, axis=2)

def visualize_results(original, noisy, denoised, method):
    """Display images side by side for comparison."""
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    plt.title("Original Image")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(cv2.cvtColor(noisy, cv2.COLOR_BGR2RGB))
    plt.title("Noisy Image")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(cv2.cvtColor(denoised, cv2.COLOR_BGR2RGB))
    plt.title(f"Denoised ({method})")
    plt.axis("off")

    plt.show()

# ========== RUN TEST ==========
# Load your actual image instead of this placeholder
original = cv2.imread("validation_data.png")
noise = np.random.normal(0, 25, original.shape).astype(np.uint8)
noisy = cv2.add(original, noise)

# Apply hybrid denoising (Bilateral + Wavelet + DCT)
denoised_image = apply_denoising(noisy, method="hybrid")

# Display results
visualize_results(original, noisy, denoised_image, "hybrid")
