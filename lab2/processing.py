"""Image-processing methods for Laboratory Work 2, variant 11."""

from __future__ import annotations

from typing import Literal

import numpy as np
from PIL import Image, ImageFilter

ThresholdMethod = Literal["mean", "gaussian"]
Foreground = Literal["dark", "light"]


def _image_array(image: Image.Image) -> tuple[np.ndarray, str]:
    if image.mode in ("1", "L", "I", "F"):
        mode = "L"
        array = np.asarray(image.convert(mode), dtype=np.uint8)
    else:
        mode = "RGB"
        array = np.asarray(image.convert(mode), dtype=np.uint8)
    return array, mode


def grayscale_histogram_percent(image: Image.Image) -> np.ndarray:
    """Return the 256-bin grayscale histogram as percentages of all pixels."""
    values = np.asarray(image.convert("L"), dtype=np.uint8)
    counts = np.bincount(values.ravel(), minlength=256)
    return counts.astype(np.float64) * (100 / values.size)


def point_operation(
    image: Image.Image,
    contrast: float = 1.0,
    brightness: float = 0.0,
    invert: bool = False,
) -> Image.Image:
    """Apply the point transform g = contrast * f + brightness."""
    if not np.isfinite(contrast) or contrast < 0:
        raise ValueError("Контраст должен быть конечным неотрицательным числом.")
    if not np.isfinite(brightness):
        raise ValueError("Яркость должна быть конечным числом.")

    pixels, mode = _image_array(image)
    transformed = np.clip(
        pixels.astype(np.float32) * contrast + brightness, 0, 255
    )
    if invert:
        transformed = 255 - transformed
    return Image.fromarray(np.rint(transformed).astype(np.uint8), mode=mode)


def linear_contrast(
    image: Image.Image,
    low_percentile: float = 1.0,
    high_percentile: float = 99.0,
) -> Image.Image:
    """Stretch percentile bounds to the full 0..255 range."""
    if not 0 <= low_percentile < high_percentile <= 100:
        raise ValueError("Процентили должны удовлетворять 0 <= low < high <= 100.")

    pixels, mode = _image_array(image)
    if pixels.ndim == 2:
        low = np.percentile(pixels, low_percentile)
        high = np.percentile(pixels, high_percentile)
        if high <= low:
            return Image.fromarray(pixels.copy(), mode=mode)
        scaled = (pixels.astype(np.float32) - low) * (255 / (high - low))
    else:
        low = np.percentile(pixels, low_percentile, axis=(0, 1))
        high = np.percentile(pixels, high_percentile, axis=(0, 1))
        span = high - low
        safe_span = np.where(span > 0, span, 1)
        scaled = (pixels.astype(np.float32) - low) * (255 / safe_span)
        scaled[..., span <= 0] = pixels[..., span <= 0]

    return Image.fromarray(
        np.rint(np.clip(scaled, 0, 255)).astype(np.uint8), mode=mode
    )


def local_threshold(
    image: Image.Image,
    method: ThresholdMethod = "mean",
    window_size: int = 31,
    offset: float = 10.0,
    foreground: Foreground = "dark",
) -> Image.Image:
    """Binarize against a local mean or Gaussian-weighted local mean."""
    if method not in ("mean", "gaussian"):
        raise ValueError("Метод должен быть 'mean' или 'gaussian'.")
    if (
        isinstance(window_size, bool)
        or not isinstance(window_size, int)
        or window_size < 3
        or window_size % 2 == 0
    ):
        raise ValueError("Размер окна должен быть нечётным числом не меньше 3.")
    if not np.isfinite(offset):
        raise ValueError("Смещение должно быть конечным числом.")
    if foreground not in ("dark", "light"):
        raise ValueError("Передний план должен быть 'dark' или 'light'.")

    grayscale = image.convert("L")
    if method == "mean":
        radius = (window_size - 1) / 2
        local_mean = grayscale.filter(ImageFilter.BoxBlur(radius))
    else:
        # PIL's GaussianBlur radius is its standard deviation; 3 sigma covers
        # approximately the selected odd-sized neighborhood.
        radius = (window_size - 1) / 6
        local_mean = grayscale.filter(ImageFilter.GaussianBlur(radius))

    source = np.asarray(grayscale, dtype=np.int16)
    threshold = np.asarray(local_mean, dtype=np.float32)
    if foreground == "dark":
        selected = source <= threshold - offset
    else:
        selected = source >= threshold + offset
    result = np.where(selected, 0, 255).astype(np.uint8)
    return Image.fromarray(result, mode="L")
