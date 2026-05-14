"""Extract simple, explainable visual statistics from PIL images.

These features drive part of the music-mapping layer; they are computed from pixels only
(no deep learning in this module).
"""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class ImageFeatures:
    """Bundle of normalized scalar cues computed from one RGB image."""

    brightness: float  # 0–1 mean luminance
    dominant_color_hex: str
    contrast: float  # 0–1 normalized std of luminance
    edge_detail_score: float  # 0–1 from Laplacian variance


def _to_rgb_array(image: Image.Image) -> np.ndarray:
    """Return the image as an H×W×3 float32 array in RGB order (converts mode if needed)."""
    if image.mode != "RGB":
        image = image.convert("RGB")
    return np.asarray(image, dtype=np.float32)


def _dominant_color_hex(rgb: np.ndarray) -> str:
    """Estimate the most common color after downsampling and coarse quantization; returns #RRGGBB."""
    # Downsample for speed, quantize to reduce noise
    small = rgb[:: max(1, rgb.shape[0] // 64), :: max(1, rgb.shape[1] // 64)]
    q = (small // 32) * 32 + 16
    q = q.reshape(-1, 3).astype(np.int32)
    if q.size == 0:
        return "#808080"
    # Mode via bincount on packed RGB
    packed = (q[:, 0] << 16) | (q[:, 1] << 8) | q[:, 2]
    counts = np.bincount(packed)
    dominant = int(np.argmax(counts))
    r = (dominant >> 16) & 255
    g = (dominant >> 8) & 255
    b = dominant & 255
    return f"#{r:02x}{g:02x}{b:02x}"


def _laplacian_variance(gray: np.ndarray) -> float:
    """Compute a discrete Laplacian on grayscale pixels and map its variance to ~0–1 (sharpness proxy)."""
    g = gray.astype(np.float64)
    lap = (
        -4 * g
        + np.roll(g, 1, 0)
        + np.roll(g, -1, 0)
        + np.roll(g, 1, 1)
        + np.roll(g, -1, 1)
    )
    v = float(np.var(lap))
    # Map to 0–1 with soft cap (tuned for typical photos)
    return float(1.0 - np.exp(-v / 500.0))


def analyze_image(image: Image.Image) -> ImageFeatures:
    """Measure brightness, dominant color, contrast, and edge/detail for one PIL image."""
    rgb = _to_rgb_array(image)
    # Perceived luminance weights
    gray = 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]
    brightness = float(np.clip(gray.mean() / 255.0, 0.0, 1.0))
    std = float(np.std(gray) / 128.0)  # rough 0–1-ish
    contrast = float(np.clip(std, 0.0, 1.0))
    edge_detail_score = float(np.clip(_laplacian_variance(gray), 0.0, 1.0))
    dominant = _dominant_color_hex(rgb)
    return ImageFeatures(
        brightness=brightness,
        dominant_color_hex=dominant,
        contrast=contrast,
        edge_detail_score=edge_detail_score,
    )


def load_image_from_bytes(data: bytes) -> Image.Image:
    """Decode raw upload bytes into an RGB PIL image (raises if the bytes are not a valid image)."""
    return Image.open(BytesIO(data)).convert("RGB")
