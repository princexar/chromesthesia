"""ML-oriented feature extraction for Chromesthesia (ResNet18 image embeddings).

Exports:
- ``extract_resnet18_embedding`` — PIL image → 512-D float vector.
- ``summarize_embedding`` — 512-D vector → a small dict of UI-friendly scalars.

This module does not perform music mapping or MIDI I/O; callers use the outputs downstream.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision.models import ResNet18_Weights, resnet18

# ---------------------------------------------------------------------------
# Why globals? Loading a pretrained model is slow and downloads weights once.
# We keep a single model + preprocess in memory for the whole Python process.
# ---------------------------------------------------------------------------
_model: nn.Module | None = None
_preprocess = None


def _pick_device() -> torch.device:
    """Use a GPU if PyTorch can see one; otherwise CPU (works everywhere)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _get_model_and_preprocess():
    """Load ResNet18 once, strip the classifier, attach ImageNet preprocess, cache on module globals.

    Returns:
        Tuple of ``(model, preprocess)`` where ``model`` outputs 512-D vectors and ``preprocess``
        is the torchvision transform chain matching the loaded weights.
    """
    global _model, _preprocess
    if _model is not None and _preprocess is not None:
        return _model, _preprocess

    # 1) Choose official ImageNet weights for ResNet18.
    #    `DEFAULT` points at the best available pretrained weights for this architecture.
    weights = ResNet18_Weights.DEFAULT

    # 2) Load the full classification network (conv backbone + pooling + linear "head").
    model = resnet18(weights=weights)

    # 3) Remove the final classification layer.
    #    ResNet18 ends with a Linear layer `fc` that maps 512 features -> 1000 ImageNet classes.
    #    Replacing it with `Identity` means the network outputs those 512 features directly
    #    (the "embedding" we want), instead of class probabilities.
    model.fc = nn.Identity()

    # 4) Freeze weights: we are not training, only running a fixed feature extractor.
    for parameter in model.parameters():
        parameter.requires_grad = False

    # 5) Set "eval" mode (disables dropout / uses batchnorm in inference mode).
    model.eval()

    # 6) Move the model to CPU or GPU so tensor math runs on the right device.
    device = _pick_device()
    model.to(device)

    # 7) Get the exact preprocessing (resize, crop, normalize) that matches these weights.
    #    Using `weights.transforms()` is safer than hand-copying numbers from a tutorial.
    preprocess = weights.transforms()

    _model, _preprocess = model, preprocess
    return _model, _preprocess


def extract_resnet18_embedding(image: Image.Image) -> np.ndarray:
    """Run a pretrained ResNet18 on one PIL image and return a 1-D NumPy embedding.

    Steps in plain language:
    1. Make sure the image is RGB (ResNet expects 3 color channels).
    2. Resize / crop / normalize with the same rules the model saw during pretraining.
    3. Add a "batch" dimension: the network expects shape [batch, channels, height, width],
       even when we only pass one image.
    4. Forward pass with gradients turned off (inference only—saves memory).
    5. Convert the PyTorch tensor to a NumPy vector you can use later (e.g. for mapping).
    """
    if not isinstance(image, Image.Image):
        raise TypeError("image must be a PIL.Image.Image")

    # --- Image prep ------------------------------------------------------------
    # ResNet was trained on color photos; convert grayscale/RGBA/etc. to RGB once here.
    if image.mode != "RGB":
        image = image.convert("RGB")

    model, preprocess = _get_model_and_preprocess()
    device = next(model.parameters()).device

    # --- Torch preprocess ------------------------------------------------------
    # `preprocess` returns a tensor shaped [3, 224, 224] for standard ResNet inputs.
    tensor_chw = preprocess(image)

    # Add batch dimension: [3, H, W] -> [1, 3, H, W]
    batch = tensor_chw.unsqueeze(0).to(device)

    # --- Forward pass ----------------------------------------------------------
    # `inference_mode()` is like `no_grad()` but tells PyTorch we will never need
    # training-style autograd bookkeeping for this call (slightly leaner).
    with torch.inference_mode():
        embedding_b512 = model(batch)

    # --- Back to NumPy ---------------------------------------------------------
    # Squeeze removes the batch dimension: shape [1, 512] -> [512]
    vector = embedding_b512.squeeze(0).detach().cpu().numpy().astype(np.float32, copy=False)
    return vector


# ResNet18 backbone outputs 512 floats after the classifier is removed.
_RESNET18_EMBEDDING_DIM = 512


def summarize_embedding(embedding: np.ndarray) -> dict[str, float]:
    """Turn a long embedding vector into a handful of easy-to-read numbers.

    **Important:** these values are *summary signals*—simple statistics computed from the
    512 floats. They describe how the activations behave as a whole (typical size, spread,
    overall "energy"). They are **not** semantic labels like "sunset" or "sad"; the network
    was trained for ImageNet object recognition, and we are only reading coarse patterns from
    its internal representation.

    Args:
        embedding: 1-D array of length 512 (e.g. output of ``extract_resnet18_embedding``).

    Returns:
        Dictionary with mean/std/min/max and two extra scores squashed into 0–1 for UI
        sliders or heuristics later on.
    """
    arr = np.asarray(embedding, dtype=np.float64).reshape(-1)
    if arr.ndim != 1 or arr.size != _RESNET18_EMBEDDING_DIM:
        raise ValueError(
            f"embedding must be a 1-D array of length {_RESNET18_EMBEDDING_DIM}, got shape {arr.shape}"
        )

    # --- Basic moments (first glance at the vector) ----------------------------
    mean_activation = float(np.mean(arr))
    activation_std = float(np.std(arr, ddof=0))  # plain "spread" around the mean
    max_activation = float(np.max(arr))
    min_activation = float(np.min(arr))

    # --- Energy score (0–1): how strong activations are on average ------------
    # Mean of squares = average squared value. Large positives/negatives both increase it.
    # It is not "brightness of the photo"; it is only a property of these 512 numbers.
    mean_squared = float(np.mean(arr**2))
    # Smooth squash: grows toward 1 as mean_squared grows, stays in (0, 1).
    energy_score = float(1.0 - np.exp(-mean_squared))

    # --- Variation score (0–1): how much values differ across dimensions -------
    # Standard deviation already measures spread; we squash it the same way for a 0–1 dial.
    # High std means the vector is "spiky" across dimensions; low std means values are similar.
    # Again, this is not a mood or scene label—just a shape summary of the embedding.
    variation_score = float(1.0 - np.exp(-activation_std))

    return {
        "mean_activation": mean_activation,
        "activation_std": activation_std,
        "max_activation": max_activation,
        "min_activation": min_activation,
        "energy_score": energy_score,
        "variation_score": variation_score,
    }
