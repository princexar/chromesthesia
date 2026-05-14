"""Zero-shot scene atmosphere hints using a frozen CLIP model (Hugging Face Transformers, local only).

Compares one PIL image to English prompts and returns softmax-ranked scores. Other modules can
call ``clip_rank_texts`` with custom prompt lists for axes like scale or duration.
"""

from __future__ import annotations

from typing import Sequence

import torch
from PIL import Image

# Small, widely used CLIP checkpoint (ViT-B/32). Swap for another HF id if you prefer.
_CLIP_MODEL_ID = "openai/clip-vit-base-patch32"

# Default prompts for atmosphere / emotion (CLIP scores are only meaningful relative to this set).
DEFAULT_SCENE_LABELS: tuple[str, ...] = (
    "a joyful bright scene",
    "a melancholic lonely scene",
    "a peaceful dreamy scene",
    "a mysterious eerie scene",
    "a chaotic energetic scene",
    "a warm nostalgic scene",
    "a cold distant scene",
    "a cinematic dramatic scene",
)

_model: torch.nn.Module | None = None
_processor = None


def _device() -> torch.device:
    """Pick CUDA when available so CLIP runs on GPU; otherwise CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _get_clip():
    """Load CLIP processor + model once and keep them in module-level globals.

    Returns:
        ``(model, processor)`` ready for ``model.eval()`` inference on ``_device()``.
    """
    global _model, _processor
    if _model is not None and _processor is not None:
        return _model, _processor

    from transformers import CLIPModel, CLIPProcessor

    _processor = CLIPProcessor.from_pretrained(_CLIP_MODEL_ID)
    _model = CLIPModel.from_pretrained(_CLIP_MODEL_ID)
    _model.eval()
    for parameter in _model.parameters():
        parameter.requires_grad = False
    _model.to(_device())
    return _model, _processor


def clip_rank_texts(image: Image.Image, texts: Sequence[str]) -> list[tuple[str, float]]:
    """Run CLIP on one image vs ``texts``; return ``(text, softmax_score)`` sorted high → low.

    Same math as ``interpret_scene``, but with an arbitrary prompt list (e.g. three scale options).
    """
    labels = tuple(texts)
    if not labels:
        return []

    if not isinstance(image, Image.Image):
        raise TypeError("image must be a PIL.Image.Image")

    if image.mode != "RGB":
        image = image.convert("RGB")

    model, processor = _get_clip()
    device = next(model.parameters()).device

    inputs = processor(text=list(labels), images=image, return_tensors="pt", padding=True)
    inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

    with torch.inference_mode():
        outputs = model(**inputs)
        logits = outputs.logits_per_image[0]
        probs = logits.softmax(dim=-1).detach().float().cpu().tolist()

    ranked = sorted(zip(labels, probs), key=lambda pair: pair[1], reverse=True)
    return [(str(lab), float(score)) for lab, score in ranked]


def interpret_scene(
    image: Image.Image,
    *,
    candidate_labels: Sequence[str] | None = None,
) -> list[tuple[str, float]]:
    """Rank default (or custom) atmosphere prompts by CLIP image–text similarity.

    CLIP was trained on image–text pairs from the web; these scores measure alignment between your
    **pixels** and each **literal prompt**, not ground-truth emotions. Use them to sanity-check
    wording, not as scientific labels.

    Args:
        image: A PIL image (RGB preferred; other modes are converted).
        candidate_labels: Optional iterable of strings; defaults to ``DEFAULT_SCENE_LABELS``.

    Returns:
        ``(label, confidence)`` tuples sorted by descending confidence. Confidences are non-negative,
        sum to **1.0** across the provided candidates, and are **not** calibrated probabilities.
    """
    labels = tuple(candidate_labels) if candidate_labels is not None else DEFAULT_SCENE_LABELS
    return clip_rank_texts(image, labels)
