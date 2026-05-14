"""Stable import path for composition planning (facade over the intelligence layer).

Production planning lives in ``composition_intelligence_layer.build_composition_plan``. This module
re-exports ``CompositionPlan`` and ``plan_composition`` so older imports keep working. The
``interpret_mood`` helper is UI-only (pixels → one prose line); it does not affect MIDI.
"""

from __future__ import annotations

from typing import Mapping

from src.composition_intelligence_layer import (
    CompositionPlan,
    build_composition_plan,
)
from src.image_analysis import ImageFeatures
from src.scene_composition_cue import SceneCompositionCue

__all__ = ["CompositionPlan", "interpret_mood", "plan_composition"]


def interpret_mood(features: ImageFeatures) -> str:
    """Return a short human-readable mood line from brightness, contrast, and edge score (for UI only)."""
    b, c, e = features.brightness, features.contrast, features.edge_detail_score
    if b > 0.65 and c < 0.35:
        return "Open and airy — calm, optimistic."
    if b < 0.35 and c > 0.45:
        return "Moody and dramatic — tense, cinematic."
    if e > 0.55 and c > 0.4:
        return "Energetic and intricate — busy, alert."
    if b > 0.5 and e < 0.35:
        return "Warm and relaxed — soft, contemplative."
    if c > 0.5:
        return "Bold and defined — confident, striking."
    return "Balanced and neutral — steady, versatile."


def plan_composition(
    features: ImageFeatures,
    ml_summary: Mapping[str, float] | None = None,
    scene_cue: SceneCompositionCue | None = None,
) -> CompositionPlan:
    """Delegate to the rule-based composition intelligence layer."""
    return build_composition_plan(features, ml_summary, scene_cue)
