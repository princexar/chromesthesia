"""Map visual features (and optional ML summaries) into a concrete composition plan.

Turns ``ImageFeatures`` plus optional ResNet-derived scalars and a ``SceneCompositionCue`` into
key, mode, tempo, rhythm, melody text, MIDI span, length, form, and harmony density for the generator.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping

from src.image_analysis import ImageFeatures
from src.scene_composition_cue import NEUTRAL_SCENE_CUE, SceneCompositionCue

# How strongly the ResNet embedding summary nudges tempo / rhythm / note span (0 = ignore ML, 1 = only ML).
_ML_TEMPO_WEIGHT = 0.45
_ML_RHYTHM_WEIGHT = 0.45
_ML_RANGE_WEIGHT = 0.5


@dataclass(frozen=True)
class CompositionPlan:
    """All parameters the MIDI engine needs: tonality, tempo, rhythm, span, length, form, harmony."""

    key: str
    tempo_bpm: int
    mode: str
    rhythm_complexity: str
    melody_description: str
    midi_low: int
    midi_high: int
    target_duration_sec: float
    measures: int
    section_form: str
    texture_density: str
    harmonic_complexity: str


def interpret_mood(features: ImageFeatures) -> str:
    """Return a short human-readable mood line from brightness, contrast, and edge score (for UI only)."""
    b, c, e = features.brightness, features.contrast, features.edge_detail_score
    # Simple heuristic labels
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


def _hue_from_hex(hex_color: str) -> float:
    """Convert #RRGGBB to a hue in [0, 1) for key selection (0 = red wedge in HSV-style wheel)."""
    h = hex_color.lstrip("#")
    if len(h) != 6:
        return 0.0
    r = int(h[0:2], 16) / 255.0
    g = int(h[2:4], 16) / 255.0
    b = int(h[4:6], 16) / 255.0
    mx, mn = max(r, g, b), min(r, g, b)
    d = mx - mn
    if d < 1e-6:
        return 0.0
    if mx == r:
        h_raw = ((g - b) / d) % 6
    elif mx == g:
        h_raw = (b - r) / d + 2
    else:
        h_raw = (r - g) / d + 4
    return (h_raw / 6.0) % 1.0


_KEYS = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def _rhythm_tier_from_unit_interval(x: float) -> int:
    """Map a 0–1 style score to Low=0 / Moderate=1 / High=2 (same cutoffs as edge detail)."""
    if x < 0.33:
        return 0
    if x < 0.66:
        return 1
    return 2


def _tier_to_label(tier: int) -> str:
    """Map integer tier 0/1/2 to rhythm complexity labels used by the MIDI generator."""
    return ("Low", "Moderate", "High")[max(0, min(2, tier))]


def _pick_section_form(cue: SceneCompositionCue) -> str:
    """Pick AABA, ABAB, or through-composed layout from CLIP top label and perceived scale."""
    tl = cue.top_clip_label
    if "mysterious" in tl or "eerie" in tl:
        return "through"
    if "chaotic" in tl:
        return "ABAB"
    if "cinematic" in tl and cue.perceived_scale == "expansive":
        return "AABA"
    if "peaceful" in tl or "melancholic" in tl or "dreamy" in tl:
        return "AABA"
    return "AABA"


def _harmonic_complexity(cue: SceneCompositionCue) -> str:
    """Map scene texture + length to simple / moderate / rich chord layers in the generator."""
    if cue.texture_density == "dense" and cue.duration_band == "long":
        return "rich"
    if cue.texture_density == "sparse" and cue.energy_level == "low":
        return "simple"
    return "moderate"


def plan_composition(
    features: ImageFeatures,
    ml_summary: Mapping[str, float] | None = None,
    scene_cue: SceneCompositionCue | None = None,
) -> CompositionPlan:
    """Blend pixels, ResNet summaries, and CLIP-derived scene cues into one ``CompositionPlan``."""
    cue = scene_cue or NEUTRAL_SCENE_CUE

    hue = _hue_from_hex(features.dominant_color_hex)
    key = _KEYS[int(hue * 12) % 12]

    # --- Tempo: image + ML + scene energy --------------------------------------
    tempo_image = int(72 + features.brightness * 56 + features.edge_detail_score * 24)
    tempo_image = max(60, min(160, tempo_image))
    energy = 0.5 if ml_summary is None else float(ml_summary["energy_score"])
    tempo_ml = int(68 + energy * 72)
    tempo_ml = max(60, min(160, tempo_ml))
    tempo = int(round((1.0 - _ML_TEMPO_WEIGHT) * tempo_image + _ML_TEMPO_WEIGHT * tempo_ml))
    tempo = max(60, min(160, tempo))
    if cue.energy_level == "low":
        tempo = max(60, tempo - 8)
    elif cue.energy_level == "high":
        tempo = min(160, tempo + 12)

    warm = hue < 0.08 or hue > 0.92 or (0.08 < hue < 0.2)
    mode = "Major" if (features.brightness > 0.45 and warm) or features.brightness > 0.72 else "Minor"

    # --- Rhythm tier: image + ML + scene texture/energy ------------------------
    tier_image = _rhythm_tier_from_unit_interval(features.edge_detail_score)
    variation = 0.5 if ml_summary is None else float(ml_summary["variation_score"])
    tier_ml = _rhythm_tier_from_unit_interval(variation)
    tier_blend = (1.0 - _ML_RHYTHM_WEIGHT) * tier_image + _ML_RHYTHM_WEIGHT * tier_ml
    rhythm_tier = int(round(tier_blend))
    rhythm_tier = max(0, min(2, rhythm_tier))
    if cue.texture_density == "dense" and cue.energy_level == "high":
        rhythm_tier = min(2, rhythm_tier + 1)
    if cue.texture_density == "sparse" and cue.energy_level == "low":
        rhythm_tier = max(0, rhythm_tier - 1)
    rhythm = _tier_to_label(rhythm_tier)

    # --- Note span: image + ML + perceived scale --------------------------------
    half_span_image = 10 + int(features.edge_detail_score * 8 + features.contrast * 6)
    half_span_image = max(8, min(26, half_span_image))
    if ml_summary is None:
        half_span = half_span_image
    else:
        mmax = float(ml_summary["max_activation"])
        ml_peak = (math.tanh(mmax / 2.5) + 1.0) / 2.0
        half_span_ml = 8 + int(ml_peak * 14)
        half_span_ml = max(8, min(24, half_span_ml))
        half_span = int(
            round((1.0 - _ML_RANGE_WEIGHT) * half_span_image + _ML_RANGE_WEIGHT * half_span_ml)
        )
    half_span = max(8, min(28, half_span))
    if cue.perceived_scale == "intimate":
        half_span = max(6, half_span - 3)
    elif cue.perceived_scale == "expansive":
        half_span = min(30, half_span + 5)

    center = 69
    midi_low = max(36, center - half_span)
    midi_high = min(102, center + half_span)

    # --- Length & form from scene cue ------------------------------------------
    target_duration_sec = float(cue.target_duration_sec)
    quarters_total = target_duration_sec * (tempo / 60.0)
    measures = int(round(quarters_total / 4.0))
    measures = max(12, min(96, measures))
    section_form = _pick_section_form(cue)
    harmonic_complexity = _harmonic_complexity(cue)

    if mode == "Major":
        melody = f"Stepwise motion in {key} major with brighter peaks on strong beats."
    else:
        melody = f"Minor arpeggios and small intervals in {key} minor, leaning melancholic."

    if features.contrast > 0.55:
        melody += " Occasional wide leaps echo high visual contrast."

    if ml_summary is not None:
        melody += (
            f" Neural cues: energy {energy:.2f} steers tempo, variation {variation:.2f} steers rhythm, "
            f"peak activation {float(ml_summary['max_activation']):.2f} widens or tightens the pitch window "
            f"(MIDI {midi_low}–{midi_high})."
        )

    melody += (
        f" Scene: {cue.emotional_mood} — {cue.atmosphere}. "
        f"Scale {cue.perceived_scale}, texture {cue.texture_density}, ~{target_duration_sec:.0f}s "
        f"({cue.duration_band}), form **{section_form}**, harmony **{harmonic_complexity}**."
    )

    return CompositionPlan(
        key=key,
        tempo_bpm=tempo,
        mode=mode,
        rhythm_complexity=rhythm,
        melody_description=melody,
        midi_low=midi_low,
        midi_high=midi_high,
        target_duration_sec=target_duration_sec,
        measures=measures,
        section_form=section_form,
        texture_density=cue.texture_density,
        harmonic_complexity=harmonic_complexity,
    )
