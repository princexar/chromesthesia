''' Perception (pixels + ML + optional CLIP cue) → structured ``CompositionPlan`` only.

This module does **not** output MIDI notes or audio. ``music_generator.py`` turns ``CompositionPlan``
into a full MIDI piece. A future trained ``composition_model`` may predict the same plan shape.
'''

# Rule-based composition intelligence (production planner).

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, Mapping

from src.image_analysis import ImageFeatures
from src.scene_composition_cue import SceneCompositionCue

_ML_TEMPO_WEIGHT = 0.45
_ML_RANGE_WEIGHT = 0.5

Mood = Literal[
    "joyful",
    "peaceful",
    "melancholic",
    "eerie",
    "chaotic",
    "cinematic",
    "nostalgic",
    "tense",
    "neutral",
]

EnergyLevel = Literal["low", "medium", "high"]
TextureDensity = Literal["sparse", "balanced", "dense"]
ScaleType = Literal["intimate", "moderate", "expansive"]
StructureType = Literal["AABA", "ABAB", "through-composed"]

_KEYS = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


@dataclass(frozen=True)
class CompositionIntent:
    mood: Mood
    energy: EnergyLevel
    atmosphere: str
    perceived_scale: ScaleType
    texture_density: TextureDensity


@dataclass(frozen=True)
class CompositionPlan:
    """Structured targets for the MIDI engine—no note lists, no audio buffers."""

    key: str
    mode: Literal["major", "minor"]
    tempo_bpm: int
    duration_seconds: float
    structure: StructureType
    texture_density: TextureDensity
    note_density: float
    melodic_range: tuple[int, int]
    harmonic_complexity: float
    instrument_feel: str
    mood: str
    explanation: list[str]


def _hue_from_hex(hex_color: str) -> float:
    """Hue in [0, 1) from #RRGGBB."""
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


def _mood_from_clip_label(label: str) -> Mood:
    """Map CLIP ``top_clip_label`` onto the internal mood vocabulary."""
    s = label.lower()
    if "joyful" in s or "bright" in s:
        return "joyful"
    if "peaceful" in s or "dreamy" in s:
        return "peaceful"
    if "melancholic" in s or "lonely" in s:
        return "melancholic"
    if "eerie" in s or "mysterious" in s:
        return "eerie"
    if "chaotic" in s or "energetic" in s:
        return "chaotic"
    if "cinematic" in s or "dramatic" in s:
        return "cinematic"
    if "nostalgic" in s or "warm" in s:
        return "nostalgic"
    if "cold" in s or "distant" in s:
        return "tense"
    return "neutral"


def _energy_level(s: str) -> EnergyLevel:
    x = s.strip().lower()
    if x in ("low", "medium", "high"):
        return x  # type: ignore[return-value]
    return "medium"


def _scale_type(s: str) -> ScaleType:
    x = s.strip().lower()
    if x in ("intimate", "moderate", "expansive"):
        return x  # type: ignore[return-value]
    return "moderate"


def _texture_type(s: str) -> TextureDensity:
    x = s.strip().lower()
    if x in ("sparse", "balanced", "dense"):
        return x  # type: ignore[return-value]
    return "balanced"


def build_composition_intent(
    features: ImageFeatures,
    scene_cue: SceneCompositionCue | None = None,
) -> CompositionIntent:
    """Blend pixels with optional CLIP scene cue (rule-based)."""
    if scene_cue is not None:
        mood = _mood_from_clip_label(scene_cue.top_clip_label)
        energy = _energy_level(scene_cue.energy_level)
        atmosphere = scene_cue.atmosphere
        scale = _scale_type(scene_cue.perceived_scale)
        texture = _texture_type(scene_cue.texture_density)
        if features.brightness < 0.28 and energy == "high":
            energy = "medium"
        if features.edge_detail_score > 0.65 and texture == "sparse":
            texture = "balanced"
        return CompositionIntent(
            mood=mood,
            energy=energy,
            atmosphere=atmosphere,
            perceived_scale=scale,
            texture_density=texture,
        )

    if features.brightness > 0.6 and features.contrast < 0.4:
        mood, energy, atm = "joyful", "medium", "bright, open (pixels only)"
    elif features.brightness < 0.35:
        mood, energy, atm = "melancholic", "low", "shadow-weighted (pixels only)"
    elif features.edge_detail_score > 0.55:
        mood, energy, atm = "chaotic", "high", "busy detail (pixels only)"
    else:
        mood, energy, atm = "neutral", "medium", "balanced (pixels only)"

    scale: ScaleType = "intimate" if features.contrast < 0.35 else "expansive" if features.contrast > 0.55 else "moderate"
    texture: TextureDensity = "sparse" if features.edge_detail_score < 0.33 else "dense" if features.edge_detail_score > 0.66 else "balanced"
    return CompositionIntent(
        mood=mood,
        energy=energy,
        atmosphere=atm,
        perceived_scale=scale,
        texture_density=texture,
    )


def _infer_structure(intent: CompositionIntent, scene_cue: SceneCompositionCue | None) -> StructureType:
    if intent.mood == "eerie":
        return "through-composed"
    if intent.mood == "chaotic":
        return "ABAB"
    if intent.mood == "cinematic" and intent.perceived_scale == "expansive":
        return "AABA"
    if intent.mood in ("peaceful", "melancholic", "nostalgic"):
        return "AABA"
    if scene_cue is not None and scene_cue.duration_band == "long" and intent.perceived_scale == "expansive":
        return "AABA"
    return "AABA"


def _tempo_bpm(features: ImageFeatures, ml_summary: Mapping[str, float] | None, intent: CompositionIntent) -> int:
    base = int(72 + features.brightness * 56 + features.edge_detail_score * 24)
    base = max(60, min(160, base))
    e_ml = 0.5 if ml_summary is None else float(ml_summary["energy_score"])
    tempo_ml = int(68 + e_ml * 72)
    tempo_ml = max(60, min(160, tempo_ml))
    t = int(round((1.0 - _ML_TEMPO_WEIGHT) * base + _ML_TEMPO_WEIGHT * tempo_ml))
    if intent.energy == "low":
        t -= 8
    elif intent.energy == "high":
        t += 12
    return max(60, min(160, t))


def _duration_seconds(features: ImageFeatures, scene_cue: SceneCompositionCue | None) -> float:
    if scene_cue is not None:
        return float(scene_cue.target_duration_sec)
    return float(max(30, min(120, 55 + features.brightness * 40)))


def _melodic_range(
    features: ImageFeatures,
    intent: CompositionIntent,
    ml_summary: Mapping[str, float] | None,
) -> tuple[int, int]:
    half_span_image = 10 + int(features.edge_detail_score * 8 + features.contrast * 6)
    half_span_image = max(8, min(26, half_span_image))
    if ml_summary is None:
        half = half_span_image
    else:
        mmax = float(ml_summary["max_activation"])
        ml_peak = (math.tanh(mmax / 2.5) + 1.0) / 2.0
        half_span_ml = 8 + int(ml_peak * 14)
        half_span_ml = max(8, min(24, half_span_ml))
        half = int(round((1.0 - _ML_RANGE_WEIGHT) * half_span_image + _ML_RANGE_WEIGHT * half_span_ml))
    half = max(8, min(28, half))
    if intent.perceived_scale == "intimate":
        half = max(6, half - 3)
    elif intent.perceived_scale == "expansive":
        half = min(30, half + 5)
    center = 69
    lo = max(36, center - half)
    hi = min(102, center + half)
    return lo, hi


def _note_density(
    features: ImageFeatures,
    intent: CompositionIntent,
    ml_summary: Mapping[str, float] | None,
) -> float:
    tex = {"sparse": 0.25, "balanced": 0.5, "dense": 0.78}[intent.texture_density]
    edge = features.edge_detail_score
    raw = 0.45 * tex + 0.55 * edge
    if ml_summary is not None:
        var = float(ml_summary["variation_score"])
        raw = 0.72 * raw + 0.28 * var
    return float(max(0.05, min(0.95, raw)))


def _harmonic_complexity_float(intent: CompositionIntent, scene_cue: SceneCompositionCue | None) -> float:
    score = 0.35
    if intent.texture_density == "dense":
        score += 0.28
    if intent.mood in ("cinematic", "joyful", "chaotic"):
        score += 0.15
    if scene_cue is not None and scene_cue.duration_band == "long":
        score += 0.12
    return float(max(0.0, min(1.0, score)))


def _instrument_feel(intent: CompositionIntent) -> str:
    """Short prose hint for GM program choice in the generator (not a sample library path)."""
    if intent.mood in ("peaceful", "melancholic", "eerie") and intent.texture_density == "sparse":
        return "soft solo piano, minimal pad optional"
    if intent.mood == "cinematic" and intent.perceived_scale == "expansive":
        return "layered strings with acoustic piano lead"
    if intent.mood == "chaotic":
        return "bright acoustic piano, punchy bass foundation"
    if intent.mood == "nostalgic":
        return "warm electric piano, mellow electric bass"
    if intent.mood == "joyful":
        return "acoustic grand piano, light pizzicato strings"
    return "acoustic grand piano lead, acoustic bass foundation"


def build_composition_plan(
    features: ImageFeatures,
    ml_summary: Mapping[str, float] | None = None,
    scene_cue: SceneCompositionCue | None = None,
    intent: CompositionIntent | None = None,
) -> CompositionPlan:
    """Rule-based planner: perception → ``CompositionPlan`` (no notes, no audio)."""
    cue = scene_cue if scene_cue is not None else None
    intent = intent or build_composition_intent(features, cue)

    hue = _hue_from_hex(features.dominant_color_hex)
    key = _KEYS[int(hue * 12) % 12]
    warm = hue < 0.08 or hue > 0.92 or (0.08 < hue < 0.2)
    mode: Literal["major", "minor"] = (
        "major" if (features.brightness > 0.45 and warm) or features.brightness > 0.72 else "minor"
    )

    tempo_bpm = _tempo_bpm(features, ml_summary, intent)
    duration_seconds = _duration_seconds(features, cue)
    structure = _infer_structure(intent, cue)
    texture_density = intent.texture_density
    note_density = _note_density(features, intent, ml_summary)
    melodic_range = _melodic_range(features, intent, ml_summary)
    harmonic_complexity = _harmonic_complexity_float(intent, cue)
    instrument_feel = _instrument_feel(intent)
    mood_line = f"{intent.mood}: {intent.atmosphere}"

    ml_e = "none" if ml_summary is None else f"{float(ml_summary['energy_score']):.2f}"
    explanation = [
        f"Key {key} from dominant-color hue; mode {mode} from brightness/warmth.",
        f"Tempo {tempo_bpm} BPM from pixels, ML energy ({ml_e}), and intent energy {intent.energy}.",
        f"Duration ~{duration_seconds:.0f}s from {'scene cue' if cue else 'pixel fallback'}.",
        f"Structure {structure} from mood/scale/duration heuristics.",
        f"Texture {texture_density}; note_density={note_density:.2f}; harmonic_complexity={harmonic_complexity:.2f}.",
        f"MIDI span {melodic_range[0]}–{melodic_range[1]} (pixels + ResNet peak blend).",
        f"Mood line: {mood_line!r}; instruments: {instrument_feel!r}.",
    ]

    return CompositionPlan(
        key=key,
        mode=mode,
        tempo_bpm=tempo_bpm,
        duration_seconds=duration_seconds,
        structure=structure,
        texture_density=texture_density,
        note_density=note_density,
        melodic_range=melodic_range,
        harmonic_complexity=harmonic_complexity,
        instrument_feel=instrument_feel,
        mood=mood_line,
        explanation=explanation,
    )
