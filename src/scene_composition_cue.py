"""Turn CLIP prompt rankings into structured composition cues (duration, scale, texture, energy).

This module does **not** load models; it calls ``semantic_interpreter.clip_rank_texts``. The cues
are consumed by ``music_mapping.plan_composition`` and ``music_generator.generate_midi``.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from PIL import Image

from src.semantic_interpreter import clip_rank_texts, interpret_scene

# --- CLIP prompt sets for single-axis softmax (winner drives that axis) -------------------------

_SCALE_PROMPTS: tuple[str, ...] = (
    "a tight intimate close-up of a small subject",
    "an everyday scene at a normal comfortable distance",
    "an expansive wide open vista or cinematic wide shot",
)

_DURATION_PROMPTS: tuple[str, ...] = (
    "a brief fleeting moment that passes quickly",
    "a sustained episode of moderate length",
    "a long unfolding journey with time to develop",
)

_TEXTURE_PROMPTS: tuple[str, ...] = (
    "lots of empty space and very few visual elements",
    "a balanced amount of detail and open space",
    "dense busy intricate detail filling the frame",
)

# Map each default atmosphere label → baseline energy / atmosphere copy / default duration & texture hints.
# CLIP axis winners still override scale / duration / texture bands where they disagree strongly.
_LABEL_BASE: dict[str, dict[str, str]] = {
    "a joyful bright scene": {
        "energy": "high",
        "atmosphere": "uplifted, sunlit, outward-looking",
        "duration_hint": "medium",
        "texture_hint": "balanced",
    },
    "a melancholic lonely scene": {
        "energy": "low",
        "atmosphere": "introspective, solitary, soft-edged",
        "duration_hint": "long",
        "texture_hint": "sparse",
    },
    "a peaceful dreamy scene": {
        "energy": "low",
        "atmosphere": "calm, floating, gentle",
        "duration_hint": "long",
        "texture_hint": "sparse",
    },
    "a mysterious eerie scene": {
        "energy": "low",
        "atmosphere": "uncanny, still, withheld",
        "duration_hint": "long",
        "texture_hint": "sparse",
    },
    "a chaotic energetic scene": {
        "energy": "high",
        "atmosphere": "restless, collisions, high motion",
        "duration_hint": "medium",
        "texture_hint": "dense",
    },
    "a warm nostalgic scene": {
        "energy": "medium",
        "atmosphere": "memory-toned, close, sentimental",
        "duration_hint": "medium",
        "texture_hint": "balanced",
    },
    "a cold distant scene": {
        "energy": "low",
        "atmosphere": "remote, austere, wide gaps",
        "duration_hint": "medium",
        "texture_hint": "sparse",
    },
    "a cinematic dramatic scene": {
        "energy": "high",
        "atmosphere": "heightened stakes, spotlight, narrative arc",
        "duration_hint": "long",
        "texture_hint": "balanced",
    },
}

_DEFAULT_BASE = {
    "energy": "medium",
    "atmosphere": "neutral, mixed cues",
    "duration_hint": "medium",
    "texture_hint": "balanced",
}


def _winner_index(rankings: list[tuple[str, float]], choices: tuple[str, ...]) -> int:
    """Index of the highest-scoring choice among ``choices`` (same strings as in rankings)."""
    scores = {lab: sc for lab, sc in rankings}
    best_i = 0
    best_s = -1.0
    for i, c in enumerate(choices):
        s = scores.get(c, 0.0)
        if s > best_s:
            best_s = s
            best_i = i
    return best_i


def _image_rng(image: Image.Image) -> random.Random:
    """Reproducible RNG from image bytes (first row of pixels + size)."""
    raw = image.tobytes()[:2048]
    seed = hash((image.size, raw)) % (2**31)
    return random.Random(seed)


@dataclass(frozen=True)
class SceneCompositionCue:
    """Structured scene readout used by the mapper and MIDI engine (all local heuristics + CLIP)."""

    emotional_mood: str
    energy_level: str  # low | medium | high
    atmosphere: str
    perceived_scale: str  # intimate | moderate | expansive
    duration_band: str  # short | medium | long
    texture_density: str  # sparse | balanced | dense
    target_duration_sec: float
    top_clip_label: str
    top_clip_score: float


def build_scene_composition_cue(
    image: Image.Image,
    mood_rankings: list[tuple[str, float]] | None = None,
) -> SceneCompositionCue:
    """Combine atmosphere ranking with dedicated CLIP passes for scale, duration feel, and texture.

    Pass ``mood_rankings`` from a prior ``interpret_scene`` call to avoid running that CLIP pass twice.
    """
    rng = _image_rng(image)

    mood_rank = mood_rankings if mood_rankings is not None else interpret_scene(image)
    if not mood_rank:
        base = _DEFAULT_BASE
        top_label, top_score = "unknown", 0.0
        mood_title = "Indistinct scene"
    else:
        top_label, top_score = mood_rank[0]
        base = _LABEL_BASE.get(top_label, _DEFAULT_BASE)
        mood_title = top_label.replace("a ", "").replace(" scene", "").strip().title()

    scale_rank = clip_rank_texts(image, _SCALE_PROMPTS)
    dur_rank = clip_rank_texts(image, _DURATION_PROMPTS)
    tex_rank = clip_rank_texts(image, _TEXTURE_PROMPTS)

    scale_i = _winner_index(scale_rank, _SCALE_PROMPTS)
    dur_i = _winner_index(dur_rank, _DURATION_PROMPTS)
    tex_i = _winner_index(tex_rank, _TEXTURE_PROMPTS)

    perceived_scale = ("intimate", "moderate", "expansive")[scale_i]

    # Map CLIP duration axis to bands (brief / sustained / long → short / medium / long).
    clip_duration_band = ("short", "medium", "long")[dur_i]

    # Blend table hint with CLIP duration winner (table nudges if CLIP is ambiguous).
    hint_band = base["duration_hint"]
    if clip_duration_band != hint_band:
        # Prefer CLIP when second-ranked CLIP score is not too close (simple confidence).
        top_d = dur_rank[0][1]
        second_d = dur_rank[1][1] if len(dur_rank) > 1 else 0.0
        if top_d - second_d > 0.08:
            duration_band = clip_duration_band
        else:
            duration_band = hint_band
    else:
        duration_band = clip_duration_band

    # Texture: prefer CLIP if confident, else table hint.
    top_t = tex_rank[0][1]
    second_t = tex_rank[1][1] if len(tex_rank) > 1 else 0.0
    if top_t - second_t > 0.1:
        texture_density = ("sparse", "balanced", "dense")[tex_i]
    else:
        texture_map = {"sparse": 0, "balanced": 1, "dense": 2}
        inv = ["sparse", "balanced", "dense"]
        blended = round(0.5 * tex_i + 0.5 * texture_map.get(base["texture_hint"], 1))
        texture_density = inv[max(0, min(2, int(blended)))]

    energy_level = base["energy"]
    atmosphere = base["atmosphere"]

    # Second-ranked mood can nudge energy slightly.
    if len(mood_rank) > 1 and mood_rank[1][1] > 0.18:
        runner = mood_rank[1][0]
        if "chaotic" in runner or "joyful" in runner or "cinematic" in runner:
            if energy_level == "low":
                energy_level = "medium"
            elif energy_level == "medium":
                energy_level = "high"
        if "peaceful" in runner or "melancholic" in runner or "eerie" in runner:
            if energy_level == "high":
                energy_level = "medium"
            elif energy_level == "medium":
                energy_level = "low"

    # Target length in seconds (bands from product brief).
    if duration_band == "short":
        lo, hi = 30.0, 45.0
    elif duration_band == "medium":
        lo, hi = 60.0, 90.0
    else:
        lo, hi = 90.0, 180.0
    target_duration_sec = float(rng.uniform(lo, hi))

    emotional_mood = f"{mood_title} ({top_score * 100:.0f}% CLIP match)"

    return SceneCompositionCue(
        emotional_mood=emotional_mood,
        energy_level=energy_level,
        atmosphere=atmosphere,
        perceived_scale=perceived_scale,
        duration_band=duration_band,
        texture_density=texture_density,
        target_duration_sec=target_duration_sec,
        top_clip_label=top_label,
        top_clip_score=float(top_score),
    )


# Used when callers skip CLIP (e.g. unit tests).
NEUTRAL_SCENE_CUE = SceneCompositionCue(
    emotional_mood="Neutral (no scene cue)",
    energy_level="medium",
    atmosphere="neutral",
    perceived_scale="moderate",
    duration_band="medium",
    texture_density="balanced",
    target_duration_sec=72.0,
    top_clip_label="",
    top_clip_score=0.0,
)
