''' This takes brightness, contrast, edge detail, dominant color, ML energy, CLIP mood, CLIP atmosphere,
    and outputs key, mode, tempo, length, structure, texture, density, and instrument feel'''

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np

#Phase 1: Perception - Utilizes visual perception + Emotional / Semantic perception

#Phase 2: Composition planner - Visual Features + Semantic interpretation = tempo, key, mode, structure, duration, harmonic density, texture, instrument palette, melodic behavior

#Phase 3: Actual composition engine

#Phase 4 - Train composition intelligence:  perception outputs -> composition plan

Mood = Literal[
    "Happy",
    "Sad",
    "Angry",
    "Fearful",
    "Surprised",
    "Disgusted",
    "Content",
    "Excited",
    "Relaxed",
]

EnergyLevel = Literal["low", "medium", "high"]
TextureDensity = Literal["sparse", "balanced", "dense"]
ScaleType = Literal["intimate", "moderate", "expansive"]
StructureType = Literal["AABA", "ABAB", "through-composed"]


@dataclass(frozen=True)
class CompositionIntent:
    mood: Mood
    energy: EnergyLevel
    atmosphere: str
    perceived_scale: ScaleType
    texture_density: TextureDensity


@dataclass(frozen=True)
class CompositionPlan:
    key: str
    mode: Literal["major", "minor"]
    tempo: int
    length: int
    structure: StructureType
    texture: TextureDensity
    note_density: float
    melodic_range: tuple[int, int]
    harmonic_complexity: float
    explanation: list[str]
