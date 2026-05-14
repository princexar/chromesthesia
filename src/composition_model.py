"""Placeholder for a *future* trainable composition planner (not used in production yet).

Today Chromesthesia uses rule-based logic in ``composition_intelligence_layer.py`` (re-exported
via ``music_mapping.plan_composition`` for stable imports). This module sketches how a **PyTorch**
model could someday learn:

    perception numbers (+ encoded text) → continuous / categorical composition targets

Nothing here is trained, loaded from disk, or wired into the Streamlit app.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class CompositionPlannerDimensions:
    """Fixed sizes for the *future* model's input vector and prediction heads.

    Continuous perception (first 5 slots, in order you listed):
    - brightness, contrast, edge_detail_score, ml_energy, ml_variation

    Then you concatenate **learned or one-hot encodings** for mood and atmosphere. Their widths
    are configurable so you can swap CLIP embedding sizes or vocabulary sizes later without
    rewriting the whole model.
    """

    mood_dim: int = 8
    atmosphere_dim: int = 16
    hidden_dim: int = 64

    @property
    def num_continuous(self) -> int:
        """Scalar features from image analysis + ResNet summary (before mood / atmosphere)."""
        return 5

    @property
    def input_dim(self) -> int:
        """Total length of one perception vector passed to ``forward``."""
        return self.num_continuous + self.mood_dim + self.atmosphere_dim

    # Output head sizes (what the model would *predict* after training)
    num_mode_classes: int = 2  # e.g. major vs minor
    num_texture_classes: int = 3  # sparse / balanced / dense
    num_structure_classes: int = 3  # AABA / ABAB / through-composed


class CompositionPlannerModel(nn.Module):
    """Small MLP skeleton: maps a single perception vector to many composition targets.

    After training (not implemented here), you would:
    1. Build batches of ``input_dim`` tensors from real images + your encoders.
    2. Build matching **labels** (tempo BPM, duration in seconds, class indices, etc.).
    3. Minimize a loss (e.g. MSE for regressions + cross-entropy for class heads).
    4. Replace or blend with the rule-based planner when quality is good enough.

    Right now ``forward`` only runs a **randomly initialized** network—outputs are meaningless noise.
    """

    def __init__(self, dims: CompositionPlannerDimensions | None = None) -> None:
        super().__init__()
        self.dims = dims or CompositionPlannerDimensions()
        d_in = self.dims.input_dim
        h = self.dims.hidden_dim

        # Shared trunk: learns joint representations of all inputs before splitting into heads.
        self.trunk = nn.Sequential(
            nn.Linear(d_in, h),
            nn.ReLU(),
            nn.Linear(h, h),
            nn.ReLU(),
        )

        # One small linear layer per target family. Later you can deepen any head.
        self.tempo = nn.Linear(h, 1)  # regression → BPM-like scalar
        self.duration = nn.Linear(h, 1)  # regression → seconds-like scalar
        self.mode_logits = nn.Linear(h, self.dims.num_mode_classes)  # classification → major/minor
        self.texture_logits = nn.Linear(h, self.dims.num_texture_classes)
        self.note_density = nn.Linear(h, 1)  # regression → 0–1 style density
        self.harmonic_complexity = nn.Linear(h, 1)  # regression → e.g. 0–1 continuous proxy
        self.structure_logits = nn.Linear(h, self.dims.num_structure_classes)

    def forward(self, perception: torch.Tensor) -> dict[str, torch.Tensor]:
        """Run a forward pass.

        Args:
            perception: shape ``[batch, input_dim]`` — concatenate continuous features then
                mood encoding then atmosphere encoding along the last dimension.

        Returns:
            Dictionary of raw model outputs (logits for class heads, scalars for regression heads).
            Downstream code would apply ``argmax``, ``sigmoid``, or scaling to turn these into
            music parameters—*after* you define training targets and losses.
        """
        if perception.ndim != 2 or perception.shape[-1] != self.dims.input_dim:
            raise ValueError(
                f"Expected perception shape [batch, {self.dims.input_dim}], got {tuple(perception.shape)}"
            )
        z = self.trunk(perception)
        return {
            "tempo": self.tempo(z),
            "duration": self.duration(z),
            "mode_logits": self.mode_logits(z),
            "texture_logits": self.texture_logits(z),
            "note_density": self.note_density(z),
            "harmonic_complexity": self.harmonic_complexity(z),
            "structure_logits": self.structure_logits(z),
        }


def explain_future_training_plan() -> str:
    """Human-readable notes on how you could train ``CompositionPlannerModel`` later.

    Not executed automatically—call it from a notebook or REPL when you are ready to learn
    the training loop. Kept as a function so this file stays import-safe for the rest of the app.
    """
    return """
Future training checklist (high level):

1. **Dataset**: pairs (perception_vector, composition_labels). Labels should match the production
   ``CompositionPlan`` schema (key, mode, tempo_bpm, duration_seconds, structure, densities,
   melodic_range, harmonic_complexity, instrument_feel, mood, explanation strings)—from the current
   rule-based planner, from human experts, or from edited MIDI you like.

2. **Losses**: combine regression losses (tempo, duration, densities) with cross-entropy on
   categorical heads (mode, texture, structure). Weight each term so one target does not dominate.

3. **Normalization**: put continuous inputs on similar scales (e.g. 0–1) before ``forward``;
   normalize regression targets too so the network does not chase huge BPM numbers early on.

4. **Encoding mood / atmosphere**: today CLIP gives text; you would turn that into a fixed-length
   vector (one-hot over a small mood list, or a small learned embedding table keyed by label).

5. **Evaluation**: hold out images, compare predicted plans to references, *and* listen to MIDI
   from both—musical quality is the real metric.

6. **Integration**: only after validation would you load ``state_dict`` here and optionally replace
   pieces of ``composition_intelligence_layer.py``—not before.

This module is intentionally disconnected from ``app.py`` and from MIDI generation.
""".strip()
