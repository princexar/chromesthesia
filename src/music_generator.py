"""Generate MIDI or musical structures from mapped image features."""

from __future__ import annotations

import random
from pathlib import Path

import mido
from mido import Message, MetaMessage, MidiFile, MidiTrack, bpm2tempo

from src.music_mapping import CompositionPlan

_ROOT_PC = {
    "C": 0,
    "C#": 1,
    "D": 2,
    "D#": 3,
    "E": 4,
    "F": 5,
    "F#": 6,
    "G": 7,
    "G#": 8,
    "A": 9,
    "A#": 10,
    "B": 11,
}


def _scale_midi_notes(key: str, mode: str, low: int, high: int) -> list[int]:
    root = _ROOT_PC[key]
    ivs = (
        [0, 2, 4, 5, 7, 9, 11]
        if mode.lower().startswith("maj")
        else [0, 2, 3, 5, 7, 8, 10]
    )
    pcs = {(root + iv) % 12 for iv in ivs}
    return [n for n in range(low, high + 1) if (n % 12) in pcs]


def generate_midi(plan: CompositionPlan, out_path: Path, seed: int | None = None) -> Path:
    """Write a short melodic MIDI file based on the composition plan."""
    rng = random.Random(seed)
    low, high = plan.midi_low, plan.midi_high
    if low >= high:
        low, high = 55, 84
    pool = _scale_midi_notes(plan.key, plan.mode, low, high)
    if not pool:
        pool = list(range(60, 73))

    mid = MidiFile(type=1)
    track = MidiTrack()
    mid.tracks.append(track)
    ticks_per_beat = mid.ticks_per_beat
    tempo = bpm2tempo(plan.tempo_bpm)
    track.append(MetaMessage("set_tempo", tempo=tempo, time=0))
    track.append(MetaMessage("track_name", name="Chromesthesia", time=0))
    track.append(Message("program_change", program=0, time=0))

    if plan.rhythm_complexity == "High":
        steps, note_ticks = 16, ticks_per_beat // 4
    elif plan.rhythm_complexity == "Moderate":
        steps, note_ticks = 12, ticks_per_beat // 2
    else:
        steps, note_ticks = 8, ticks_per_beat

    velocity = 72
    last = pool[len(pool) // 2]
    for i in range(steps):
        if rng.random() < 0.15 and i > 0:
            continue
        span = 7 if plan.rhythm_complexity == "High" else 5
        candidates = [p for p in pool if abs(p - last) <= span] or pool
        n = rng.choice(candidates)
        if rng.random() < 0.12 and plan.mode.lower().startswith("min"):
            n = min(n + 12, max(pool))
        track.append(Message("note_on", note=n, velocity=velocity, time=0))
        track.append(Message("note_off", note=n, velocity=0, time=note_ticks))
        last = n

    out_path.parent.mkdir(parents=True, exist_ok=True)
    mid.save(str(out_path))
    return out_path
