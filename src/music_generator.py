"""Build multi-measure MIDI from a ``CompositionPlan`` (melody + optional harmony track).

Uses ``mido`` only—no neural inference. Scene-driven fields control length, rests, subdivisions,
section character, and bass/chord density.
"""

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

_BEATS_PER_MEASURE = 4


def _scale_midi_notes(key: str, mode: str, low: int, high: int) -> list[int]:
    """List every MIDI note in [low, high] that belongs to the diatonic scale for key/mode."""
    root = _ROOT_PC[key]
    ivs = (
        [0, 2, 4, 5, 7, 9, 11]
        if mode.lower().startswith("maj")
        else [0, 2, 3, 5, 7, 8, 10]
    )
    pcs = {(root + iv) % 12 for iv in ivs}
    return [n for n in range(low, high + 1) if (n % 12) in pcs]


def _slots_per_measure(rhythm_complexity: str, texture_density: str) -> int:
    """How many melody subdivisions per 4/4 measure (powers of two for clean bar math)."""
    base = {"Low": 2, "Moderate": 4, "High": 8}[rhythm_complexity]
    mul = {"sparse": 0.75, "balanced": 1.0, "dense": 1.35}[texture_density]
    raw = int(round(base * mul))
    for snap in (16, 8, 4, 2):
        if raw >= snap - 1:
            return snap
    return 2


def _rest_probability(plan: CompositionPlan) -> float:
    """Higher for sparse / slow scenes (longer breaths between notes)."""
    tex = plan.texture_density
    rhyth = plan.rhythm_complexity
    p = {"sparse": 0.34, "balanced": 0.2, "dense": 0.09}[tex]
    if rhyth == "High":
        p *= 0.7
    elif rhyth == "Low":
        p *= 1.25
    return max(0.06, min(0.55, p))


def _section_label_at_measure(form: str, measure_idx: int, measures: int) -> str:
    """Return A/B/T section tag for this measure (drives local RNG stream for motifs)."""
    if measures < 1:
        return "A"
    if form == "through":
        return "T"
    parts = 4
    w = max(1, measures // parts)
    q = min(measure_idx // w, parts - 1)
    if form == "ABAB":
        return ("A", "B", "A", "B")[q]
    return ("A", "A", "B", "A")[q]


def _chord_root_offset_semitones(measure_i: int, mode: str) -> int:
    """Simple rotating progression: I–IV–vi–V (major) or i–iv–bVI–bVII (natural minor feel)."""
    if mode.lower().startswith("maj"):
        cycle = (0, 5, 9, 7)
    else:
        cycle = (0, 5, 8, 10)
    return cycle[measure_i % len(cycle)]


def _bass_root_midi(measure_i: int, key: str, mode: str, octave: int = 3) -> int:
    """MIDI note number for the chord root in the bass register."""
    kr = _ROOT_PC[key]
    pc = (kr + _chord_root_offset_semitones(measure_i, mode)) % 12
    return 12 * octave + pc


def _triad_midi(root_midi: int, mode: str) -> tuple[int, int, int]:
    """Close-position triad above ``root_midi`` (major or minor third stack)."""
    if mode.lower().startswith("maj"):
        return root_midi, root_midi + 4, root_midi + 7
    return root_midi, root_midi + 3, root_midi + 7


def generate_midi(plan: CompositionPlan, out_path: Path, seed: int | None = None) -> Path:
    """Write a multi-measure MIDI file: melody track plus optional bass/harmony based on the plan.

    Args:
        plan: Includes ``measures``, ``section_form``, ``texture_density``, ``harmonic_complexity``.
        out_path: Destination ``.mid`` file.
        seed: RNG seed for reproducible note choices.

    Returns:
        ``out_path`` after save.
    """
    low, high = plan.midi_low, plan.midi_high
    if low >= high:
        low, high = 55, 84
    pool = _scale_midi_notes(plan.key, plan.mode, low, high)
    if not pool:
        pool = list(range(60, 73))

    mid = MidiFile(type=1)
    ticks_per_beat = mid.ticks_per_beat
    ticks_per_measure = ticks_per_beat * _BEATS_PER_MEASURE
    measures = max(1, plan.measures)

    mel = MidiTrack()
    mid.tracks.append(mel)
    mel.append(MetaMessage("set_tempo", tempo=bpm2tempo(plan.tempo_bpm), time=0))
    mel.append(MetaMessage("track_name", name="Melody", time=0))
    mel.append(Message("program_change", program=0, time=0))

    harm: MidiTrack | None = None
    if plan.harmonic_complexity != "simple":
        harm = MidiTrack()
        mid.tracks.append(harm)
        harm.append(MetaMessage("track_name", name="Harmony", time=0))
        harm.append(Message("program_change", program=32, time=0))

    slots_pm = _slots_per_measure(plan.rhythm_complexity, plan.texture_density)
    step_ticks = max(1, ticks_per_measure // slots_pm)
    rest_p = _rest_probability(plan)

    last = pool[len(pool) // 2]
    leap = 7 if plan.rhythm_complexity == "High" else 5

    pending = 0
    for m in range(measures):
        sec = _section_label_at_measure(plan.section_form, m, measures)
        rloc = random.Random((seed or 0) + m * 1009 + sum(ord(c) for c in sec) * 17)

        for slot in range(slots_pm):
            if rloc.random() < rest_p:
                pending += step_ticks
                continue

            candidates = [p for p in pool if abs(p - last) <= leap] or pool
            n = rloc.choice(candidates)
            if rloc.random() < 0.1 and plan.mode.lower().startswith("min"):
                n = min(n + 12, max(pool))

            note_len = step_ticks
            if plan.texture_density == "sparse" and plan.rhythm_complexity == "Low":
                note_len = min(ticks_per_measure, step_ticks * 2)

            mel.append(Message("note_on", note=n, velocity=76, time=pending))
            mel.append(Message("note_off", note=n, velocity=0, time=note_len))
            pending = 0
            last = n

    if pending > 0:
        mel.append(MetaMessage("end_of_track", time=pending))
    else:
        mel.append(MetaMessage("end_of_track", time=0))

    if harm is not None:
        h_pending = 0
        for m in range(measures):
            root = _bass_root_midi(m, plan.key, plan.mode, octave=3)
            if plan.harmonic_complexity == "moderate":
                fifth = root + 7
                harm.append(Message("note_on", note=root, velocity=54, time=h_pending))
                harm.append(Message("note_off", note=root, velocity=0, time=ticks_per_beat * 2))
                harm.append(Message("note_on", note=fifth, velocity=50, time=0))
                harm.append(Message("note_off", note=fifth, velocity=0, time=ticks_per_beat * 2))
                h_pending = 0
            else:
                r1, r2, r3 = _triad_midi(root, plan.mode)
                arp = (r1, r2, r3, min(r1 + 12, 79))
                for note, vel in zip(arp, (56, 50, 52, 54)):
                    harm.append(Message("note_on", note=note, velocity=vel, time=h_pending))
                    harm.append(Message("note_off", note=note, velocity=0, time=ticks_per_beat))
                    h_pending = 0
        harm.append(MetaMessage("end_of_track", time=h_pending))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    mid.save(str(out_path))
    return out_path
