"""Build multi-measure MIDI from a ``CompositionPlan`` (melody + optional harmony track).

Uses ``mido`` only—no neural inference. Consumes the structured plan from
``composition_intelligence_layer`` (re-exported by ``music_mapping`` for convenience).
"""

from __future__ import annotations

import random
from pathlib import Path

import mido
from mido import Message, MetaMessage, MidiFile, MidiTrack, bpm2tempo

from src.composition_intelligence_layer import CompositionPlan

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


def _measures_from_plan(duration_seconds: float, tempo_bpm: int) -> int:
    quarters_total = duration_seconds * (tempo_bpm / 60.0)
    m = int(round(quarters_total / 4.0))
    return max(12, min(96, m))


def _section_form_for_generator(structure: str) -> str:
    s = structure.lower().strip()
    if "through" in s:
        return "through"
    if s == "abab":
        return "ABAB"
    return "AABA"


def _rhythm_complexity_label(plan: CompositionPlan) -> str:
    tex_w = {"sparse": 0.2, "balanced": 0.5, "dense": 0.85}[plan.texture_density]
    x = 0.55 * plan.note_density + 0.45 * tex_w
    if x < 0.33:
        return "Low"
    if x < 0.58:
        return "Moderate"
    return "High"


def _harmonic_band(h: float) -> str:
    if h < 0.38:
        return "simple"
    if h < 0.68:
        return "moderate"
    return "rich"


def _melody_program(instrument_feel: str) -> int:
    s = instrument_feel.lower()
    if "electric piano" in s:
        return 4
    return 0


def _harmony_program(instrument_feel: str) -> int:
    s = instrument_feel.lower()
    if "strings" in s and "piano" in s:
        return 48
    if "electric bass" in s:
        return 33
    return 32


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


def _rest_probability(plan: CompositionPlan, rhythm_label: str) -> float:
    """Higher for sparse / slow scenes (longer breaths between notes)."""
    tex = plan.texture_density
    p = {"sparse": 0.34, "balanced": 0.2, "dense": 0.09}[tex]
    p *= 1.1 - 0.45 * plan.note_density
    if rhythm_label == "High":
        p *= 0.7
    elif rhythm_label == "Low":
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
        plan: Structured targets from ``build_composition_plan`` (no raw note lists).
        out_path: Destination ``.mid`` file.
        seed: RNG seed for reproducible note choices.

    Returns:
        ``out_path`` after save.
    """
    low, high = plan.melodic_range
    if low >= high:
        low, high = 55, 84
    pool = _scale_midi_notes(plan.key, plan.mode, low, high)
    if not pool:
        pool = list(range(60, 73))

    mid = MidiFile(type=1)
    ticks_per_beat = mid.ticks_per_beat
    ticks_per_measure = ticks_per_beat * _BEATS_PER_MEASURE
    measures = _measures_from_plan(plan.duration_seconds, plan.tempo_bpm)
    section_form = _section_form_for_generator(plan.structure)
    rhythm_label = _rhythm_complexity_label(plan)
    harm_band = _harmonic_band(plan.harmonic_complexity)
    mel_prog = _melody_program(plan.instrument_feel)
    harm_prog = _harmony_program(plan.instrument_feel)

    mel = MidiTrack()
    mid.tracks.append(mel)
    mel.append(MetaMessage("set_tempo", tempo=bpm2tempo(plan.tempo_bpm), time=0))
    mel.append(MetaMessage("track_name", name="Melody", time=0))
    mel.append(Message("program_change", program=mel_prog, time=0))

    harm: MidiTrack | None = None
    if harm_band != "simple":
        harm = MidiTrack()
        mid.tracks.append(harm)
        harm.append(MetaMessage("track_name", name="Harmony", time=0))
        harm.append(Message("program_change", program=harm_prog, time=0))

    slots_pm = _slots_per_measure(rhythm_label, plan.texture_density)
    step_ticks = max(1, ticks_per_measure // slots_pm)
    rest_p = _rest_probability(plan, rhythm_label)

    last = pool[len(pool) // 2]
    leap = 7 if rhythm_label == "High" else 5

    pending = 0
    for m in range(measures):
        sec = _section_label_at_measure(section_form, m, measures)
        rloc = random.Random((seed or 0) + m * 1009 + sum(ord(c) for c in sec) * 17)

        for _slot in range(slots_pm):
            if rloc.random() < rest_p:
                pending += step_ticks
                continue

            candidates = [p for p in pool if abs(p - last) <= leap] or pool
            n = rloc.choice(candidates)
            if rloc.random() < 0.1 and plan.mode.lower().startswith("min"):
                n = min(n + 12, max(pool))

            note_len = step_ticks
            if plan.texture_density == "sparse" and rhythm_label == "Low":
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
            if harm_band == "moderate":
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
