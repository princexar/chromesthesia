"""Helpers for exploring and preparing the MAESTRO piano MIDI dataset (no training here).

MAESTRO (MIDI and Audio Edited for Synchronous Tracks and Organization) is a curated collection of
classical piano performances distributed with aligned audio. This module only **reads MIDI** and
extracts **simple scalar summaries** you might later use as labels or regression targets for a
future composition model.

**Not used by the Streamlit app.** Import this from notebooks or small scripts when you are ready
to profile your on-disk MAESTRO copy.

**Dependencies:** ``pretty_midi`` (MIDI parsing), ``pandas`` (tabular summaries). See ``requirements.txt``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import pretty_midi

# ---------------------------------------------------------------------------
# Where to put MAESTRO on your machine
# ---------------------------------------------------------------------------
# 1. Download MAESTRO from the Magenta / TensorFlow hosting page (search "MAESTRO dataset").
#    You typically get a folder like ``maestro-v3.0.0`` containing ``metadata.csv``, ``midi/``,
#    ``audio/``, etc.
#
# 2. Recommended layout inside *this* project (ignored by git; see ``.gitignore``):
#
#        chromesthesia/
#            data/
#                maestro-v3.0.0/          # common: unzip the archive directly here
#                    midi/
#                    audio/
#                    ...
#        or
#            data/
#                maestro/
#                    maestro-v3.0.0/
#                        midi/
#                        ...
#
#    Point ``scan_maestro_midis_to_dataframe`` at ``data/`` (recursive) or at ``.../midi`` alone.
#
# 3. See the repo ``.gitignore`` for ``data/maestro/``, ``data/maestro-*/``, and optional root ``/maestro-v*/``.
# ---------------------------------------------------------------------------

# Example path if you unzip MAESTRO straight into ``data/`` (ignored: ``data/maestro-*/`` etc.).
MAESTRO_SUGGESTED_ROOT = Path("data") / "maestro-v3.0.0"
@dataclass(frozen=True)
class MaestroMidiFeatures:
    """High-level numbers summarizing one MIDI file (good for dataset statistics or future labels).

    All times are in **seconds** as ``pretty_midi`` interprets them after resolving tempo events.
    Pitches use **MIDI note numbers** (middle C = 60; each step is one semitone / piano key).
    """

    source_path: str
    """Filesystem path to the MIDI file (string so rows serialize cleanly in pandas)."""

    duration_seconds: float
    """How long the file lasts from the first audible event to the last note release.

    Musically: the **span of the performance** on the timeline, not the length printed on sheet
    music (rubato and fermatas already baked into the MIDI clock).
    """

    average_pitch: float
    """Typical note height, in MIDI pitch space.

    Computed as a **duration-weighted** mean: a whole note counts more than a thirty-second note.
    Musically: tells you if the piece sits in the **bass**, **middle**, or **high** register on
    average (MAESTRO is piano-only, so this mostly reflects hand position and repertoire).
    """

    pitch_range_semitones: float
    """Distance between the lowest and highest note used (in semitones).

    Musically: **how wide the composer / performer spans on the keyboard** in this file—small
    values suggest a narrow motif; large values suggest runs, leaps, or full-range writing.
    """

    note_density_per_second: float
    """Note onsets per second of ``duration_seconds``.

    Musically: a rough **busy-ness** meter—many fast notes in a short time push this up; sparse
    music or long rests push it down. (Counting is per ``pretty_midi`` ``Note`` objects, so chords
    count as multiple simultaneous onsets at that instant.)
    """

    average_velocity: float
    """Mean MIDI velocity (1–127) across notes, duration-weighted.

    Musically: **how loud / accented** the performance is on average in the file—higher means more
    forceful key strikes in the underlying performance capture (still only a crude proxy for true
    loudness, which also depends on sound design).
    """

    tempo_estimate_bpm: float
    """A single beats-per-minute guess for the whole file.

    If the MIDI contains explicit tempo meta-events, we use the **mean** of those values (MAESTRO
    often has one global tempo per piece). Otherwise we fall back to ``pretty_midi``'s empirical
    ``estimate_tempo()``—useful, but less reliable when tempos drift a lot.
    """

    sustain_pedal_usage: float | None
    """Fraction of the timeline (0.0–1.0) where the sustain (damper) pedal is likely **down**.

    ``None`` means no sustain control-change data was found (some MIDI exports omit pedal lanes).

    Musically: high values mean **more overlapping resonance**—the piano strings ring together even
    after keys are released until the pedal is lifted. This strongly changes perceived harmony blur.
    """

    num_notes: int
    """Raw count of ``Note`` objects across non-drum instruments (helpful sanity check)."""


def load_maestro_midi(path: Path | str) -> pretty_midi.PrettyMIDI:
    """Load one MIDI file with ``pretty_midi`` (raises if the file is corrupt or unreadable)."""
    return pretty_midi.PrettyMIDI(str(path))


def _iter_notes(pm: pretty_midi.PrettyMIDI) -> Iterable[pretty_midi.Note]:
    """Yield every note on non-drum instruments (MAESTRO is piano; drums should not appear)."""
    for instrument in pm.instruments:
        if instrument.is_drum:
            continue
        yield from instrument.notes


def _tempo_estimate_bpm(pm: pretty_midi.PrettyMIDI) -> float:
    """Pick one representative tempo in BPM (see ``MaestroMidiFeatures.tempo_estimate_bpm``)."""
    try:
        _times, tempi = pm.get_tempo_changes()
    except (ValueError, AttributeError):
        tempi = np.array([], dtype=float)
    if tempi is not None and len(tempi) > 0:
        # ``tempi`` entries are instantaneous BPM values; mean is a simple global summary.
        return float(np.clip(np.mean(tempi), 20.0, 300.0))
    est = float(pm.estimate_tempo())
    return float(np.clip(est, 20.0, 300.0))


def _sustain_pedal_on_ratio(pm: pretty_midi.PrettyMIDI, duration_seconds: float) -> float | None:
    """Return fraction of [0, duration] where sustain pedal CC64 is interpreted as ON.

    Convention: velocity ``>= 64`` means pedal depressed; ``< 64`` means released (common in MIDI).
    """
    if duration_seconds <= 0:
        return None

    events: list[tuple[float, bool]] = []
    for instrument in pm.instruments:
        if instrument.is_drum:
            continue
        for cc in instrument.control_changes:
            if cc.number != 64:
                continue
            events.append((float(cc.time), cc.value >= 64))

    if not events:
        return None

    events.sort(key=lambda pair: pair[0])

    pedal_on_time = 0.0
    current_on = False
    prev_t = 0.0

    for t, is_on in events:
        segment_end = min(t, duration_seconds)
        if segment_end > prev_t and current_on:
            pedal_on_time += segment_end - prev_t
        prev_t = max(prev_t, t)
        current_on = is_on

    if prev_t < duration_seconds and current_on:
        pedal_on_time += duration_seconds - prev_t

    return float(np.clip(pedal_on_time / duration_seconds, 0.0, 1.0))


def extract_maestro_midi_features(
    path: Path | str,
    pm: pretty_midi.PrettyMIDI | None = None,
) -> MaestroMidiFeatures:
    """Compute ``MaestroMidiFeatures`` for a single MIDI path (loads the file if ``pm`` is None)."""
    path = Path(path)
    pm = pm if pm is not None else load_maestro_midi(path)

    notes = list(_iter_notes(pm))
    duration = float(pm.get_end_time())
    if duration <= 0:
        duration = 0.0

    if not notes:
        return MaestroMidiFeatures(
            source_path=str(path.resolve()),
            duration_seconds=duration,
            average_pitch=float("nan"),
            pitch_range_semitones=0.0,
            note_density_per_second=0.0,
            average_velocity=float("nan"),
            tempo_estimate_bpm=_tempo_estimate_bpm(pm),
            sustain_pedal_usage=_sustain_pedal_on_ratio(pm, duration),
            num_notes=0,
        )

    pitches = np.array([n.pitch for n in notes], dtype=float)
    velocities = np.array([n.velocity for n in notes], dtype=float)
    starts = np.array([n.start for n in notes], dtype=float)
    ends = np.array([n.end for n in notes], dtype=float)
    lengths = np.maximum(ends - starts, 1e-6)

    # Duration-weighting: longer notes contribute more to "typical" pitch and loudness.
    weights = lengths / lengths.sum()
    average_pitch = float(np.sum(weights * pitches))
    average_velocity = float(np.sum(weights * velocities))

    pitch_min = float(np.min(pitches))
    pitch_max = float(np.max(pitches))
    pitch_range = pitch_max - pitch_min

    num_notes = len(notes)
    density = (num_notes / duration) if duration > 0 else 0.0

    return MaestroMidiFeatures(
        source_path=str(path.resolve()),
        duration_seconds=duration,
        average_pitch=average_pitch,
        pitch_range_semitones=pitch_range,
        note_density_per_second=float(density),
        average_velocity=average_velocity,
        tempo_estimate_bpm=_tempo_estimate_bpm(pm),
        sustain_pedal_usage=_sustain_pedal_on_ratio(pm, duration),
        num_notes=num_notes,
    )


def features_to_dict(features: MaestroMidiFeatures) -> dict[str, Any]:
    """Convert a dataclass row into a plain dict (handy before building a DataFrame)."""
    return asdict(features)


def scan_maestro_midis_to_dataframe(
    root: Path | str,
    *,
    recursive: bool = True,
    extensions: tuple[str, ...] = (".mid", ".midi", ".MID", ".MIDI"),
) -> tuple[pd.DataFrame, list[str]]:
    """Walk ``root`` for MIDI files, extract features, and return a pandas DataFrame plus errors.

    Args:
        root: Directory whose MIDI files you want to summarize (e.g. ``data/maestro``).
        recursive: If True, use ``rglob`` so nested MAESTRO year folders are included.
        extensions: Filename suffixes to treat as MIDI.

    Returns:
        A tuple ``(df, errors)`` where ``df`` has one row per successfully parsed file and
        ``errors`` holds human-readable failure messages for files that could not be read.

    This does **not** train anything—it is only for exploration and CSV export practice.
    """
    root = Path(root)
    if not root.is_dir():
        raise NotADirectoryError(f"Expected a directory, got: {root}")

    rows: list[dict[str, Any]] = []
    errors: list[str] = []

    iterator: Iterable[Path]
    if recursive:
        iterator = (p for ext in extensions for p in root.rglob(f"*{ext}"))
    else:
        iterator = (p for ext in extensions for p in root.glob(f"*{ext}"))

    for midi_path in sorted(set(iterator)):
        try:
            feats = extract_maestro_midi_features(midi_path)
            rows.append(features_to_dict(feats))
        except Exception as exc:  # noqa: BLE001 - exploration helper: capture any parse failure
            errors.append(f"{midi_path}: {exc}")

    df = pd.DataFrame(rows)
    return df, errors
