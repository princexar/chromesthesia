"""Offline MIDI→WAV rendering using the FluidSynth command-line tool and SoundFont files."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path


def find_soundfont(assets_dir: Path) -> Path | None:
    """Return the first ``.sf2`` / ``.sf3`` file found under ``assets_dir``, or None if none exist."""
    for ext in ("*.sf2", "*.SF2", "*.sf3", "*.SF3"):
        matches = list(assets_dir.glob(ext))
        if matches:
            return matches[0]
    return None


def render_wav_with_fluidsynth(
    midi_path: Path,
    wav_path: Path,
    soundfont: Path,
) -> bool:
    """Run ``fluidsynth`` to synthesize ``midi_path`` into ``wav_path`` using ``soundfont``.

    Requires the ``fluidsynth`` executable on ``PATH``. Returns True only if the WAV file
    exists and is non-empty after the subprocess exits successfully.

    Args:
        midi_path: Input MIDI file.
        wav_path: Output WAV path (overwritten if present).
        soundfont: Path to a compatible SoundFont (``.sf2`` / ``.sf3``).

    Returns:
        Whether rendering appears to have succeeded.
    """
    fluidsynth = shutil.which("fluidsynth")
    if not fluidsynth:
        return False
    wav_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        fluidsynth,
        "-ni",
        str(soundfont),
        str(midi_path),
        "-F",
        str(wav_path),
        "-r",
        "44100",
    ]
    try:
        subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
            timeout=120,
        )
        return wav_path.is_file() and wav_path.stat().st_size > 0
    except (subprocess.CalledProcessError, OSError, subprocess.TimeoutExpired):
        return False
