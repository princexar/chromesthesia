"""Render MIDI to audio (e.g. soundfonts, synthesis)."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path


def find_soundfont(assets_dir: Path) -> Path | None:
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
    """Render MIDI to WAV using the `fluidsynth` CLI if available."""
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
