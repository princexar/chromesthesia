# Chromesthesia

Chromesthesia is an experimental **multimodal** project that explores **computational synesthesia**: it reads a **still image** and produces a **full multi-bar MIDI** (and optional **WAV**) whose high-level musical parameters are driven by **pixels**, a **frozen ResNet18** summary, and optional **CLIP** scene cues. Nothing here trains a custom “composer” end-to-end; planning is **rule-based**, and neural nets are used only as **fixed feature extractors** or **zero-shot rankers**.

---

## What you get

- **Streamlit UI** (`app.py`): upload an image, see visual stats, CLIP atmosphere rankings, ResNet summary numbers, the generated **composition plan**, and a step-by-step “how the image becomes music” panel.
- **MIDI export**: diatonic melody in the chosen key/mode, length tied to tempo and planned duration, optional second track for bass/harmony when the plan’s harmony level is not minimal.
- **Optional WAV preview**: if [FluidSynth](https://www.fluidsynth.org/) is on your `PATH` and you place a SoundFont (`.sf2` / `.sf3`) under `assets/soundfonts/`, the app can render a WAV for in-browser playback and download.

---

## Architecture (modular pipeline)

Data flows in one direction; the **composition intelligence layer** never emits raw notes or audio—only a structured **`CompositionPlan`**.

| Stage | Module(s) | Role |
|-------|-----------|------|
| Image → pixel stats | `src/image_analysis.py` | Brightness, dominant color, contrast, edge/detail (no deep learning). |
| Image → scene cue | `src/semantic_interpreter.py`, `src/scene_composition_cue.py` | CLIP ranks prompts; cues fold in scale, duration band, texture, energy. |
| Image → embedding summary | `src/ml_features.py` | ResNet18 backbone → 512-D vector → small scalar summary (energy, variation, peak activation, …). |
| Perception → plan | `src/composition_intelligence_layer.py` | Rule-based **`CompositionPlan`**: key, mode, tempo, duration, structure, densities, melodic range, harmonic complexity, instrument feel, mood line, explanations. |
| Stable import path | `src/music_mapping.py` | Thin facade: `plan_composition` delegates to the intelligence layer; `interpret_mood` is UI-only prose from pixels. |
| Plan → MIDI | `src/music_generator.py` | Builds multi-measure MIDI with `mido` from a `CompositionPlan`. |
| MIDI → WAV | `src/audio_renderer.py` | Optional FluidSynth subprocess + SoundFont. |

A placeholder **trainable** planner sketch lives in `src/composition_model.py`; it is **not** wired into the app or MIDI path yet.

---

## Quick start

### 1. Environment

Use Python **3.10+** (3.11 or 3.12 recommended). Create a virtual environment from the project root:

```bash
cd chromesthesia
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

First launch will **download** CLIP and ResNet weights from Hugging Face / PyTorch; ensure you have disk space and (for CLIP) the `transformers` dependency installed.

### 2. Run the app

```bash
streamlit run app.py
```

Open the URL Streamlit prints (usually `http://localhost:8501`), upload an image, then use **Generate Music**. MIDI is written under `outputs/midi/`; WAV appears under `outputs/audio/` when rendering succeeds.

### 3. Optional: hear WAV in the browser

1. Install **FluidSynth** so the `fluidsynth` command exists on your `PATH`.
2. Add at least one **General MIDI** SoundFont file (`.sf2` or `.sf3`) to `assets/soundfonts/` (that directory is gitignored for large binaries—create it if needed).

---

## Project layout

```text
chromesthesia/
├── app.py                 # Streamlit entrypoint
├── requirements.txt
├── README.md
├── docs/
│   └── project_notes.md   # Loose design / scratch notes
├── src/
│   ├── image_analysis.py
│   ├── semantic_interpreter.py
│   ├── scene_composition_cue.py
│   ├── ml_features.py
│   ├── composition_intelligence_layer.py  # production planner → CompositionPlan
│   ├── music_mapping.py                     # facade + UI mood line
│   ├── music_generator.py                 # CompositionPlan → MIDI
│   ├── audio_renderer.py
│   └── composition_model.py               # future PyTorch sketch (unused in app)
├── assets/soundfonts/     # add your .sf2 / .sf3 here for WAV
├── outputs/midi/          # generated .mid (patterns may be gitignored)
└── outputs/audio/         # generated .wav when FluidSynth works
```

---

## Dependencies (summary)

| Package | Typical use in this project |
|---------|-----------------------------|
| `streamlit` | Web UI |
| `pillow`, `numpy` | Images and pixel math |
| `torch`, `torchvision` | ResNet18 embedding |
| `transformers` | CLIP (if missing, the app falls back to a neutral scene cue) |
| `mido` | MIDI file I/O |

---

## Limitations (honest scope)

- **CLIP and ResNet scores are not “ground truth” emotions or objects**; they are similarity signals and vector statistics useful for heuristics.
- **Musical content** is generated by **hand-tuned rules** (scales, random walks, simple chord cycles)—not a large generative audio model.
- **Training**: the PyTorch planner in `composition_model.py` is a **stub** until you add data, losses, and integration yourself.

---

## Contributing / extending

- Tweak **how images map to plans**: edit `src/composition_intelligence_layer.py` (`build_composition_plan` and helpers).
- Tweak **how plans become notes**: edit `src/music_generator.py`.
- Tweak **CLIP wording or axes**: edit `src/semantic_interpreter.py` (default labels) and `src/scene_composition_cue.py` (prompt sets and tables).

If you add new Python files under `src/`, keep imports package-style (`from src.…`) when running from the repo root with Streamlit, matching `app.py`.
