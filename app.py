"""Streamlit UI: image upload, visual analysis, mapped composition, MIDI/WAV output."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import streamlit as st

from src.audio_renderer import find_soundfont, render_wav_with_fluidsynth
from src.image_analysis import analyze_image, load_image_from_bytes
from src.ml_features import extract_resnet18_embedding, summarize_embedding
from src.music_generator import generate_midi
from src.music_mapping import interpret_mood, plan_composition
from src.scene_composition_cue import NEUTRAL_SCENE_CUE, build_scene_composition_cue
from src.semantic_interpreter import interpret_scene

PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUT_MIDI_DIR = PROJECT_ROOT / "outputs" / "midi"
OUTPUT_AUDIO_DIR = PROJECT_ROOT / "outputs" / "audio"
SOUNDFONTS_DIR = PROJECT_ROOT / "assets" / "soundfonts"

HOW_IT_WORKS = """
Chromesthesia follows a **modular pipeline**: classic pixels, a ResNet embedding summary, and a
**CLIP scene cue** feed the rule-based **composition intelligence layer**, which outputs a structured
**CompositionPlan** (key, mode, tempo, duration, form, densities, ranges, harmony scalar, instrument
feel, mood line, and short explanations). **music_generator** turns that plan into multi-measure MIDI
(melody plus optional harmony); **WAV** still needs **fluidsynth** + a **`.sf2`** under `assets/soundfonts/`.

#### 1. Classic image analysis (explainable pixels)

- **Dominant color (hue)** → **musical key**.
- **Brightness** and **warm vs cool** hues → **major vs minor mode**.
- **Brightness** and **edge / detail** → **tempo** starting point.
- **Contrast** and **edges** → **melodic range** (MIDI span) and **note density** inputs.

#### 2. Neural embedding (ResNet18 backbone)

The **512**-D backbone vector is summarized into scalars such as **energy**, **variation**, and
**max activation**. Those blend into **tempo**, **note density**, and **melodic range** in the plan.
They are **not** scene captions.

#### 3. Scene composition cue (frozen CLIP)

**CLIP** drives duration bands, scale, texture, energy, and atmosphere strings that steer the
planner’s **structure**, **texture_density**, **duration_seconds**, and prose **mood** / **instrument_feel**
hints.

#### 4. What you hear

**Generate Music** maps the plan to subdivisions and rests, then writes diatonic MIDI; harmony depth
comes from the plan’s **harmonic_complexity** scalar (thresholded inside the generator).
"""


def _pipeline_arrow() -> None:
    st.markdown(
        '<div style="text-align:center;color:#888;font-size:1.35rem;line-height:1.2;">↓</div>',
        unsafe_allow_html=True,
    )


def _pipeline_card(title: str, headline: str, lines: list[str]) -> None:
    """Single pipeline step as a light card (Streamlit-friendly HTML)."""
    body = "<br/>".join(lines)
    st.markdown(
        f"<div style='border:1px solid rgba(49,51,63,0.35);border-radius:10px;padding:14px 16px;"
        f"margin:6px 0;background:rgba(250,250,250,0.55);'>"
        f"<div style='font-size:0.8rem;color:#555;text-transform:uppercase;letter-spacing:0.04em;'>"
        f"{title}</div>"
        f"<div style='font-size:1.05rem;font-weight:600;margin:6px 0 10px 0;color:#111;'>{headline}</div>"
        f"<div style='font-size:0.92rem;line-height:1.55;color:#333;'>{body}</div></div>",
        unsafe_allow_html=True,
    )


def render_translation_pipeline(
    image,
    features,
    ml_summary: Mapping[str, float],
    emb_dim: int,
    comp,
    scene_cue,
) -> None:
    """Transparent left-to-right *process* as a vertical pipeline (not a network graph)."""
    st.divider()
    st.subheader("How the Image Becomes Music")
    st.caption(
        "A **translation pipeline**: pixels, a ResNet summary, and a **CLIP scene cue** feed the "
        "**composition intelligence layer**, which emits a ``CompositionPlan``; ``music_generator`` "
        "writes multi-measure MIDI from that plan. No model is trained here—frozen nets only suggest structure."
    )

    # --- 1. Image input ---------------------------------------------------------
    r1, r2 = st.columns([1.35, 1], gap="medium")
    with r1:
        _pipeline_card(
            "Stage 1",
            "Image input",
            [
                "Your photo is the single source for every downstream stage.",
                "The same image feeds both classic pixel statistics and the ResNet pass.",
            ],
        )
    with r2:
        st.image(image, width=200, caption="Source image")

    _pipeline_arrow()

    # --- 2. Visual features -----------------------------------------------------
    _pipeline_card(
        "Stage 2",
        "Visual feature extraction (classic, explainable)",
        [
            f"<b>Brightness</b> — {features.brightness * 100:.0f}% (mean luminance)",
            f"<b>Dominant color</b> — <code>{features.dominant_color_hex}</code>",
            f"<b>Contrast</b> — {features.contrast * 100:.0f}% relative spread",
            f"<b>Edge / detail score</b> — {features.edge_detail_score * 100:.0f}% (sharpness proxy)",
        ],
    )
    _pipeline_arrow()

    # --- 3. ResNet embedding ----------------------------------------------------
    _pipeline_card(
        "Stage 3",
        "Learned visual embedding (frozen ResNet18)",
        [
            f"<b>Embedding size</b> — {emb_dim} floats (backbone vector, classifier removed)",
            f"<b>Neural energy score</b> — {ml_summary['energy_score']:.3f} (summary of activation strength)",
            f"<b>Variation score</b> — {ml_summary['variation_score']:.3f} (summary of spread across dimensions)",
            "These are <b>shape statistics</b> on the vector—not a caption of the scene.",
        ],
    )
    _pipeline_arrow()

    _pipeline_card(
        "Stage 4",
        "Scene composition cue (frozen CLIP, zero-shot)",
        [
            f"<b>Mood</b> — {scene_cue.emotional_mood}",
            f"<b>Energy</b> — {scene_cue.energy_level} &nbsp;|&nbsp; <b>Atmosphere</b> — {scene_cue.atmosphere}",
            f"<b>Scale</b> — {scene_cue.perceived_scale} &nbsp;|&nbsp; <b>Texture</b> — {scene_cue.texture_density}",
            f"<b>Duration</b> — {scene_cue.duration_band} (~{scene_cue.target_duration_sec:.0f}s target)",
        ],
    )
    _pipeline_arrow()

    # --- 5. Composition plan (intelligence layer) -------------------------------
    bars = max(12, min(96, int(round(comp.duration_seconds * comp.tempo_bpm / 240))))
    _pipeline_card(
        "Stage 5",
        "Composition plan (rule-based intelligence layer)",
        [
            f"<b>Key / mode</b> — {comp.key} {comp.mode}",
            f"<b>Tempo</b> — {comp.tempo_bpm} BPM",
            f"<b>Duration / form</b> — ~{comp.duration_seconds:.0f}s, **{comp.structure}** (~{bars} bars @ 4/4)",
            f"<b>Texture / note density</b> — {comp.texture_density}, {comp.note_density:.2f}",
            f"<b>Melodic range</b> — MIDI {comp.melodic_range[0]}–{comp.melodic_range[1]}",
            f"<b>Harmonic complexity</b> — {comp.harmonic_complexity:.2f} (thresholded in generator)",
            f"<b>Mood</b> — {comp.mood[:120]}{'…' if len(comp.mood) > 120 else ''}",
        ],
    )
    _pipeline_arrow()

    # --- 6. MIDI generator ------------------------------------------------------
    _pipeline_card(
        "Stage 6",
        "MIDI generator (``music_generator``, not a neural composer)",
        [
            f"<b>Length</b> — ~{comp.duration_seconds:.0f}s (~{bars} × 4/4 measures at {comp.tempo_bpm} BPM)",
            f"<b>Melody</b> — section-aware random walk ({comp.structure}) with rests from texture + note density",
            f"<b>Scale</b> — diatonic {comp.key} {comp.mode} within MIDI {comp.melodic_range[0]}–{comp.melodic_range[1]}",
            f"<b>Harmony track</b> — from harmonic complexity scalar + <code>{comp.instrument_feel[:80]}…</code>"
            if len(comp.instrument_feel) > 80
            else f"<b>Harmony track</b> — from harmonic complexity scalar + {comp.instrument_feel!r}",
            "<b>MIDI</b> — written with <code>mido</code> (melody + optional second track)",
        ],
    )
    _pipeline_arrow()

    # --- 7. Audio output ----------------------------------------------------------
    _pipeline_card(
        "Stage 7",
        "Audio output",
        [
            "<b>Playback</b> — WAV in the browser when fluidsynth + a SoundFont render the MIDI",
            "<b>Download MIDI</b> — always available after “Generate Music”",
            "<b>Download WAV</b> — only when offline rendering succeeds",
        ],
    )


def main() -> None:
    st.set_page_config(page_title="Chromesthesia", layout="wide")
    st.title("Chromesthesia")
    st.caption("Visual scenes translated into music through AI")

    uploaded = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg", "webp"])

    if uploaded is not None:
        data = uploaded.getvalue()
        upload_key = (uploaded.name, len(data))
        if st.session_state.get("_upload_key") != upload_key:
            st.session_state.pop("midi_path", None)
            st.session_state.pop("wav_path", None)
            st.session_state.pop("scene_rankings", None)
            st.session_state.pop("_scene_cache_key", None)
            st.session_state.pop("scene_composition_cue", None)
            st.session_state["_upload_key"] = upload_key

        image = load_image_from_bytes(data)
        features = analyze_image(image)
        mood = interpret_mood(features)

        st.session_state["features"] = features
        st.session_state["mood"] = mood
        st.session_state["upload_name"] = Path(uploaded.name).stem

        col_left, col_right = st.columns(2, gap="large")
        with col_left:
            st.subheader("Image preview")
            st.image(image, use_container_width=True)
        with col_right:
            st.subheader("Image analysis")
            st.metric("Brightness", f"{features.brightness * 100:.0f}%")
            st.markdown(f"**Dominant color** `{features.dominant_color_hex}`")
            st.markdown(
                f'<div style="height:12px;width:120px;background:{features.dominant_color_hex};'
                f'border-radius:4px;border:1px solid #666;margin:0.25rem 0 0.75rem 0"></div>',
                unsafe_allow_html=True,
            )
            st.metric("Contrast", f"{features.contrast * 100:.0f}% (relative)")
            st.metric("Edge / detail score", f"{features.edge_detail_score * 100:.0f}%")
            st.markdown("**Mood interpretation**")
            st.write(mood)

        st.divider()
        st.subheader("Scene Interpretation")
        if (
            st.session_state.get("_scene_cache_key") != upload_key
            or "scene_rankings" not in st.session_state
            or "scene_composition_cue" not in st.session_state
        ):
            with st.spinner("Loading CLIP and building scene composition cue…"):
                try:
                    scene_rankings = interpret_scene(image)
                    scene_cue = build_scene_composition_cue(image, scene_rankings)
                except ModuleNotFoundError as exc:
                    if exc.name != "transformers":
                        raise
                    st.error(
                        "Missing package **transformers** (needed for CLIP). In your project venv run:  \n"
                        "`pip install -r requirements.txt`  \n"
                        "Then reload this app."
                    )
                    scene_rankings = []
                    scene_cue = NEUTRAL_SCENE_CUE
                st.session_state["_scene_cache_key"] = upload_key
                st.session_state["scene_rankings"] = scene_rankings
                st.session_state["scene_composition_cue"] = scene_cue
        else:
            scene_rankings = st.session_state["scene_rankings"]
            scene_cue = st.session_state.get("scene_composition_cue", NEUTRAL_SCENE_CUE)
        st.caption(
            "Local **CLIP** compares your image to fixed phrases (ranked below), then **three extra CLIP passes** "
            "pick intimate vs expansive, short vs long feel, and sparse vs dense texture. "
            "That **scene composition cue** steers tempo, length, form, rests, and harmony in the generator."
        )
        for rank, (label, score) in enumerate(scene_rankings, start=1):
            st.write(f"{rank}. **{score * 100:.1f}%** — {label}")

        st.markdown("**Structured composition cue** *(used in mapping + MIDI)*")
        sc1, sc2, sc3 = st.columns(3)
        with sc1:
            st.write("**Emotional mood**", scene_cue.emotional_mood)
            st.write("**Energy**", scene_cue.energy_level)
        with sc2:
            st.write("**Atmosphere**", scene_cue.atmosphere)
            st.write("**Perceived scale**", scene_cue.perceived_scale)
        with sc3:
            st.write("**Duration**", f"{scene_cue.duration_band} (~{scene_cue.target_duration_sec:.0f}s)")
            st.write("**Texture density**", scene_cue.texture_density)

        st.divider()
        st.subheader("Neural Visual Interpretation")
        if st.session_state.get("_ml_cache_key") != upload_key or "ml_summary" not in st.session_state:
            with st.spinner("Computing ResNet18 embedding…"):
                embedding = extract_resnet18_embedding(image)
                ml_summary = summarize_embedding(embedding)
            st.session_state["_ml_cache_key"] = upload_key
            st.session_state["ml_summary"] = ml_summary
            st.session_state["ml_embedding_dim"] = int(embedding.shape[0])
        else:
            ml_summary = st.session_state["ml_summary"]
        emb_dim = st.session_state["ml_embedding_dim"]
        st.caption(
            "Summary numbers from the 512-float backbone vector—not scene labels. "
            "See `summarize_embedding` in `src/ml_features.py` for definitions."
        )
        m1, m2, m3 = st.columns(3)
        with m1:
            st.metric("Energy score", f"{ml_summary['energy_score']:.3f}")
        with m2:
            st.metric("Variation score", f"{ml_summary['variation_score']:.3f}")
        with m3:
            st.metric("Embedding dimensions", f"{emb_dim}")

        comp = plan_composition(features, ml_summary, scene_cue)
        st.session_state["composition"] = comp
        bars = max(12, min(96, int(round(comp.duration_seconds * comp.tempo_bpm / 240))))

        st.divider()
        st.subheader("Generated composition")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.write("**Key**", comp.key)
            st.write("**Tempo**", f"{comp.tempo_bpm} BPM")
            st.caption(f"MIDI span **{comp.melodic_range[0]}–{comp.melodic_range[1]}**")
        with c2:
            st.write("**Mode**", comp.mode)
            st.write("**Note density**", f"{comp.note_density:.2f}")
            st.caption(f"Texture **{comp.texture_density}**")
        with c3:
            st.write("**Length**", f"~{comp.duration_seconds:.0f}s")
            st.write("**Measures**", f"~{bars} bars (4/4)")
            st.caption(f"Structure **{comp.structure}**")
        with c4:
            st.write("**Harmonic complexity**", f"{comp.harmonic_complexity:.2f}")
            st.write("**Mood**")
            st.caption(comp.mood[:220] + ("…" if len(comp.mood) > 220 else ""))
        with st.expander("Instrument feel & planner explanation"):
            st.write(comp.instrument_feel)
            for line in comp.explanation:
                st.markdown(f"- {line}")

        render_translation_pipeline(image, features, ml_summary, emb_dim, comp, scene_cue)

        if st.button("Generate Music", type="primary"):
            stem = st.session_state.get("upload_name", "chromesthesia")
            midi_path = OUTPUT_MIDI_DIR / f"{stem}.mid"
            generate_midi(comp, midi_path, seed=hash(features.dominant_color_hex) % (2**31))
            st.session_state["midi_path"] = midi_path

            wav_path = OUTPUT_AUDIO_DIR / f"{stem}.wav"
            sf = find_soundfont(SOUNDFONTS_DIR)
            wav_ok = False
            if sf is not None:
                wav_ok = render_wav_with_fluidsynth(midi_path, wav_path, sf)
            st.session_state["wav_path"] = wav_path if wav_ok else None

        st.divider()
        st.subheader("Audio output")
        midi_path = st.session_state.get("midi_path")
        wav_path = st.session_state.get("wav_path")

        if midi_path and Path(midi_path).is_file():
            if wav_path and Path(wav_path).is_file():
                st.audio(str(wav_path))
            else:
                st.info(
                    "WAV preview not available. Add a `.sf2` under `assets/soundfonts/` and install "
                    "`fluidsynth` for in-browser audio, or download the MIDI below."
                )
            b1, b2 = st.columns(2)
            with b1:
                st.download_button(
                    label="Download MIDI",
                    data=Path(midi_path).read_bytes(),
                    file_name=Path(midi_path).name,
                    mime="audio/midi",
                )
            with b2:
                if wav_path and Path(wav_path).is_file():
                    st.download_button(
                        label="Download WAV",
                        data=Path(wav_path).read_bytes(),
                        file_name=Path(wav_path).name,
                        mime="audio/wav",
                    )
                else:
                    st.caption("WAV download appears when rendering succeeds.")
        else:
            st.caption("Use **Generate Music** to create MIDI and optional WAV.")

        st.divider()
        st.subheader("How it works")
        st.markdown(HOW_IT_WORKS)
    else:
        st.info("Upload an image to see analysis and composition mapping.")


if __name__ == "__main__":
    main()
