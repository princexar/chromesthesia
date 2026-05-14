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
Chromesthesia builds one **composition plan** from **three parallel readings** of your image—classic
pixels, a ResNet embedding summary, and a **CLIP scene cue** (mood, energy, scale, duration band,
texture)—then maps them into tempo, span, **length in measures**, **section form**, and **harmony
density**. A **rule-based** program writes multi-measure MIDI (melody plus optional bass line).

#### 1. Classic image analysis (explainable pixels)

These are hand-designed signals from the photo itself—easy to reason about in the **Image analysis**
panel:

- **Dominant color (hue)** → **musical key** (twelve chromatic keys around the color wheel).
- **Brightness** and **warm vs cool** hues → **major vs minor mode**.
- **Brightness** and **edge / detail** → a **tempo** starting point (brighter, busier scenes lean faster).
- **Edge / detail** → a **rhythm complexity** tier (Low / Moderate / High).
- **Contrast** and **edges** → how **wide the MIDI pitch window** can be, and extra wording in the
  **melody description** (e.g. wide leaps when contrast is high).

The **mood** line is a separate human-readable gloss on those same pixel stats; it does not drive
the MIDI math by itself.

#### 2. Neural embedding (ResNet18 backbone)

The image is also run through a pretrained **ResNet18** (classification weights, **final layer
removed**). The **512** backbone values are summarized into a few scalars (see **Neural Visual
Interpretation**). Those summaries are **not** scene labels like “beach” or “sad”; they describe the
shape of the activation vector. They are **blended** into the same plan as the classic cues:

- **Energy score** → mixed with the classic tempo idea so overall **tempo** reflects both.
- **Variation score** → mixed with the edge-based **rhythm complexity** tier.
- **Peak activation (`max_activation`)** → mixed with the classic span so **note range** (MIDI low–high)
  reflects both fine structure in the embedding and contrast/detail in the image.

#### 3. Scene composition cue (frozen CLIP)

**CLIP** ranks atmosphere prompts, then scores separate prompt sets for **intimate vs expansive**,
**short vs long feel**, and **sparse vs dense** texture. Those choices set a **target duration**
(roughly 30–45s / 60–90s / 90–180s bands), nudge **tempo** and **rhythm**, and pick **AABA**, **ABAB**,
or **through-composed** form plus **simple / moderate / rich** harmony in the MIDI engine.

#### 4. What you hear

**Generate Music** writes **many measures** of diatonic material: scene-driven **rest probability** and
subdivisions control **note density**; a second track adds **roots / fifths / arpeggiated triads** when
harmony is not *simple*. **WAV** playback still needs **fluidsynth** + a **`.sf2`** under `assets/soundfonts/`.
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
        "A **translation pipeline**: pixels, a ResNet summary, and a **CLIP scene cue** become **numbers**; "
        "those numbers are **mapped** to musical parameters; a **rule-based** program writes multi-measure MIDI. "
        "No model is trained here—frozen nets only suggest structure."
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

    # --- 5. Music mapping -------------------------------------------------------
    _pipeline_card(
        "Stage 5",
        "Music mapping (pixels + ResNet + scene cue)",
        [
            f"<b>Key</b> — {comp.key} (hue from pixels)",
            f"<b>Mode</b> — {comp.mode} (brightness / warmth heuristics)",
            f"<b>Tempo</b> — {comp.tempo_bpm} BPM (pixels + ResNet energy + scene energy)",
            f"<b>Rhythm</b> — {comp.rhythm_complexity} (edges + variation + scene texture)",
            f"<b>Note span</b> — MIDI {comp.midi_low}–{comp.midi_high} (contrast + ML peak + scene scale)",
            f"<b>Measures / form</b> — {comp.measures} bars, **{comp.section_form}**",
            f"<b>Harmony depth</b> — {comp.harmonic_complexity}",
        ],
    )
    _pipeline_arrow()

    # --- 6. Composition engine --------------------------------------------------
    _pipeline_card(
        "Stage 6",
        "Composition engine (rule-based, not a generative model)",
        [
            f"<b>Length</b> — ~{comp.target_duration_sec:.0f}s ({comp.measures} × 4/4 measures)",
            f"<b>Melody</b> — section-aware random walk ({comp.section_form}) with scene-driven rests",
            f"<b>Scale</b> — diatonic pitches for {comp.key} {comp.mode} inside MIDI {comp.midi_low}–{comp.midi_high}",
            f"<b>Harmony track</b> — {'none (simple)' if comp.harmonic_complexity == 'simple' else comp.harmonic_complexity + ' bass / arpeggio'}",
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

        st.divider()
        st.subheader("Generated composition")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.write("**Key**", comp.key)
            st.write("**Tempo**", f"{comp.tempo_bpm} BPM")
            st.caption(f"MIDI **{comp.midi_low}–{comp.midi_high}**")
        with c2:
            st.write("**Mode**", comp.mode)
            st.write("**Rhythm**", comp.rhythm_complexity)
            st.caption(f"Texture **{comp.texture_density}**")
        with c3:
            st.write("**Length**", f"~{comp.target_duration_sec:.0f}s")
            st.write("**Measures**", f"{comp.measures} bars")
            st.caption(f"Form **{comp.section_form}**")
        with c4:
            st.write("**Harmony**", comp.harmonic_complexity)
            st.write("**Melody / scene**")
            st.caption(comp.melody_description[:220] + ("…" if len(comp.melody_description) > 220 else ""))

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
