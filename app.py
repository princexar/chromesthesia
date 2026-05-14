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
from src.semantic_interpreter import interpret_scene

PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUT_MIDI_DIR = PROJECT_ROOT / "outputs" / "midi"
OUTPUT_AUDIO_DIR = PROJECT_ROOT / "outputs" / "audio"
SOUNDFONTS_DIR = PROJECT_ROOT / "assets" / "soundfonts"

HOW_IT_WORKS = """
Chromesthesia builds one **composition plan** from **two parallel readings** of your image, then
uses that plan to roll a short **MIDI** phrase (and optionally **WAV** if you have a SoundFont and
`fluidsynth`).

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

#### 3. What you hear

**Generate Music** samples notes inside the chosen **key**, **mode**, **tempo**, **rhythm** setting,
and **pitch span**. Classic and neural influences are **combined**, not switched off—so neither side
fully overrides the other unless one signal is very strong.

**Audio:** in-browser playback needs a **WAV** rendered with **fluidsynth** and a **`.sf2`** under
`assets/soundfonts/`; otherwise download the **MIDI** and play it in any sequencer.
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
) -> None:
    """Transparent left-to-right *process* as a vertical pipeline (not a network graph)."""
    st.divider()
    st.subheader("How the Image Becomes Music")
    st.caption(
        "A **translation pipeline**: pixels and a frozen vision model produce **numbers**; those numbers "
        "are **mapped** to musical parameters; a small **rule-based** program writes MIDI. "
        "The neural net does **not** compose or improvise—it only supplies part of the visual signal."
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

    # --- 3. Learned embedding ---------------------------------------------------
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

    # --- 4. Music mapping -------------------------------------------------------
    _pipeline_card(
        "Stage 4",
        "Music mapping (blend of Stage 2 + Stage 3)",
        [
            f"<b>Key</b> — {comp.key} (hue from pixels, unchanged by the net)",
            f"<b>Mode</b> — {comp.mode} (brightness / warmth heuristics)",
            f"<b>Tempo</b> — {comp.tempo_bpm} BPM (classic tempo blended with energy score)",
            f"<b>Rhythm complexity</b> — {comp.rhythm_complexity} (edges blended with variation score)",
            f"<b>Note span</b> — MIDI {comp.midi_low}–{comp.midi_high} (contrast/detail blended with peak activation)",
        ],
    )
    _pipeline_arrow()

    # --- 5. Composition engine --------------------------------------------------
    _pipeline_card(
        "Stage 5",
        "Composition engine (small program, not a generative model)",
        [
            "<b>Melody generation</b> — short sequence of random choices inside the mapped constraints",
            f"<b>Scale selection</b> — diatonic pitches for {comp.key} {comp.mode} inside the MIDI span",
            "<b>MIDI creation</b> — note-on / note-off events written with <code>mido</code>",
        ],
    )
    _pipeline_arrow()

    # --- 6. Audio output ----------------------------------------------------------
    _pipeline_card(
        "Stage 6",
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
        if st.session_state.get("_scene_cache_key") != upload_key or "scene_rankings" not in st.session_state:
            with st.spinner("Loading CLIP and scoring prompts…"):
                scene_rankings = interpret_scene(image)
            st.session_state["_scene_cache_key"] = upload_key
            st.session_state["scene_rankings"] = scene_rankings
        else:
            scene_rankings = st.session_state["scene_rankings"]
        st.caption(
            "Local **CLIP** (Hugging Face `transformers`): your image is compared to the fixed phrases "
            "below. Scores are **softmax over this list only**—they rank the prompts relative to each "
            "other, not absolute truth. **Not used for MIDI yet.**"
        )
        for rank, (label, score) in enumerate(scene_rankings, start=1):
            st.write(f"{rank}. **{score * 100:.1f}%** — {label}")

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

        comp = plan_composition(features, ml_summary)
        st.session_state["composition"] = comp

        st.divider()
        st.subheader("Generated composition")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.write("**Key**", comp.key)
            st.write("**Tempo**", f"{comp.tempo_bpm} BPM")
            st.caption(f"MIDI pitch span **{comp.midi_low}–{comp.midi_high}** (image + neural blend)")
        with c2:
            st.write("**Mode**", comp.mode)
            st.write("**Rhythm complexity**", comp.rhythm_complexity)
        with c3:
            st.write("**Melody description**")
            st.write(comp.melody_description)

        render_translation_pipeline(image, features, ml_summary, emb_dim, comp)

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
