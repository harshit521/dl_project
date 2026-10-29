"""Streamlit demo for AI photo restoration and captioning."""

from __future__ import annotations

import io
import json
import sys
import zipfile
from pathlib import Path

import streamlit as st
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.enhance import is_low_saturation  # noqa: E402
from src.pipeline import Pipeline, PipelineOptions  # noqa: E402
from src.preprocess import image_to_bytes, limit_image, open_image  # noqa: E402

st.set_page_config(
    page_title="AI Photo Restoration & Captioning",
    page_icon="🖼️",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource(show_spinner=False)
def get_pipeline() -> Pipeline:
    return Pipeline()


def preview(image: Image.Image, max_side: int = 1400) -> Image.Image:
    return limit_image(image.copy(), max_side)


def process_upload(
    data: bytes,
    options: PipelineOptions,
    filename: str = "upload",
):
    pipeline = get_pipeline()
    result = pipeline.process(data, options=options)
    return result


def show_result(result, filename: str = "restored") -> None:
    before = preview(result.before)
    after = preview(result.after)

    tab_slider, tab_side = st.tabs(["Before / after slider", "Side by side"])
    with tab_slider:
        try:
            from streamlit_image_comparison import image_comparison

            image_comparison(
                img1=before,
                img2=after,
                label1="Before",
                label2="After",
                width=760,
                in_memory=True,
                make_responsive=True,
            )
        except Exception as exc:  # pragma: no cover - component/CDN issues
            st.warning(f"Slider unavailable ({exc}); showing side-by-side view.")
            left, right = st.columns(2)
            left.image(before, caption="Before", width="stretch")
            right.image(after, caption="After", width="stretch")

    with tab_side:
        left, right = st.columns(2)
        left.image(before, caption="Before", width="stretch")
        right.image(after, caption="After", width="stretch")

    if result.caption:
        st.subheader("Caption")
        st.code(result.caption, language=None)

    download_col, caption_col, json_col = st.columns(3)
    download_col.download_button(
        "Download restored image",
        data=image_to_bytes(result.after, "PNG"),
        file_name=f"{Path(filename).stem}_restored.png",
        mime="image/png",
        width="stretch",
    )
    if result.caption:
        caption_col.download_button(
            "Download caption",
            data=result.caption.encode("utf-8"),
            file_name=f"{Path(filename).stem}_caption.txt",
            mime="text/plain",
            width="stretch",
        )
    json_col.download_button(
        "Download metadata",
        data=json.dumps(result.meta, indent=2).encode("utf-8"),
        file_name=f"{Path(filename).stem}_metadata.json",
        mime="application/json",
        width="stretch",
    )

    with st.expander("Processing details"):
        st.json(result.meta)


def sidebar_options() -> PipelineOptions:
    st.sidebar.header("Restoration settings")
    scale = st.sidebar.radio(
        "Upscale factor",
        options=(2, 4),
        index=1,
        horizontal=True,
        help="4x produces the largest image; 2x is faster.",
    )
    max_side = st.sidebar.select_slider(
        "Maximum input size",
        options=(512, 768, 1024, 1280),
        value=768,
        help="The output is this size multiplied by the upscale factor.",
    )
    face_enhance = st.sidebar.checkbox(
        "Enhance faces",
        value=True,
        help="Detects faces and sharpens them with GFPGAN when available, "
        "otherwise with Real-ESRGAN face crops.",
    )
    auto_enhance = st.sidebar.checkbox(
        "Auto color / contrast",
        value=True,
        help="Conservative color-cast removal, contrast stretch, and local contrast.",
    )
    scratch_removal = st.sidebar.checkbox(
        "Reduce dust & scratches (experimental)",
        value=False,
        help="Inpaints thin bright/dark damage before upscaling. Aggressive on "
        "textured areas; compare with it off.",
    )
    colorize = st.sidebar.checkbox(
        "Colorize black-and-white / faded photos",
        value=False,
        help="Uses the SIGGRAPH 2017 colorizer. Best for grayscale or sepia input.",
    )
    caption = st.sidebar.checkbox("Generate caption", value=True)

    st.sidebar.divider()
    st.sidebar.caption(
        "All processing runs locally on this machine. Uploaded images are "
        "not sent to a server or stored permanently."
    )
    return PipelineOptions(
        scale=scale,
        face_enhance=face_enhance,
        caption=caption,
        auto_enhance=auto_enhance,
        scratch_removal=scratch_removal,
        colorize=colorize,
        max_side=max_side,
    )


def restore_tab(options: PipelineOptions) -> None:
    st.title("AI Photo Restoration & Captioning")
    st.caption(
        "Upscale, denoise, sharpen faces, correct faded color, and describe a photo — "
        "using pretrained models running locally on Apple Silicon."
    )

    sample_col, upload_col = st.columns([1, 2])
    with sample_col:
        sample_name = st.selectbox(
            "Or try a sample",
            ("None", "Portrait (synthetic old photo)", "Landscape (synthetic old photo)"),
        )
    with upload_col:
        uploaded = st.file_uploader(
            "Upload a photo",
            type=["jpg", "jpeg", "png", "webp", "bmp"],
            help="Maximum recommended size: 15 MB / 12 MP.",
        )

    source_bytes: bytes | None = None
    filename = "sample"
    if uploaded is not None:
        source_bytes = uploaded.getvalue()
        filename = uploaded.name
    elif sample_name != "None":
        sample_file = (
            "sample_portrait_old.jpg"
            if sample_name.startswith("Portrait")
            else "sample_scene_old.jpg"
        )
        sample_path = ROOT / "data" / "demo" / sample_file
        if sample_path.exists():
            source_bytes = sample_path.read_bytes()
            filename = sample_file
        else:
            st.warning(
                "Sample image not found. Run: python scripts/make_synthetic_old.py "
                "--input assets --out data/demo --all"
            )

    if source_bytes is not None:
        try:
            source_image = open_image(source_bytes)
        except Exception as exc:
            st.error(f"Could not read image: {exc}")
            return

        info_cols = st.columns(3)
        info_cols[0].metric("Input size", f"{source_image.width} × {source_image.height}")
        info_cols[1].metric("Est. output", f"{source_image.width * options.scale} × {source_image.height * options.scale}")
        info_cols[2].metric("Device", get_pipeline().manager.device.upper())

        if options.colorize and not is_low_saturation(source_image):
            st.info(
                "This image already has color. Colorization preserves luminance and "
                "may still shift the palette; turn it off if the result looks unnatural."
            )

        if st.button("Restore photo", type="primary", width="stretch"):
            cache_key = (
                hash(source_bytes),
                options.scale,
                options.face_enhance,
                options.auto_enhance,
                options.scratch_removal,
                options.colorize,
                options.caption,
                options.max_side,
            )
            cached = st.session_state.get("last_result")
            if cached and cached.get("key") == cache_key:
                result = cached["result"]
                filename = cached["filename"]
            else:
                with st.spinner("Restoring, enhancing, and captioning…"):
                    try:
                        result = process_upload(source_bytes, options, filename)
                    except Exception as exc:
                        st.error(f"Processing failed: {exc}")
                        st.exception(exc)
                        return
                st.session_state["last_result"] = {
                    "key": cache_key,
                    "result": result,
                    "filename": filename,
                }
            show_result(result, filename)
        else:
            st.image(source_image, caption="Input preview", width="stretch")

        with st.expander("What each stage does"):
            st.markdown(
                """
- **Real-ESRGAN** upscales and removes compression/blur artifacts.
- **Face enhancement** detects faces and restores them with GFPGAN or a
  Real-ESRGAN face-crop fallback.
- **Auto color/contrast** neutralizes a color cast and stretches tones.
- **Dust & scratch reduction** (optional) inpaints thin damage before upscaling.
- **Colorization** predicts plausible colors for grayscale/sepia photos.
- **BLIP** generates a one-sentence caption.
                """
            )


def batch_tab(options: PipelineOptions) -> None:
    st.header("Batch restore")
    st.caption("Process several photos with the current sidebar settings.")
    files = st.file_uploader(
        "Upload images",
        type=["jpg", "jpeg", "png", "webp", "bmp"],
        accept_multiple_files=True,
        key="batch_uploader",
    )
    if not files:
        return
    if len(files) > 12:
        st.warning("Please process at most 12 images per batch on this machine.")
        files = files[:12]

    if st.button("Process batch", type="primary"):
        pipeline = get_pipeline()
        progress = st.progress(0.0, text="Starting…")
        archive = io.BytesIO()
        captions: list[str] = []
        failures: list[str] = []

        with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as bundle:
            for index, uploaded in enumerate(files, start=1):
                progress.progress(
                    (index - 1) / len(files),
                    text=f"Processing {uploaded.name} ({index}/{len(files)})…",
                )
                try:
                    result = pipeline.process(uploaded.getvalue(), options=options)
                    bundle.writestr(
                        f"{Path(uploaded.name).stem}_restored.png",
                        image_to_bytes(result.after, "PNG"),
                    )
                    if result.caption:
                        captions.append(f"{uploaded.name}\t{result.caption}")
                except Exception as exc:
                    failures.append(f"{uploaded.name}: {exc}")
            if captions:
                bundle.writestr(
                    "captions.tsv",
                    "filename\tcaption\n" + "\n".join(captions) + "\n",
                )

        progress.progress(1.0, text="Done")
        st.success(f"Processed {len(files) - len(failures)} of {len(files)} images.")
        for failure in failures:
            st.error(failure)
        st.download_button(
            "Download ZIP",
            data=archive.getvalue(),
            file_name="restored_batch.zip",
            mime="application/zip",
            width="stretch",
        )


def about_tab() -> None:
    st.header("About this project")
    st.markdown(
        """
### Pipeline
1. Normalize image (EXIF orientation, RGB, size limits).
2. Tiled **Real-ESRGAN** super-resolution (x2 or x4).
3. Face detection with **YuNet**, enhancement with **GFPGAN** (optional) or
   Real-ESRGAN face crops.
4. Conservative auto color/contrast correction.
5. Optional **SIGGRAPH 2017** colorization.
6. **BLIP** captioning.

### Models
| Stage | Model | License |
|---|---|---|
| Super-resolution | Real-ESRGAN x4plus | BSD-3-Clause |
| Face restoration | GFPGAN v1.4 (optional) | Apache-2.0 |
| Face detection | OpenCV YuNet | Apache-2.0 |
| Captioning | BLIP base | BSD-3-Clause |
| Colorization | SIGGRAPH 2017 | BSD-2-Clause |

### Limitations
- Real-ESRGAN can sharpen and invent texture; it is not a forensic tool.
- Face enhancement can alter identity. Keep the toggle available.
- Scratch and dust removal are not included; existing scratches often remain.
- Colorization predicts plausible but not historically accurate colors.
- Captions can hallucinate details. Treat them as suggestions.
- Synthetic old photos are easier than real historical prints; evaluate accordingly.

### Privacy
Processing is local. The app does not persist uploads; batch results are held in
memory and downloaded by the user.
        """
    )


def main() -> None:
    options = sidebar_options()
    restore, batch, about = st.tabs(["Restore", "Batch", "About"])
    with restore:
        restore_tab(options)
    with batch:
        batch_tab(options)
    with about:
        about_tab()


if __name__ == "__main__":
    main()
