import streamlit as st
import tempfile
import os
import numpy as np
from PIL import Image, ImageDraw
from ultralytics import YOLO

CONFIDENCE_THRESHOLD = 0.5

@st.cache_resource
def load_model():
    from huggingface_hub import hf_hub_download
    filename = st.secrets["HF_FILENAME"]
    repo_id  = st.secrets["HF_REPO_ID"]
    token    = st.secrets["HF_TOKEN"]
    model_path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        token=token,
        repo_type="model",
    )
    return YOLO(model_path)

model = load_model()

def process_image(img: Image.Image):
    frame_array = np.array(img.convert("RGB"))
    results = model(frame_array)[0]
    current_labels = set()
    draw = ImageDraw.Draw(img)

    if hasattr(results, 'boxes'):
        for box in results.boxes:
            conf = box.conf[0].item()
            if conf >= CONFIDENCE_THRESHOLD:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls = int(box.cls[0].item())
                label = f"{model.names[cls]} {conf:.0%}"
                current_labels.add(model.names[cls])
                draw.rectangle([x1, y1, x2, y2], outline=(0, 255, 0), width=3)
                draw.rectangle([x1, y1 - 18, x1 + len(label) * 7, y1], fill=(0, 255, 0))
                draw.text((x1 + 2, y1 - 16), label, fill=(0, 0, 0))

    return img, current_labels

def format_labels(label_set, color):
    if label_set:
        return " ".join(
            f"<span style='background-color:{color}; color:white; padding:6px 12px; "
            f"border-radius:8px; font-size:14px; margin:4px; display:inline-block;'>{label}</span>"
            for label in sorted(label_set)
        )
    return "<span style='color:gray; font-size:14px;'>None detected</span>"

def main():
    st.title("🚗 Car Damage Detection")
    st.markdown("### Real-time YOLO-based Damage Detection")

    mode = st.radio("Select Input Mode", ["📷 Camera", "🎥 Upload Video"], horizontal=True)

    col1, col2 = st.columns(2)
    all_detected_labels = set()

    # ── CAMERA MODE ──────────────────────────────────────────────
    if mode == "📷 Camera":
        camera_image = st.camera_input("Take a photo of the damaged car")

        if camera_image:
            img = Image.open(camera_image)
            annotated_img, labels = process_image(img)
            all_detected_labels.update(labels)

            st.image(annotated_img, caption="Detection Result", use_container_width=True)

            col1.markdown(
                f"### 📌 Detected Damage:<br>{format_labels(labels, '#FF9800')}",
                unsafe_allow_html=True
            )
            col2.markdown(
                f"### ✅ All Detected:<br>{format_labels(all_detected_labels, '#4CAF50')}",
                unsafe_allow_html=True
            )

    # ── VIDEO UPLOAD MODE ─────────────────────────────────────────
    else:
        uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov"])

        if uploaded_file is None:
            st.info("Please upload a video file to begin detection.")
            return

        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name

        try:
            results_gen = model(tmp_path, stream=True, conf=CONFIDENCE_THRESHOLD)

            stframe = st.empty()
            current_label_box = col1.empty()
            all_label_box = col2.empty()

            for result in results_gen:
                current_labels = set()
                frame_rgb = result.orig_img[:, :, ::-1].copy()
                img = Image.fromarray(frame_rgb)
                draw = ImageDraw.Draw(img)

                if hasattr(result, 'boxes'):
                    for box in result.boxes:
                        conf = box.conf[0].item()
                        if conf >= CONFIDENCE_THRESHOLD:
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            cls = int(box.cls[0].item())
                            label = f"{model.names[cls]} {conf:.0%}"
                            current_labels.add(model.names[cls])
                            draw.rectangle([x1, y1, x2, y2], outline=(0, 255, 0), width=3)
                            draw.rectangle([x1, y1 - 18, x1 + len(label) * 7, y1], fill=(0, 255, 0))
                            draw.text((x1 + 2, y1 - 16), label, fill=(0, 0, 0))

                all_detected_labels.update(current_labels)
                stframe.image(img, use_container_width=True)

                current_label_box.markdown(
                    f"### 📌 Current Labels:<br>{format_labels(current_labels, '#FF9800')}",
                    unsafe_allow_html=True
                )
                all_label_box.markdown(
                    f"### ✅ All Labels So Far:<br>{format_labels(all_detected_labels, '#4CAF50')}",
                    unsafe_allow_html=True
                )

            st.success("🎉 Video Processing Completed!")

        finally:
            os.unlink(tmp_path)

if __name__ == "__main__":
    main()
