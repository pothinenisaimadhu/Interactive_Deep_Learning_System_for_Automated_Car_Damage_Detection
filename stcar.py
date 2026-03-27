import streamlit as st
import tempfile
import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO

CONFIDENCE_THRESHOLD = 0.5

@st.cache_resource
def load_model():
    filename = st.secrets.get("HF_FILENAME", "allyolov8best.pt")
    model_path = f"/tmp/{filename}"
    if not os.path.exists(model_path):
        from huggingface_hub import hf_hub_download
        # Try model repo first, then dataset repo
        for repo_type in ["model", "dataset"]:
            try:
                hf_hub_download(
                    repo_id=st.secrets["HF_REPO_ID"],
                    filename=filename,
                    token=st.secrets["HF_TOKEN"],
                    repo_type=repo_type,
                    local_dir="/tmp"
                )
                break
            except Exception:
                continue
    return YOLO(model_path)

model = load_model()

def process_frame(frame_array):
    results = model(frame_array)[0]
    current_labels = set()
    img = Image.fromarray(frame_array)
    draw = ImageDraw.Draw(img)

    if hasattr(results, 'boxes'):
        for box in results.boxes:
            conf = box.conf[0].item()
            if conf >= CONFIDENCE_THRESHOLD:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls = int(box.cls[0].item())
                label = model.names[cls]
                current_labels.add(label)
                draw.rectangle([x1, y1, x2, y2], outline=(0, 255, 0), width=2)
                draw.text((x1, y1 - 10), label, fill=(0, 255, 0))

    return np.array(img), current_labels

def format_labels(label_set, color, newline=False):
    if label_set:
        separator = "<br>" if newline else " "
        return separator.join(
            f"<span style='background-color:{color}; color:white; padding:6px 12px; border-radius:8px; font-size:14px; margin:4px; display:inline-block;'>{label}</span>"
            for label in sorted(label_set)
        )
    return "<span style='color:gray; font-size:14px;'>None</span>"

def main():
    st.title("🚗 Car Damage Detection")
    st.markdown("### Real-time YOLO-based Damage Detection")

    uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov"])

    if uploaded_file is None:
        st.info("Please upload a video file to begin detection.")
        return

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    try:
        # Use ultralytics built-in video reading (uses its own cv2 internally)
        results_gen = model(tmp_path, stream=True, conf=CONFIDENCE_THRESHOLD)

        stframe = st.empty()
        col1, col2 = st.columns(2)
        all_detected_labels = set()
        current_label_box = col1.empty()
        all_label_box = col2.empty()

        for result in results_gen:
            current_labels = set()
            frame_array = result.orig_img  # BGR numpy array from ultralytics

            # Convert BGR to RGB
            frame_rgb = frame_array[:, :, ::-1].copy()
            img = Image.fromarray(frame_rgb)
            draw = ImageDraw.Draw(img)

            if hasattr(result, 'boxes'):
                for box in result.boxes:
                    conf = box.conf[0].item()
                    if conf >= CONFIDENCE_THRESHOLD:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        cls = int(box.cls[0].item())
                        label = model.names[cls]
                        current_labels.add(label)
                        draw.rectangle([x1, y1, x2, y2], outline=(0, 255, 0), width=2)
                        draw.text((x1, max(0, y1 - 10)), label, fill=(0, 255, 0))

            all_detected_labels.update(current_labels)
            stframe.image(img)

            current_label_box.markdown(
                f"### 📌 Current Labels:<br>{format_labels(current_labels, '#FF9800', newline=True)}",
                unsafe_allow_html=True
            )
            all_label_box.markdown(
                f"### ✅ All Labels So Far:<br>{format_labels(all_detected_labels, '#4CAF50', newline=True)}",
                unsafe_allow_html=True
            )

        st.success("🎉 Video Processing Completed!")

    finally:
        os.unlink(tmp_path)

if __name__ == "__main__":
    main()
