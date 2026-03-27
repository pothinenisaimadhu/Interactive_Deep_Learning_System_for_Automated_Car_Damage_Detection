import streamlit as st
import cv2
import tempfile
import os
from ultralytics import YOLO

CONFIDENCE_THRESHOLD = 0.5

@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")  # fallback to default if custom model not present

def get_model():
    model_path = "models/allyolov8best.pt"
    if os.path.exists(model_path):
        return YOLO(model_path)
    return load_model()

model = get_model()

def process_frame(frame):
    results = model(frame)[0]
    current_labels = set()

    if hasattr(results, 'boxes'):
        for box in results.boxes:
            conf = box.conf[0].item()
            if conf >= CONFIDENCE_THRESHOLD:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls = int(box.cls[0].item())
                label = model.names[cls]
                current_labels.add(label)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    return frame, current_labels

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

    # Save uploaded file to a temp path
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    cap = cv2.VideoCapture(tmp_path)
    stframe = st.empty()
    col1, col2 = st.columns(2)
    all_detected_labels = set()
    current_label_box = col1.empty()
    all_label_box = col2.empty()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        processed_frame, current_labels = process_frame(frame)
        all_detected_labels.update(current_labels)

        frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
        stframe.image(frame_rgb, channels="RGB")

        current_label_box.markdown(
            f"### 📌 Current Labels Detected:<br>{format_labels(current_labels, '#FF9800', newline=True)}",
            unsafe_allow_html=True
        )
        all_label_box.markdown(
            f"### ✅ All Labels Detected So Far:<br>{format_labels(all_detected_labels, '#4CAF50', newline=True)}",
            unsafe_allow_html=True
        )

    cap.release()
    os.unlink(tmp_path)
    st.success("🎉 Video Processing Completed!")

if __name__ == "__main__":
    main()
