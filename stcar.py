import streamlit as st
import tempfile
import os
import numpy as np
from PIL import Image, ImageDraw
from ultralytics import YOLO
import av
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration

CONFIDENCE_THRESHOLD = 0.5

RTC_CONFIG = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})

@st.cache_resource
def load_model():
    from huggingface_hub import hf_hub_download
    model_path = hf_hub_download(
        repo_id=st.secrets["HF_REPO_ID"],
        filename=st.secrets["HF_FILENAME"],
        token=st.secrets["HF_TOKEN"],
        repo_type="model",
    )
    return YOLO(model_path)

model = load_model()

def annotate(img: Image.Image):
    frame_array = np.array(img.convert("RGB"))
    results = model(frame_array)[0]
    labels = set()
    draw = ImageDraw.Draw(img)
    if hasattr(results, "boxes"):
        for box in results.boxes:
            conf = box.conf[0].item()
            if conf >= CONFIDENCE_THRESHOLD:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls = int(box.cls[0].item())
                text = f"{model.names[cls]} {conf:.0%}"
                labels.add(model.names[cls])
                draw.rectangle([x1, y1, x2, y2], outline=(0, 255, 0), width=3)
                draw.rectangle([x1, y1 - 18, x1 + len(text) * 7, y1], fill=(0, 200, 0))
                draw.text((x1 + 2, y1 - 16), text, fill=(255, 255, 255))
    return img, labels

def label_html(label_set, color):
    if not label_set:
        return "<span style='color:gray'>None detected</span>"
    return " ".join(
        f"<span style='background:{color};color:white;padding:5px 10px;"
        f"border-radius:8px;margin:3px;display:inline-block'>{l}</span>"
        for l in sorted(label_set)
    )

# ── WebRTC Video Processor ────────────────────────────────────────
class DamageDetector(VideoProcessorBase):
    def __init__(self):
        self.labels = set()

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_image()  # PIL Image
        annotated, labels = annotate(img)
        self.labels = labels
        return av.VideoFrame.from_image(annotated)

def main():
    st.title("🚗 Car Damage Detection")

    mode = st.radio("Input Mode", ["📷 Live Camera", "🖼 Single Photo", "🎥 Upload Video"], horizontal=True)

    # ── LIVE CAMERA (WebRTC) ──────────────────────────────────────
    if mode == "📷 Live Camera":
        st.info("Allow camera access when prompted. Detection runs on every frame in real-time.")

        ctx = webrtc_streamer(
            key="car-damage-live",
            video_processor_factory=DamageDetector,
            rtc_configuration=RTC_CONFIG,
            media_stream_constraints={"video": True, "audio": False},
        )

        col1, col2 = st.columns(2)
        cur_box = col1.empty()
        all_box = col2.empty()

        if ctx.video_processor:
            import time
            all_labels: set = set()
            while ctx.state.playing:
                cur = ctx.video_processor.labels
                all_labels.update(cur)
                cur_box.markdown(f"### 📌 Current Labels:<br>{label_html(cur, '#FF9800')}", unsafe_allow_html=True)
                all_box.markdown(f"### ✅ All Labels So Far:<br>{label_html(all_labels, '#4CAF50')}", unsafe_allow_html=True)
                time.sleep(0.1)

    # ── SINGLE PHOTO ──────────────────────────────────────────────
    elif mode == "🖼 Single Photo":
        snap = st.camera_input("Take a photo of the damaged car")
        if snap:
            img = Image.open(snap)
            annotated, labels = annotate(img)
            st.image(annotated, use_container_width=True)
            col1, col2 = st.columns(2)
            col1.markdown(f"### 📌 Current Labels:<br>{label_html(labels, '#FF9800')}", unsafe_allow_html=True)
            if "photo_all_labels" not in st.session_state:
                st.session_state.photo_all_labels = set()
            st.session_state.photo_all_labels.update(labels)
            col2.markdown(f"### ✅ All Labels So Far:<br>{label_html(st.session_state.photo_all_labels, '#4CAF50')}", unsafe_allow_html=True)

    # ── VIDEO UPLOAD ──────────────────────────────────────────────
    else:
        uploaded = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])
        if not uploaded:
            st.info("Upload a video file to begin detection.")
            return

        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            tmp.write(uploaded.read())
            tmp_path = tmp.name

        try:
            stframe    = st.empty()
            col1, col2 = st.columns(2)
            cur_box    = col1.empty()
            all_box    = col2.empty()
            all_labels: set = set()

            for result in model(tmp_path, stream=True, conf=CONFIDENCE_THRESHOLD):
                cur_labels = set()
                frame_rgb  = result.orig_img[:, :, ::-1].copy()
                img        = Image.fromarray(frame_rgb)
                draw       = ImageDraw.Draw(img)

                if hasattr(result, "boxes"):
                    for box in result.boxes:
                        conf = box.conf[0].item()
                        if conf >= CONFIDENCE_THRESHOLD:
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            cls  = int(box.cls[0].item())
                            text = f"{model.names[cls]} {conf:.0%}"
                            cur_labels.add(model.names[cls])
                            draw.rectangle([x1, y1, x2, y2], outline=(0, 255, 0), width=3)
                            draw.rectangle([x1, y1 - 18, x1 + len(text) * 7, y1], fill=(0, 200, 0))
                            draw.text((x1 + 2, y1 - 16), text, fill=(255, 255, 255))

                all_labels.update(cur_labels)
                stframe.image(img, use_container_width=True)
                cur_box.markdown(f"**Current:** {label_html(cur_labels, '#FF9800')}", unsafe_allow_html=True)
                all_box.markdown(f"**All so far:** {label_html(all_labels, '#4CAF50')}", unsafe_allow_html=True)

            st.success("🎉 Video Processing Completed!")
        finally:
            os.unlink(tmp_path)

if __name__ == "__main__":
    main()
