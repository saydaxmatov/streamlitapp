import streamlit as st
from ultralytics import YOLO
import cv2
import av
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase

@st.cache_resource
def load_model():
    model = YOLO("./best.pt")
    return model

model = load_model()

conf_threshold = st.sidebar.slider("Pastroq ishonch darajasi (Confidence threshold)", 0.0, 1.0, 0.5, 0.01)
iou_threshold = st.sidebar.slider("IOU threshold", 0.0, 1.0, 0.6, 0.01)

st.title("🍅 Kamera orqali real vaqt pomidor aniqlash")

if model is None:
    st.error("Model yuklab bo'lmadi! Iltimos, model faylini tekshiring.")
    st.stop()

class TomatoDetector(VideoProcessorBase):
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        # Modelga kiritish uchun tasvirni kichraytirish (tezlik uchun)
        small_img = cv2.resize(img, (640, 360))
        # Modelni ishga tushurish
        results = model.predict(small_img, conf=conf_threshold, iou=iou_threshold)
        result = results[0]
        # Belgilangan tasvirni olish
        annotated_img = result.plot()
        # Asl o'lchamga qayta o'lchash
        annotated_img = cv2.resize(annotated_img, (img.shape[1], img.shape[0]))
        return av.VideoFrame.from_ndarray(annotated_img, format="bgr24")

webrtc_streamer(
    key="tomato_live_detection",
    video_processor_factory=TomatoDetector,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)

