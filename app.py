import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2
import tempfile

# Заголовок и настройки
st.set_page_config(
    page_title="Обнаружение помидоров",
    page_icon="🍅",
    layout="wide"
)

# Добавление CSS стилей
st.markdown("""
<style>
    .main-header { font-size: 3rem; color: #ff4b4b; text-align: center; }
    .subheader { font-size: 1.5rem; color: #ff6b6b; }
    .stButton>button { background-color: #ff4b4b; color: white; }
    .stSlider>div>div>div { background-color: #ff4b4b; }
</style>
""", unsafe_allow_html=True)

# Заголовок
st.markdown('<h1 class="main-header">🍅 Обнаружение помидоров</h1>', unsafe_allow_html=True)
st.markdown("### Обнаружение и подсчет помидоров с помощью модели YOLO")

# Загрузка модели
@st.cache_resource
def load_model():
    try:
        model = YOLO("./best.pt")
        return model
    except:
        st.error("Файл модели не найден. Пожалуйста, проверьте путь к файлу модели.")
        return None

model = load_model()

# Боковая панель с настройками
st.sidebar.header("Настройки")
conf_threshold = st.sidebar.slider("Порог доверия", 0.0, 1.0, 0.5, 0.01)
iou_threshold = st.sidebar.slider("Порог IOU", 0.0, 1.0, 0.6, 0.01)

# Основной контент — вкладки
tab1, tab2, tab3 = st.tabs(["📷 Загрузить изображение", "🎥 Загрузить видео", "📹 Камера"])

# Tab 1: Image
with tab1:
    uploaded_image = st.file_uploader("Загрузите изображение помидоров", type=['jpg', 'jpeg', 'png'])
    if uploaded_image is not None and model is not None:
        image = Image.open(uploaded_image).convert('RGB')
        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption="Загруженное изображение", use_column_width=True)
        with st.spinner("Модель анализирует изображение..."):
            results = model.predict(np.array(image), conf=conf_threshold, iou=iou_threshold)
            result = results[0]
            annotated_image = result.plot()
            annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
        with col2:
            st.image(annotated_image, caption="Обнаруженные помидоры", use_column_width=True)
            detections = len(result.boxes)
            st.success(f"✅ Обнаружено {detections} помидоров!")
            if detections > 0:
                with st.expander("Подробная информация"):
                    for i, box in enumerate(result.boxes):
                        conf = box.conf[0].item()
                        cls = result.names[box.cls[0].item()]
                        st.write(f"{i+1}. {cls} - вероятность: {conf:.2f}")

# Tab 2: Video
with tab2:
    uploaded_video = st.file_uploader("Загрузите видео с помидорами", type=['mp4', 'mov', 'avi'])
    if uploaded_video is not None and model is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_video.read())
        st.info("Видео обрабатывается... Это может занять некоторое время")
        results = model.predict(tfile.name, conf=conf_threshold, iou=iou_threshold, save=True, project=".", name="output", exist_ok=True)
        try:
            video_results = list(results)[0]
            output_video_path = video_results.save_dir
            st.success("Видео успешно обработано!")
            st.video("output/output_video.mp4")
        except:
            st.error("Ошибка при обработке видео")

# Tab 3: Camera
with tab3:
    st.info("Камера включается через ваш браузер")
    run_camera = st.checkbox("Запустить камеру")
    FRAME_WINDOW = st.image([])
    COORDS_WINDOW = st.empty()  # Koordinatalar uchun joy
    if run_camera and model is not None:
        cap = cv2.VideoCapture(0)
        while run_camera:
            ret, frame = cap.read()
            if not ret:
                st.error("Не удалось получить кадр с камеры")
                break
            
            # Agar frame 4-kanal bo'lsa RGB ga o'tkazish
            if frame.shape[2] == 4:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
            
            results = model.predict(frame, conf=conf_threshold, iou=iou_threshold)
            result = results[0]
            annotated_frame = result.plot()
            annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            FRAME_WINDOW.image(annotated_frame)

            # Pomidor koordinatalarini chiqarish
            coords_text = ""
            for i, box in enumerate(result.boxes):
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                coords_text += f"{i+1}. x1={int(x1)}, y1={int(y1)}, x2={int(x2)}, y2={int(y2)}\n"
            if coords_text:
                COORDS_WINDOW.text(f"📌 Координаты помидоров:\n{coords_text}")
            else:
                COORDS_WINDOW.text("📌 Координаты помидоров: не обнаружено")
        cap.release()
