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
    .main-header {
        font-size: 3rem;
        color: #ff4b4b;
        text-align: center;
    }
    .subheader {
        font-size: 1.5rem;
        color: #ff6b6b;
    }
    .stButton>button {
        background-color: #ff4b4b;
        color: white;
    }
    .stSlider>div>div>div {
        background-color: #ff4b4b;
    }
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
tab1, tab2 = st.tabs(["📷 Загрузить изображение", "🎥 Загрузить видео"])

with tab1:
    uploaded_image = st.file_uploader("Загрузите изображение помидоров", type=['jpg', 'jpeg', 'png'])
    
    if uploaded_image is not None and model is not None:
        # Загрузка изображения
        image = Image.open(uploaded_image)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.image(image, caption="Загруженное изображение", use_column_width=True)
        
        # Предсказание модели
        with st.spinner("Модель анализирует изображение..."):
            results = model.predict(np.array(image), conf=conf_threshold, iou=iou_threshold)
            result = results[0]
            
            # Получить изображение с аннотацией
            annotated_image = result.plot()
            annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
            
            with col2:
                st.image(annotated_image, caption="Обнаруженные помидоры", use_column_width=True)
        
        # Вывод результатов
        detections = len(result.boxes)
        st.success(f"✅ Обнаружено {detections} помидоров!")
        
        # Подробная информация о каждом объекте
        if detections > 0:
            with st.expander("Подробная информация"):
                for i, box in enumerate(result.boxes):
                    conf = box.conf[0].item()
                    cls = result.names[box.cls[0].item()]
                    st.write(f"{i+1}. {cls} - вероятность: {conf:.2f}")

with tab2:
    uploaded_video = st.file_uploader("Загрузите видео с помидорами", type=['mp4', 'mov', 'avi'])
    
    if uploaded_video is not None and model is not None:
        # Временное сохранение видео
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_video.read())
        
        # Обработка видео
        st.info("Видео обрабатывается... Это может занять некоторое время")
        
        # Предсказание для видео с сохранением результата
        results = model.predict(tfile.name, conf=conf_threshold, iou=iou_threshold, save=True, project=".", name="output", exist_ok=True)
        
        try:
            # Путь к последнему обработанному видео
            video_results = list(results)[0]
            output_video_path = video_results.save_dir
            
            # Отображение видео с результатами
            st.success("Видео успешно обработано!")
            st.video("output/output_video.mp4")
            
        except:
            st.error("Ошибка при обработке видео")

# Футер
st.markdown("---")
st.markdown("### О проекте")
st.markdown("""
Этот проект разработан для обнаружения и подсчета помидоров с помощью модели YOLO.
Используются следующие технологии:
- **YOLOv8** - модель для обнаружения объектов
- **Streamlit** - веб-интерфейс
- **OpenCV** - обработка изображений и видео
""")
