# main_ecovision.py
'''
# Aplicación Streamlit para EcoVision - Detección de latas y botellas con YOLOv8
# Autores: Jose Luis Martinez Diaz, Juan David Arroyave Ramirez, Neiberth Aponte Aristizabal, Stevens Ricardo Bohorquez Ruiz
# Fecha: 2025-10
# Licencia: Apache 2.0
'''

import streamlit as st
import os
import cv2
import datetime
from PIL import Image
from src.data.load_model import load_pytorch_model
from src.data.processing import process_frame
import pandas as pd, yaml, os

# 🔹 Para video en vivo desde navegador
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import av

# Configuración de la página
st.set_page_config(page_title="♻️👁️EcoVision", layout="wide", page_icon="♻️")

# Cargar y mostrar logo
def show_logo():
    if os.path.exists("ecovision_logo.png"):
        logo = Image.open("ecovision_logo.png")
        st.image(logo, width=200)

show_logo()

# Sidebar
def sidebar_info():
    with st.sidebar:
        st.title("💻 Autores")
        st.subheader("Desarrollado por:")
        st.markdown("Jose Luis Martinez Diaz")
        st.markdown("Juan David Arroyave Ramirez")
        st.markdown("Neiberth Aponte Aristizabal")
        st.markdown("Stevens Ricardo Bohorquez Ruiz")
        st.caption("Repositorio ECOVISION")
        st.markdown('https://github.com/davidarroyave/ecovision', unsafe_allow_html=True)
        st.caption("EcoVision ♻️👁️")
        st.markdown("---")
        st.info("Sistema de IA en PyTorch para detección de latas y botellas.")
        st.markdown("---")
        year = datetime.datetime.now().year
        st.markdown(f"©{year} Equipo EcoVision. Licencia Apache 2.0")

sidebar_info()

# Título y descripción
st.title("♻️EcoVision👁️")
st.markdown(
    """
Bienvenido a **EcoVision**: Detección de latas y botellas mediante PyTorch YOLOv8.

- **Cámara**: Detección en tiempo real
- **Métricas**: Precisión, recall, mAP
- **Informe**: Documentación del proyecto
"""
)

# Cargar modelo
model = load_pytorch_model()

# Crear pestañas
tab_camera, tab_metrics, tab_report = st.tabs(["📹 Cámara", "📊 Métricas", "🧾 Informe"])

# ------------------------------
# 📹 CÁMARA
# ------------------------------
with tab_camera:
    st.subheader("📸 Captura y Detección")

    # Selector de modo
    modo = st.radio("Selecciona modo:", ["📷 Captura de foto", "🎥 Video en vivo"], horizontal=True)
    conf = st.slider("Umbral de confianza", 0.0, 1.0, 0.5, 0.05)

    # -----------------------
    # 📷 CAPTURA DE FOTO
    # -----------------------
    if modo == "📷 Captura de foto":
        st.markdown("Usa tu cámara para tomar una foto y detectar objetos.")
        img_file = st.camera_input("Toma una foto")

        if img_file is not None:
            image = Image.open(img_file)
            st.image(image, caption="Imagen original", use_container_width=True)
            annotated = process_frame(image, model, conf)
            st.image(annotated, caption="Resultado de detección", use_container_width=True)

    # -----------------------
    # 🎥 VIDEO EN VIVO
    # -----------------------
    elif modo == "🎥 Video en vivo":
        st.markdown("Transmisión en vivo desde la cámara para detección en tiempo real.")

        class VideoProcessor(VideoProcessorBase):
            def __init__(self):
                self.conf = conf

            def recv(self, frame):
                img = frame.to_ndarray(format="bgr24")
                annotated = process_frame(img, model, self.conf)
                return av.VideoFrame.from_ndarray(annotated, format="bgr24")

        webrtc_streamer(
            key="ecovision-live",
            video_processor_factory=VideoProcessor,
            media_stream_constraints={"video": True, "audio": False},
        )

# ------------------------------
# 📊 MÉTRICAS
# ------------------------------

with tab_metrics:
    st.subheader("Métricas del modelo entrenado: LatVision")

    results_path = "runs/LatVision/results.csv"
    args_path = "runs/LatVision/args.yaml"

    if os.path.exists(results_path):
        df = pd.read_csv(results_path)
        st.markdown("### Curvas de precisión y recall")
        st.line_chart(df[["metrics/precision(B)", "metrics/recall(B)"]])
        st.markdown("### Curvas mAP")
        st.line_chart(df[["metrics/mAP50(B)", "metrics/mAP50-95(B)"]])
    else:
        st.warning("No se encontró el archivo de métricas.")

    # Mostrar imágenes de YOLO
    st.markdown("### Resultados visuales del entrenamiento")
    for img in ["results.png", "confusion_matrix.png", "BoxPR_curve.png", "MaskPR_curve.png", "BoxF1_curve.png", "BoxP_curve.png", "BoxR_curve.png", "labels.jpg", "MaskP_curve.png", "MaskR_curve.png", "train_batch0.jpg", "train_batch1.jpg", "train_batch2.jpg", "val_batch0_labels.jpg", "val_batch0_pred.jpg"]:
        path = os.path.join("runs/LatVision", img)
        if os.path.exists(path):
            st.image(path, caption=img, use_container_width=True)

    # Mostrar parámetros del entrenamiento
    if os.path.exists(args_path):
        with open(args_path, "r") as f:
            args = yaml.safe_load(f)
        st.markdown("### Configuración de entrenamiento")
        st.json(args)


# ------------------------------
# 🧾 INFORME
# ------------------------------
with tab_report:
    st.subheader("🧾 Informe del Proyecto")
    st.markdown("## 1. Introducción")
    st.markdown(
        "EcoVision es un sistema basado en visión por computadora **PyTorch** y YOLOv8 para detectar latas y botellas en tiempo real."
    )
    st.markdown("## 2. Tecnologías Utilizadas")
    st.markdown(
        "- **Framework:** PyTorch\n- **Modelo:** YOLOv8 (Ultralytics)\n- **Captura:** OpenCV, Streamlit\n- **Interfaz:** Streamlit"
    )
    st.markdown("## 3. Funcionamiento")
    st.markdown(
        "1. Captura de frames desde la cámara del navegador\n2. Inferencia con modelo PyTorch\n3. Visualización de detecciones en tiempo real"
    )
    st.markdown("## 4. Resultados Esperados")
    st.markdown(
        "- **Precisión objetivo:** >90%\n- **FPS:** 15-20 fps en hardware estándar"
    )
    st.markdown("## 5. Conclusiones")
    st.markdown(
        "EcoVision ofrece una solución rápida y precisa para automatizar la clasificación de reciclables."
    )
