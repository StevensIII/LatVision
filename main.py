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

# Cámara
with tab_camera:
    st.subheader("📹 Cámara en Vivo")
    col1, col2 = st.columns([2, 1])
    video_placeholder = col1.empty()
    start = col2.button("🎥 Iniciar")
    stop = col2.button("⏹️ Detener")
    conf = col2.slider("Umbral confianza", 0.0, 1.0, 0.5, 0.05)

    if 'running' not in st.session_state:
        st.session_state.running = False
    if start:
        st.session_state.running = True
    if stop:
        st.session_state.running = False

    if st.session_state.running:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error("No se pudo acceder a la cámara")
        while st.session_state.running:
            ret, frame = cap.read()
            if not ret:
                st.error("Error leyendo frame")
                break
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            annotated = process_frame(frame_rgb, model, conf)
            video_placeholder.image(annotated, use_column_width=True)
        cap.release()

# Métricas
with tab_metrics:
    st.subheader("📊 Métricas de Entrenamiento")
    st.metric("Precisión", "--")
    st.metric("Recall", "--")
    st.metric("mAP@0.5", "--")

# Informe
with tab_report:
    st.subheader("🧾 Informe del Proyecto")
    st.markdown("## 1. Introducción")
    st.markdown(
        "EcoVision es un sistema basado en vision por computadora **PyTorch** y YOLOv8 para detectar latas y botellas en tiempo real."
    )
    st.markdown("## 2. Tecnologías Utilizadas")
    st.markdown(
        "- **Framework:** PyTorch\n- **Modelo:** YOLOv8 (Ultralytics)\n- **Captura:** OpenCV\n- **Interfaz:** Streamlit"
    )
    st.markdown("## 3. Funcionamiento")
    st.markdown(
        "1. Captura de frames de cámara\n2. Inferencia con modelo PyTorch\n3. Visualización de detecciones en tiempo real"
    )
    st.markdown("## 4. Resultados Esperados")
    st.markdown(
        "- **Precisión objetivo:** >90%\n- **FPS:** 15-20 fps en hardware estándar"
    )
    st.markdown("## 5. Conclusiones")
    st.markdown(
        "EcoVision ofrece una solución rápida y precisa para automatizar la clasificación de reciclables."
    )
