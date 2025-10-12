# load_model.py
"""
Módulo para carga de modelo PyTorch YOLOv8
"""
import os
from ultralytics import YOLO
import streamlit as st

MODEL_PATH = "src/models/best_latas.pt" #Reemplazar por modelo final.

@st.cache_resource
def load_pytorch_model(path: str = MODEL_PATH):
    """Carga el modelo YOLOv8 entrenado en PyTorch."""
    if not os.path.isfile(path):
        st.warning(f"🔍 Modelo no encontrado en '{path}'.")
        return None
    try:
        model = YOLO(path)
        return model
    except Exception as e:
        st.error(f"❌ Error cargando el modelo: {e}")
        return None
