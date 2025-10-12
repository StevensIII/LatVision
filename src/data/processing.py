# processing.py
"""
Módulo para procesamiento de frames con modelo YOLOv8.
"""
import cv2


def process_frame(frame, model, conf: float):
    """Realiza inferencia en un frame y devuelve la imagen anotada.

    Args:
        frame (ndarray): Imagen RGB.
        model: Objeto YOLOv8.
        conf (float): Umbral de confianza.

    Returns:
        ndarray: Frame con bounding boxes.
    """
    if model is None:
        return frame
    results = model(frame, conf=conf)
    annotated = results[0].plot()
    return annotated
