import os
import pytest
from src.data.load_model import load_pytorch_model

def test_model_file_not_found(monkeypatch):
    """Debe devolver None si el archivo del modelo no existe"""
    # Ruta de modelo inexistente
    fake_path = "src/models/no_existe.pt"

    # Ejecutar
    model = load_pytorch_model(fake_path)

    # Verificar
    assert model is None, "El modelo debería ser None cuando el archivo no existe"

def test_model_file_exists(monkeypatch, tmp_path):
    """Debe intentar cargar el modelo si el archivo existe (mockeado)"""
    # Crear un archivo vacío temporal que simule un modelo
    fake_model_path = tmp_path / "fake_model.pt"
    fake_model_path.write_text("fake content")

    # Mockear la función YOLO para no cargar nada real
    monkeypatch.setattr("src.data.load_model.YOLO", lambda path: f"Modelo cargado: {path}")

    # Ejecutar
    model = load_pytorch_model(str(fake_model_path))

    # Verificar
    assert model == f"Modelo cargado: {fake_model_path}", "El modelo no se cargó correctamente"
