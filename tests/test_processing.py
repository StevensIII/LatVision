import numpy as np
import pytest
from src.data.processing import process_frame

def test_process_frame_no_model():
    """Debe devolver el mismo frame si model es None"""
    dummy_frame = np.zeros((100, 100, 3), dtype=np.uint8)
    result = process_frame(dummy_frame, model=None, conf=0.5)
    assert np.array_equal(result, dummy_frame), "El frame debería ser igual si model es None"

def test_process_frame_with_mock(monkeypatch):
    """Debe retornar el frame anotado usando un modelo mockeado"""
    
    dummy_frame = np.zeros((100, 100, 3), dtype=np.uint8)

    # Crear un mock del objeto modelo
    class MockResult:
        def plot(self):
            # Retornamos un frame modificado para simular anotación
            return dummy_frame + 1

    class MockModel:
        def __call__(self, frame, conf=0.5):
            return [MockResult()]

    mock_model = MockModel()
    result = process_frame(dummy_frame, model=mock_model, conf=0.5)

    # Verificar que el frame fue "anotado" (mock)
    assert np.array_equal(result, dummy_frame + 1), "El frame no fue procesado correctamente"
