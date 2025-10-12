# Makefile for EcoVision Project
# Sistema de detección de latas y botellas con TensorFlow y Streamlit
# Autores: Jose Luis Martinez Diaz, Juan David Arroyave Ramirez, Neiberth Aponte Aristizabal, Stevens Ricardo Bohorquez Ruiz
# Fecha: 2025-10

MODELO = models/yolov8s-seg.pt
MODEL = src/data/models/
DATA = datasets/20251007v1/data.yaml
EPOCHS = 1
IMGSZ = 640
RUN_NAME = EcoVision_Prueba_Register
APP    = main.py
REQ    = requirements.txt
IMAGE  = docker-ecovision
PORT   = 8501
MLFLOW_PORT = 5000
# Para el registro del modelo en MLflow
RUN_ID = 6e544580f9744497a274832cb3af07e1 
MODEL_NAME = EcoVisionModel10102025v1
ARTIFACT_PATH := weights_model

all: train

# Entrenar modelo
train:
	python src/data/train.py --model $(MODELO) --data $(DATA) --epochs $(EPOCHS) --imgsz $(IMGSZ) --run_name "$(RUN_NAME)" --project ./runs

# Abrir MLflow UI
mlflow:
	mlflow ui --backend-store-uri file:./mlruns --port $(MLFLOW_PORT)
	
# Limpiar artefactos de entrenamiento
cleanml:
	if exist mlruns rmdir /s /q mlruns
	if exist loss_plot.png del /q loss_plot.png
	if exist runs rmdir /s /q runs

#Streamlit
run:
	streamlit run main.py

#Docker
.PHONY: all train mlflow cleanml run docker-build docker docker-status docker-stop clean test-modelo test-proceso

docker-build:
	docker build -t $(IMAGE):latest -f docker/Dockerfile .
	echo "Imagen $(IMAGE) construida"

docker:
	docker run -d -p 8501:8501 $(IMAGE):latest

docker-status:
	docker ps

docker-stop:
	docker stop 6b82d5c6b3b5
	
clean:
	rm -rf __pycache__ .pytest_cache

# Ejecutar solo tests de carga del modelo
test-modelo:
	set PYTHONPATH=.&& uv run pytest -v tests/test_load_model.py

test-proceso:
	set PYTHONPATH=.&& uv run pytest -v tests/test_processing.py
