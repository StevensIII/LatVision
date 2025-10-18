# Dockerfile for LatVision
# Sistema de detección de latas y botellas con PyTorch y Streamlit

FROM python:3.11-slim

# Información del contenedor
LABEL maintainer="Stevens Bohorquez Ruiz <stevens.bohorquez@uao.edu.co>"
LABEL description="LatVision - Sistema de detección de latas y botellas con IA"
LABEL version="1.0"

# Variables de entorno
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0

# Instalar dependencias del sistema necesarias para OpenCV, Torch y Streamlit-WebRTC
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    libgthread-2.0-0 \
    curl \
    libfontconfig1 \
    libavformat-dev \
    libavdevice-dev \
    libavfilter-dev \
    libavcodec-dev \
    libswscale-dev \
    libavutil-dev \
    libopus-dev \
    libvpx-dev \
    && rm -rf /var/lib/apt/lists/*

# Crear directorio de trabajo
WORKDIR /app

# Instalar uv (gestor ultrarrápido)
RUN pip install --no-cache-dir uv

# Copiar archivos de configuración (para aprovechar cache en Docker)
COPY pyproject.toml ./
COPY uv.lock* ./

# Instalar dependencias (usa el lock si existe, o sincroniza si no)
RUN if [ -f uv.lock ]; then uv sync --frozen --no-cache; else uv sync --no-cache; fi

# Copiar carpeta de métricas
COPY runs/ ./runs/

# Copiar código fuente
COPY main.py .
COPY src/ ./src/
COPY ecovision_logo.png .

# Crear directorios necesarios
RUN mkdir -p outputs models logs temp

# Exponer puerto de Streamlit
EXPOSE 8501

# Healthcheck para verificar que la app responda
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Ejecutar la aplicación usando el Python del entorno virtual
CMD ["/app/.venv/bin/python", "-m", "streamlit", "run", "main.py", "--server.port=8501", "--server.address=0.0.0.0"]

