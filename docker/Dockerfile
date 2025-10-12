# Dockerfile for EcoVision
# Sistema de detección de latas y botellas con PyTorch y Streamlit

FROM python:3.11-slim

# Información del mantenedor
LABEL maintainer="Juan David Arroyave Ramirez <juan_dav.arroyave@uao.edu.co>"
LABEL description="EcoVision - Sistema de detección de latas y botellas con IA"
LABEL version="1.0"

# Variables de entorno
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0

# Instalar dependencias del sistema
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
    && rm -rf /var/lib/apt/lists/*

# Crear directorio de trabajo
WORKDIR /app

# Instalar uv (gestor de paquetes ultrarrápido)
RUN pip install --no-cache-dir uv

# Copiar archivos de configuración primero (para cache de Docker)
COPY pyproject.toml .
COPY uv.lock* ./

# Instalar dependencias con uv (mucho más rápido que pip)
RUN uv pip install --system --no-cache -r pyproject.toml

# Copiar código fuente
COPY main.py .
COPY src/ ./src/

# Copiar assets
COPY ecovision_logo.png .

# Crear directorios necesarios
RUN mkdir -p outputs models logs temp

# Exponer puerto
EXPOSE 8501

# Comando de salud para verificar que la aplicación esté funcionando
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Comando por defecto
CMD ["streamlit", "run", "main.py", "--server.port=8501", "--server.address=0.0.0.0"]