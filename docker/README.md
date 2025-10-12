# 🐳 Docker Setup - EcoVision

> EcoVision: Sistema de inteligencia artificial basado en visión por computadora para la detección y segmentación de latas y botellas, desplegado con Streamlit, usando PyTorch.

---

## 📋 Información General

| **Campo** | **Valor** |
|------------|-----------|
| **🏷️ Repositorio** | `davidjonesja/ecovision` |
| **Tamaño** | 3.89GB |
| **Python** | 3.11.13 |
| **OpenCV** | 4.11.0 |
| **Última actualización** | 5 de Octubre, 2025 |

---

## 🚀 Inicio Rápido

### Opción 1: Usar imagen preconstruida (Recomendado)

```bash
# 1. Descargar imagen desde Docker Hub
docker pull davidjonesja/ecovision:latest

# 2. Ejecutar la app (en Linux o macOS con XQuartz/xming configurado)
docker run -it --rm \
  -e DISPLAY=$DISPLAY \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  --name ecovision \
  davidjonesja/ecovision:latest


# 1. Clonar repositorio
git clone https://github.com/davidarroyave/ecovision.git
cd ecovision

# 2. Construir la imagen Docker
docker build -t ecovision:latest .

# 3. Ejecutar la app
docker run -it --rm \
  -e DISPLAY=$DISPLAY \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v $(pwd):/app \
  --name ecovision \
  ecovision:latest

# 1. Clonar repositorio
git clone https://github.com/davidarroyave/ecovision.git
cd ecovision

# 2. Construir la imagen Docker
docker build -t ecovision:latest .

# 3. Ejecutar la app
docker run -it --rm \
  -e DISPLAY=$DISPLAY \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v $(pwd):/app \
  --name ecovision \
  ecovision:latest

# Permitir acceso al servidor X (solo una vez)
xhost +local:docker

# Ejecutar con GUI
docker run -it --rm \
  -e DISPLAY=$DISPLAY \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v $(pwd):/app \
  --name ecovision \
  ecovision:latest

# Instalar XQuartz (si no tienes)
brew install --cask xquartz

# Permitir conexiones XQuartz
xhost +localhost

# Ejecutar contenedor
docker run -it --rm \
  -e DISPLAY=host.docker.internal:0 \
  -v $(pwd):/app \
  --name ecovision \
  ecovision:latest

# Detecta IP de Windows
WINDOWS_IP=$(ip route show | grep -i default | awk '{ print $3 }')

# Ejecuta el contenedor con display configurado
docker run --rm \
  -e DISPLAY="$WINDOWS_IP:0" \
  -v $(pwd):/app \
  --net=host \
  ecovision:latest

docker run -it --rm \
  -e DISPLAY=$DISPLAY \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v $(pwd):/app \
  --name ecovision \
  davidjonesja/ecovision:latest

docker run --rm ecovision:latest python -c "print('Sistema ECOVISION funciona')"

📝 **Notas importantes**
Asegúrate de que tu modelo .pt esté en el directorio correcto y accesible desde la carpeta del proyecto.
La imagen Docker ya trae todo lo necesario para ejecutar el sistema en entorno Linux/macOS/Windows con X11 para la interfaz gráfica.

**No incluye TensorFlow, solo uso de PyTorch y OpenCV.**
🔍 Soporte y Contacto
Para cualquier duda, abrir issues en el repositorio o contactar a los autores en https://github.com/davidarroyave/ecovision.

📖 Funcionalidades

📁 Carga de imágenes y video

🔍 Detección en tiempo real de latas y botellas

🎨 Visualización con bounding boxes

📊 Métricas de precisión, recall y mAP

💾 Exportación de resultados en outputs/

📞 Soporte
Docker Hub: https://hub.docker.com/r/davidjonesja/ecovision

GitHub: https://github.com/davidarroyave/ecovision

Generado el 5 de Octubre, 2025

¡Gracias por usar EcoVision con Docker!

text
