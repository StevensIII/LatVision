# EcoVision: 
## Sistema de inteligencia artificial basado en visión por computadora para la detección y segmentación de latas y botellas

[![Python Version](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/pytorch%2B-orange.svg)](https://pytorch.org)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache2.0-yellow.svg)](https://www.apache.org/licenses/LICENSE-2.0)
[![Docker](https://img.shields.io/badge/docker-ready-blue.svg)](https://docker.com)


##  Autores
**Jose Luis Martinez Diaz** Codigo-UAO: ***2247574***

**Juan David Arroyave Ramirez** Codigo-UAO: ***2250424***

**Neiberth Aponte Aristizabal** Codigo-UAO: ***2251022*** 

**Stevens Ricardo Bohorquez Ruiz** Codigo-UAO: ***2250760***

##  Descripción del Proyecto
La gestión de residuos sólidos representa un desafío crítico en Colombia, donde la acumulación de latas y botellas plásticas contamina ecosistemas acuáticos y terrestres y eleva costos operativos de recolección. Este trabajo propone un sistema de inteligencia artificial basado en visión por computadora y Deep Learning para la detección y segmentación de latas y botellas, entrenado con YOLOv8 y desplegado
mediante una interfaz Streamlit. Los objetivos incluyen la recolección y anotación de un dataset diverso en Roboflow, el entrenamiento y optimización del modelo con técnicas de Data Augmentation y ajuste de hiperparámetros, la validación en entornos controlados usando métricas de precisión y graficas de perdida. El alcance comprende el desarrollo del modelo, la aplicación web, la documentación técnica y
pruebas de campo con videos de Playas, orillas de ríos, zonas urbanas, o CCTV libres de derechos de autor que estén disponibles para uso académico sin fines comerciales o en datasets públicos o los generados por
EcoVision, excluyendo la recolección manual de datos en tiempo real, la provisión de infraestructura de hardware, mantenimiento o soporte continuo post-entrega y desarrollo de modelos adicionales distintos a
YOLOv8. Este enfoque aporta datos objetivos para optimizar rutas de recolección, apoyar iniciativas de economía circular y facilitar la toma de decisiones en autoridades ambientales y redes de recicladores

### Objetivos del Proyecto

**Objetivo General**: Desarrollar un sistema de inteligencia artificial basado en visión por computadora para la detección y segmentación de latas y botellas, con el fin de apoyar la identificación automatizada
de contaminantes y fortalecer estrategias de gestión ambiental.

 **Objetivo Específico**: Entrenar y evaluar un modelo de visión por computadora para la detección y segmentación de objetos utilizando YOLOv8.

 **Objetivo Técnico**: Desplegar el sistema de inteligencia artificial basado en visión por computadora mediante la plataforma Streamlit, desarrollando una interfaz web interactiva que permita la carga de imágenes, videos, activación de la cámara para la utilización del modelo en tiempo real.


### Metodología

- **Modelo**: YOLOv8 para detección de objetos en tiempo real que utiliza una arquitectura de redes neuronales convolucionales (CNN).
- **Entrada**: Imágenes o videos de latas y botellas.
- **Preprocesamiento**: Entrenamiento de imagenes relacionadas, proceso de normalización, y fine-tuning.
- **Visualización**: Camara en tiempo real o inserción manual de imagenes o videos relacionados para explicabilidad del modelo
- **Interfaz**: Graphic User Interface (GUI) desarrollada en Streamlit para facilidad de uso.

---

## Estructura del Proyecto

```
ECOVISION/
│
├── 📁 venv/
├── 📁 data/
|   └── 📁 external 
|   └── 📁 processed
|   └── 📂 raw                                      
├── 📁 docs/
|   └── 📖 README.md
├── 📁 notebooks/
├── 📁 reports/                                        
├── 📁 src/                                           # Código fuente principal
|   └── 📁 data 
|         ├── ▶PENDIENTE.py                           # codigo original, alto acople y sin cohesion
|         ├── ▶PENDIENTE.py                           #
|         ├── ▶integrator.py                          # Módulo integrador del pipeline
|         ├── ▶load_model.py                          # Carga del modelo Ecovision-YOLOv8
|         ├── ▶preprocess_img.py                      # Preprocesamiento de imágenes
|         ├── ▶read_img.py                            # Lectura de imágenes JPG/PNG
|   └── 📁 features
|   └── 📂 models                                     # Modelo .H5 .PKL                   
|   └── 📂 visualizations/
│         ├── PENDIENTE.png                           # Visualizacion del modelo desde Netron.app
│         ├── PENDIENTE.png                           # Visualizacion del diagrama flujo de datos
│         ├── PENDIENTE.png                           # Visualizacion de la app funcionando
│         ├── Model_Summary.png                       # Visualizacion del modelo desde tf.keras
├── 📁 tests/
│         ├── 📂JPG/
│               ├── Imagenes para testeo .jpg 
│         ├── Archivos .py utilizados para las pruebas unitarias
├── 🚫 .gitignore                                     # Archivos ignorados por Git
├── 🔢 .python-version                                # Versión de Python especificada
├── 🐳 Dockerfile                                     # Configuración para la imagen contenedora
├── ⚖️ LICENSE                                        # Licencia Apache 2.0
├── 🤖 main.py                                        # Archivo principal Streamlit
├── 📋 pyproject.toml                                 # Configuración del proyecto UV
├── 📄 requirements.txt                               # Dependencias del proyecto
├── 🔒 uv.lock                                        # Lock file de dependencias UV
├── 🚀 Makefile

```

---

## Requisitos

### Versión de Python
- **Python**: 3.11 para mejor compatibilidad con TorchVision y YOLOv8

### 💻 Requisitos del Sistema
- **RAM**: Mínimo 4GB (recomendado 8GB o superior)
- **Espacio en disco**: 5GB libres como minimo

---

## Instalación del Repositorio

### Método 1: Instalación con UV (Recomendado)

#### 1. Clonar el repositorio
```bash
git clone https://github.com/davidarroyave/EcoVision
cd UAO-Neumonia
```

#### 2. Instalar UV (si no lo tienes)
```bash
# Linux/macOS
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows (PowerShell)
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

#### 3. Crear entorno e instalar dependencias
```bash
# Crear entorno virtual automáticamente e instalar dependencias
uv sync

# Activar entorno virtual
source .venv/bin/activate  # Linux/macOS
# o
.venv\Scripts\activate     # Windows
.venv\bin\activate         # Windows

```
#### 4. Descargar el modelo (si no está incluido)
```bash
# El modelo .PT debe estar en la carpeta models/
# Si no está presente, contactar al equipo de desarrollo
```

#### 5. Ejecutar la aplicación
```bash
make streamlit #Opción 1 (Recomendada)
streamlit run main.py #Opción 2
#Si no se abre automaticamente, ingresar al link que arroja la consola: http://localhost:8501
```
### Método 2: Instalación con Docker [![Docker](https://img.shields.io/badge/docker-ready-blue.svg)](docker/README.md)


## Método 3: Instalación Manual con pip

#### 1. Clonar y preparar entorno
```bash
git clone https://github.com/davidarroyave/ecovision
cd ecovision

# Crear entorno virtual
python -m venv venv
source venv/bin/activate  # Linux/macOS
# o
venv\Scripts\activate     # Windows
.venv\bin\activate        # Windows  
```

#### 2. Instalar dependencias
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

#### 3. Verificar instalación
```bash
python -c "import torch; print(f'Torch: {torch.__version__}')"
```

### Verificación de Instalación

```bash
# Ejecutar tests básicos
python -m pytest tests/ -v

# Verificar modelo
python -c "
from src.load_model import model_fun
model = model_fun()
print('Modelo cargado correctamente' if model else 'Error al cargar modelo')
"

# Probar interfaz (modo headless para servidores)
python main.py --test
```
## 🔬 Tipos de residuos de latas y botellas Detectados

El modelo **YOLOv8** está entrenado para clasificar las siguientes condiciones:

| Clase | Código | Descripción | Precisión |
|-------|--------|-------------|-----------|
| **🥫 Lata** | `###` | Latas detectadas por el modelo | %%%% |
| **🍾 Botella** | `###` | Botellas detectadas por el modelo | %%%% |

**Precisión general del modelo**:%%%%

---

## Descripción Detallada de Módulos

### `main.py` - Interfaz Streamlit Principal
**Función**: Punto de entrada de la aplicación con interfaz Streamlit.

**Características**:


**Widgets principales**:


### `src/data/integrator.py` - Módulo Integrador Principal
**Función**: Orquesta todo el pipeline de predicción.

**Flujo de trabajo**:


**Funciones clave**:


### `src/data/camera.py` - Activación de la cámara en tiempo real
**Función**: Activa la cámara en tiempo real desde streamlit en tu dispositivo, ya sea computador o celular.

**Capacidades**:


**Funciones**:



### 🤖 `src/data/load_model.py` - Carga del Modelo
**Función**: Gestión y carga del modelo de red neuronal.

**Características**:
- 📂 Carga de `models/.pt`
- ✅ Verificación de existencia de archivo
- 🛡️ Manejo de errores de compatibilidad
- 🔍 Validación de arquitectura (YOLOv8)

---

## 🚀 Ejecuciones
- 🖥 Abrir MLflow UI: Terminal en `C:\EcoVision`, activar `.\venv\Scripts\activate`, ejecutar `make mlflow`, abrir [http://localhost:5000](http://localhost:5000).  
- 🧪 Test carga del modelo: `make test-modelo` → verifica rutas inexistentes y carga simulada con mocks.  
- 🧪 Test proceso del modelo: `make test-proceso` → valida `process_frame` sin modelo y con mocks.  
- 📌 YOLOv8: versión estable y compatible con Python/Roboflow; versiones recientes aún en adopción temprana.  
- 📊 Tracking MLflow: configurar `EPOCHS = 5` en `Makefile`, ejecutar `make train`; resultados en `runs/segment/<run_name>`.  
- 💾 DVC: instalar `pip install dvc dvc-azure`, `dvc init`, configurar remote Azure (`dvc remote add/modify`), agregar/subir archivos (`dvc add --to-remote` o `dvc push`), otros comandos: `dvc remote list`, `dvc remote default`, `dvc pull`.


---

### 🏗️ Arquitectura del Modelo YOLOv8

```bash

               from  n    params  module                                       arguments                     
  0                  -1  1       464  ultralytics.nn.modules.conv.Conv             [3, 16, 3, 2]                 
  1                  -1  1      4672  ultralytics.nn.modules.conv.Conv             [16, 32, 3, 2]                
  2                  -1  1      7360  ultralytics.nn.modules.block.C2f             [32, 32, 1, True]             
  3                  -1  1     18560  ultralytics.nn.modules.conv.Conv             [32, 64, 3, 2]                
  4                  -1  2     49664  ultralytics.nn.modules.block.C2f             [64, 64, 2, True]             
  5                  -1  1     73984  ultralytics.nn.modules.conv.Conv             [64, 128, 3, 2]               
  6                  -1  2    197632  ultralytics.nn.modules.block.C2f             [128, 128, 2, True]           
  7                  -1  1    295424  ultralytics.nn.modules.conv.Conv             [128, 256, 3, 2]              
  8                  -1  1    460288  ultralytics.nn.modules.block.C2f             [256, 256, 1, True]           
  9                  -1  1    164608  ultralytics.nn.modules.block.SPPF            [256, 256, 5]                 
 10                  -1  1         0  torch.nn.modules.upsampling.Upsample         [None, 2, 'nearest']          
 11             [-1, 6]  1         0  ultralytics.nn.modules.conv.Concat           [1]                           
 12                  -1  1    148224  ultralytics.nn.modules.block.C2f             [384, 128, 1]                 
 13                  -1  1         0  torch.nn.modules.upsampling.Upsample         [None, 2, 'nearest']          
 14             [-1, 4]  1         0  ultralytics.nn.modules.conv.Concat           [1]                           
 15                  -1  1     37248  ultralytics.nn.modules.block.C2f             [192, 64, 1]                  
 16                  -1  1     36992  ultralytics.nn.modules.conv.Conv             [64, 64, 3, 2]                
 17            [-1, 12]  1         0  ultralytics.nn.modules.conv.Concat           [1]                           
 18                  -1  1    123648  ultralytics.nn.modules.block.C2f             [192, 128, 1]                 
 19                  -1  1    147712  ultralytics.nn.modules.conv.Conv             [128, 128, 3, 2]              
 20             [-1, 9]  1         0  ultralytics.nn.modules.conv.Concat           [1]                           
 21                  -1  1    493056  ultralytics.nn.modules.block.C2f             [384, 256, 1]                 
 22        [15, 18, 21]  1    897664  ultralytics.nn.modules.head.Detect           [80, [64, 128, 256]]          
Model summary: 129 layers, 3,157,200 parameters, 3,157,184 gradients, 8.9 GFLOPs

```
---

## ⚖️ Licencia

Este proyecto está licenciado bajo la **Licencia Apache 2.0** - ver el archivo [LICENSE](LICENSE) para detalles completos.

# 📚 Referencias y Bibliografía

[1] Banco Mundial, What a Waste
2.0: A Global Snapshot of Solid Waste Management to 2050, Accessed: Sept. 27, 2025, sep. de
2018. dirección: https://www.bancomundial.org/es/news/immersive - story / 2018 / 09 /20/what-a-waste-an-updatedlook- into- the- future- ofsolid-waste-management.


[2] J. Planelles, “Cada año se vierten más de 52 millones de toneladas de plásticos al medioambiente según un estudio de Nature”, El País, sep. de 2024, Accessed: Sept. 27, 2025. dirección:
https://elpais.com/climay-medio-ambiente/2024-09-04 / cada - ano - se - vierten -mas - de - 52 - millones - de -toneladas-de-plasticos-almedioambiente-segun-un-estudiode-nature.html.


[3] Greenpeace Colombia, El problema de los residuos, Accessed: Sept. 27, 2025, 2025. dirección:
https://www.greenpeace.org/colombia / el - problema - de -los-residuos/.


[4] “Contaminación de residuos só-
lidos y sus efectos en la salud y medio ambiente”, Revista INVECOM, vol. 4, n.o 2, dic. de
2024, Accessed: Sept. 27, 2025.dirección: http://revistainvecom.14org/index.php/invecom/article/
view/3557.


[5] Condensa, Aluminio vs. Plástico: La sostenibilidad en envases, Accessed: Sept. 27, 2025,
abr. de 2024. dirección: https:/ / condensa . com / 2024 / 04 /17 / aluminio - vs - plastico -
la-sostenibilidad-en-envases/.


[6] ARPAL, Memoria de Actividades 2024, Accessed: Sept. 27, 2025,
jun. de 2025. dirección: https://aluminio.org/arpal-presentasu-memoria-de-actividades-
2024/.


[7] “Sistema de procesamiento de imágenes para la detección de residuos sólidos”, AIBI Revista
de Investigación, vol. 13, n.o 1, abr. de 2025, Accessed: Sept. 27, 2025. dirección: https://revistas.
udes.edu.co/aibi/article/
view/4426.


[8] Reciclaje y Gestión, Tendencias
y tecnologías en gestión de residuos para 2024, Accessed: Sept.
27, 2025, mayo de 2024. dirección: https://reciclajeygestion.es/2024/05/27/tendenciasy-tecnologias-en-gestionde-residuos-para-2024-innovacionesy - desarrollos - recientes -
en-el-sector/.


[9] Recycleye, La IA y el reconocimiento de residuos: por qué funciona, Accessed: Sept. 27, 2025,
jun. de 2024. dirección: https://recycleye.com/es/la-iay - el - reconocimiento - de -residuos/.


[10] “Uso de la Visión Artificial para
la Clasificación de Residuos Orgánicos e Inorgánicos”, Nexos Científicos, Accessed: Sept. 27, 2025.
dirección: https://nexoscientificos.vidanueva.edu.ec/index.php/ojs / article / download / 61 /
268/501.


[11] RCB Trace, Perspectivas futuras: avances tecnológicos en la
gestión de residuos, Accessed: Sept.27, 2025, abr. de 2024. dirección: https://www.rcbtrace.com/perspectivas- futurasavances - tecnologicos - en -la-gestion-de-residuos/.


[12] Recyclever, Reverse Vending: Beneficios y Futuro Sostenible, Accessed: Sept. 27, 2025, feb. de
2025. dirección: https://www.recyclever.com/es/blog/article-10/reverse-vending-beneficiosy-futuro-sostenible-310.


[13] Tiempo, Reverse vending: el fenómeno de las máquinas que te
pagan por reciclar, Accessed: Sept.27, 2025, dic. de 2024. dirección:https : / / www . tiempo . com /
noticias/actualidad/reversevending - el - fenomeno - de -las-maquinas-que-te-paganpor-reciclar.html.


[14] Ultralytics, Segmentación de instancias con seguimiento de objetos, Accessed: Sept. 27, 2025,
mayo de 2025. dirección: https://docs.ultralytics.com/es/15guides/instance-segmentationand-tracking/.


[15] Ultralytics, Segmentación de Instancias, Accessed: Sept. 27, 2025,
mar. de 2025. dirección: https://docs.ultralytics.com/es/tasks/segment/.


[16] Detección y Cuantificación de Erosión Fluvial con Visión por Computadora, Accessed: Sept. 27, 2025,
dic. de 2021. arXiv: 2507.11301. dirección: https://arxiv.org/html/2507.11301v1.


[17] Ultralytics, Un futuro más verde mediante Vision AI y Ultralytics YOLO, Accessed: Sept. 27,
2025, sep. de 2025. dirección: https: //www.ultralytics.com/es/blog/greener-future-throughvision-ai-and-ultralyticsyolo


**Última Actualización**: Octubre 8, 2025
**Estado del Proyecto**: Producción en desarrollo 🟡  