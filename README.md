# Hands-On SupCon

¡Bienvenido a **Hands-On SupCon**! 👋

Un proyecto práctico diseñado para explorar y experimentar con arquitecturas de Deep Learning, Aprendizaje de Similitud (Similarity Learning) y **Supervised Contrastive Learning (SupCon)** utilizando PyTorch.

## 🚀 Sobre el Proyecto

Este repositorio contiene implementaciones modulares y cuadernos interactivos orientados a desmitificar cómo las redes neuronales extraen características y construyen espacios latentes. A través del código, abordamos desde arquitecturas convolucionales básicas hasta modelos codificadores avanzados entrenados con funciones de pérdida contrastivas basadas en márgenes. 

Además, se hace un fuerte énfasis en la **visualización**, permitiéndote observar representaciones del espacio latente en 2D y 3D, y comprender el comportamiento de los modelos de forma intuitiva.

## 📂 Estructura del Repositorio

El proyecto está dividido principalmente en dos áreas: código fuente reutilizable y cuadernos experimentales.

### `src/` (Código Fuente)
- **`datasets.py`**: Clases para la manipulación y carga de datos. Incluye `ContrastiveLearningDataset` para generar los pares de imágenes positivos y negativos necesarios en el entrenamiento contrastivo estilo Siamese.
- **`losses.py`**: Implementación de funciones de pérdida, como `ContrastiveLoss` basada en márgenes (margin-based).
- **`models.py`**: Arquitecturas PyTorch modulares que incluyen desde clasificadores estándar (`MiniConvNetClassifier`) hasta codificadores especializados para extraer *embeddings* (`MiniConvNetEncoder`) y evaluadores lineales (`MiniConvNetLinearProbe`).
- **`plot_data.py`**: Funciones de visualización avanzadas utilizando Matplotlib y Plotly. Permite inspeccionar mapas de características de convoluciones, lotes de datos estilo Siamese, y proyecciones interactivas 2D/3D del espacio latente.

### `notebooks/` (Experimentos Interactivos)
- 📓 `CNN_Architecture.ipynb`: Exploración y entrenamiento de una arquitectura CNN clásica.
- 📓 `CNN_Architecture_Overtrained.ipynb`: Análisis empírico del sobreajuste (overfitting) en redes convolucionales.
- 📓 `Embedding_Architectures_&_Similarity_Learning.ipynb`: Estudio sobre cómo diseñar redes para obtener *embeddings* útiles y evaluar la similitud entre ellos.
- 📓 `SupCon.ipynb`: El núcleo del proyecto, donde se aplica y analiza el Aprendizaje Contrastivo Supervisado en su totalidad.

## 🛠️ Tecnologías y Dependencias

El ecosistema principal del proyecto está basado en:
- **PyTorch & Torchvision**: Construcción, entrenamiento y evaluación de modelos de Deep Learning.
- **UMAP (`umap-learn`)**: Reducción de dimensionalidad rápida y efectiva para analizar *embeddings*.
- **Plotly, Matplotlib & Seaborn**: Creación de gráficos interactivos y estáticos de alta calidad.
- **Jupyter**: Entorno para la ejecución de los experimentos.

El proyecto gestiona sus dependencias de forma moderna a través de `pyproject.toml` y `uv`.

## ⚙️ Instalación y Uso

1. **Clona el repositorio**:
   ```bash
   git clone https://github.com/unlikeghost/hands-on-SupCon.git
   cd hands-on-SupCon
   ```

2. **Configura el entorno virtual e instala las dependencias**:
   El proyecto incluye un `uv.lock`, por lo que se recomienda encarecidamente utilizar [uv](https://github.com/astral-sh/uv) para una instalación extremadamente rápida:
   ```bash
   # Crear entorno e instalar todo sincronizado con uv.lock
   uv sync
   
   # Activar el entorno virtual (en Linux/macOS)
   source .venv/bin/activate
   ```
   *(Alternativamente, puedes usar pip estándar: `pip install -e .`)*

3. **Inicia Jupyter y explora**:
   ```bash
   jupyter notebook
   ```
   Dirígete a la carpeta `notebooks/` y abre cualquier cuaderno para comenzar a interactuar con el código.
