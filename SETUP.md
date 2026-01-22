# Guía de Instalación y Uso - Proyecto Steam Games ML

Esta guía te ayudará a configurar y ejecutar el proyecto paso a paso.

---

## 📋 Pre-requisitos

- **Python 3.8 o superior** instalado en tu sistema
- **Git** (opcional, para control de versiones)
- **8 GB RAM** mínimo recomendado
- **Conexión a internet** para descargar el dataset

---

## 🚀 Instalación Paso a Paso

### 1. Configurar el Entorno Virtual

El proyecto ya tiene un entorno virtual creado. Para activarlo:

**En Windows:**
```bash
cd C:\MMLL
venv\Scripts\activate
```

**En macOS/Linux:**
```bash
cd /path/to/MMLL
source venv/bin/activate
```

Verás que tu terminal ahora muestra `(venv)` al inicio de la línea.

---

### 2. Instalar Dependencias

Con el entorno virtual activado, instala todas las librerías necesarias:

```bash
pip install -r requirements.txt
```

Este proceso puede tardar unos minutos. Las dependencias incluyen:
- scikit-learn (modelos de ML)
- pandas y numpy (manipulación de datos)
- matplotlib y seaborn (visualización)
- jupyter (notebooks interactivos)

---

### 3. Descargar el Dataset

El proyecto necesita un dataset de Steam. Descárgalo de una de estas fuentes:

**Opción 1: Kaggle - Steam Store Games**
1. Ve a: https://www.kaggle.com/datasets/nikdavis/steam-store-games
2. Descarga el archivo CSV
3. Renómbralo a `steam_games.csv`
4. Colócalo en: `C:\MMLL\data\raw\steam_games.csv`

**Opción 2: Kaggle - Steam Games Dataset (alternativo)**
1. Ve a: https://www.kaggle.com/datasets/fronkongames/steam-games-dataset
2. Descarga el archivo CSV
3. Renómbralo a `steam_games.csv`
4. Colócalo en: `C:\MMLL\data\raw\steam_games.csv`

**Estructura esperada:**
```
MMLL/
├── data/
│   └── raw/
│       └── steam_games.csv   ← Aquí debe estar el archivo
```

---

## 🎯 Ejecución del Proyecto

Tienes dos formas de ejecutar el proyecto:

### Opción A: Pipeline Completo Automatizado (Recomendado)

Ejecuta todo el pipeline de una vez:

```bash
python run_pipeline.py
```

Este comando ejecutará en orden:
1. Preprocesado de datos
2. Modelos de clasificación
3. Modelos de regresión
4. Análisis no supervisado (PCA y Clustering)

**Opciones adicionales:**

Ejecutar solo ciertos pasos:
```bash
# Solo clasificación
python run_pipeline.py --steps classification

# Solo regresión y clustering
python run_pipeline.py --steps regression unsupervised

# Saltar preprocesado si ya lo hiciste antes
python run_pipeline.py --skip-preprocess
```

---

### Opción B: Ejecución Manual por Pasos

Puedes ejecutar cada módulo individualmente:

#### 1. Preprocesado de Datos
```bash
python src/preprocessing.py
```
Esto genera: `data/processed/steam_games_processed.csv`

#### 2. Modelos de Clasificación
```bash
python src/classification.py
```
Genera modelos y resultados en `models/classification/` y `results/figures/classification/`

#### 3. Modelos de Regresión
```bash
python src/regression.py
```
Genera modelos y resultados en `models/regression/` y `results/figures/regression/`

#### 4. Análisis No Supervisado
```bash
python src/unsupervised.py
```
Genera análisis PCA y clustering en `models/unsupervised/` y `results/figures/unsupervised/`

---

## 📊 Uso de Jupyter Notebooks

Para análisis interactivo y exploración de datos:

### 1. Iniciar Jupyter
```bash
jupyter notebook
```

Esto abrirá tu navegador automáticamente.

### 2. Abrir Notebooks

Navega a la carpeta `notebooks/` y abre:
- `01_exploratory_analysis.ipynb` - Análisis exploratorio de datos

### 3. Ejecutar Celdas

- Ejecuta celda por celda con: `Shift + Enter`
- O ejecuta todas: `Cell → Run All`

---

## 📁 Estructura de Resultados

Después de ejecutar el pipeline, encontrarás:

```
MMLL/
├── data/
│   └── processed/
│       └── steam_games_processed.csv   # Datos preprocesados
├── models/
│   ├── classification/                 # Modelos de clasificación (.joblib)
│   ├── regression/                     # Modelos de regresión (.joblib)
│   └── unsupervised/                   # Modelos PCA y K-Means (.joblib)
├── results/
│   ├── figures/
│   │   ├── classification/             # Gráficos de clasificación
│   │   ├── regression/                 # Gráficos de regresión
│   │   └── unsupervised/               # Gráficos PCA y clustering
│   └── metrics/
│       ├── classification_metrics.csv   # Métricas de clasificación
│       ├── regression_metrics.csv       # Métricas de regresión
│       └── cluster_statistics.csv       # Estadísticas de clusters
```

---

## 🔧 Solución de Problemas Comunes

### Error: "ModuleNotFoundError"
**Causa:** El entorno virtual no está activado o las dependencias no están instaladas.

**Solución:**
```bash
# Activar entorno virtual
venv\Scripts\activate  # Windows
source venv/bin/activate  # macOS/Linux

# Reinstalar dependencias
pip install -r requirements.txt
```

---

### Error: "FileNotFoundError: steam_games.csv"
**Causa:** El dataset no está en la ubicación correcta.

**Solución:**
1. Verifica que el archivo esté en `data/raw/steam_games.csv`
2. Verifica el nombre exacto del archivo
3. Asegúrate de que sea un archivo `.csv`

---

### Error: "Memory Error" o sistema lento
**Causa:** Dataset muy grande para tu RAM.

**Solución:**
1. Reduce el tamaño del dataset (toma una muestra):
```python
# En preprocessing.py, después de cargar el dataset:
df = df.sample(n=10000, random_state=42)  # Solo 10,000 juegos
```

2. Cierra otras aplicaciones para liberar memoria

---

### Jupyter Notebook no se abre
**Causa:** Jupyter no está instalado o el entorno no está activado.

**Solución:**
```bash
# Activar entorno
venv\Scripts\activate

# Reinstalar jupyter
pip install jupyter

# Iniciar de nuevo
jupyter notebook
```

---

## 📚 Siguientes Pasos

1. **Explora los resultados:** Revisa las carpetas `results/figures/` y `results/metrics/`

2. **Analiza las métricas:** Abre los archivos CSV en `results/metrics/` con Excel o un editor de texto

3. **Experimenta con notebooks:** Abre `notebooks/01_exploratory_analysis.ipynb` y ejecuta el análisis exploratorio

4. **Modifica parámetros:** Edita los scripts en `src/` para probar diferentes configuraciones:
   - Cambia el umbral de éxito en `preprocessing.py` (línea 23)
   - Ajusta hiperparámetros de modelos en `classification.py` o `regression.py`
   - Prueba diferentes números de clusters en `unsupervised.py`

5. **Documenta tus hallazgos:** Añade conclusiones en los notebooks

---

## 💡 Comandos Útiles

```bash
# Ver versión de Python
python --version

# Ver paquetes instalados
pip list

# Ver uso de GPU (si tienes)
nvidia-smi

# Desactivar entorno virtual
deactivate

# Ver estructura de carpetas
tree /F  # Windows
tree     # macOS/Linux
```

---

## 📧 Ayuda y Soporte

Si encuentras problemas:

1. Revisa la sección de **Solución de Problemas** arriba
2. Verifica que todas las dependencias estén instaladas: `pip list`
3. Asegúrate de que el dataset esté en la ubicación correcta
4. Revisa los logs de error para identificar el problema específico

---

## 🎓 Buenas Prácticas

1. **Siempre activa el entorno virtual** antes de trabajar
2. **Guarda tus cambios** regularmente
3. **Documenta tus experimentos** en los notebooks
4. **Haz backup** de tus resultados importantes
5. **No modifiques** los datos raw originales (siempre trabaja con copias)

---

¡Listo! Ya tienes todo configurado para trabajar en tu proyecto de Machine Learning. 🚀
