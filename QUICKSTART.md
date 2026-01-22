# QUICKSTART - Steam Games ML Project

## Inicio Rápido en 5 Minutos

### 1. Activar Entorno Virtual

**Windows:**
```bash
cd C:\MMLL
venv\Scripts\activate
```

**macOS/Linux:**
```bash
cd /path/to/MMLL
source venv/bin/activate
```

### 2. Descargar Dataset

Descarga el dataset de Steam y colócalo en:
```
C:\MMLL\data\raw\steam_games.csv
```

**Fuentes recomendadas:**
- https://www.kaggle.com/datasets/nikdavis/steam-store-games
- https://www.kaggle.com/datasets/fronkongames/steam-games-dataset

### 3. Ejecutar Pipeline Completo

```bash
python run_pipeline.py
```

¡Eso es todo! El pipeline ejecutará automáticamente:
- Preprocesamiento
- Modelos de clasificación
- Modelos de regresión
- Análisis no supervisado (PCA y Clustering)

---

## Comandos Útiles

### Ejecutar pasos individuales:

```bash
# Solo preprocesado
python src/preprocessing.py

# Solo clasificación
python src/classification.py

# Solo regresión
python src/regression.py

# Solo análisis no supervisado
python src/unsupervised.py
```

### Usar Jupyter Notebooks:

```bash
jupyter notebook
```

Luego abre: `notebooks/01_exploratory_analysis.ipynb`

---

## Estructura de Archivos Principales

```
MMLL/
├── data/
│   ├── raw/                 # Coloca aquí steam_games.csv
│   └── processed/           # Datos procesados (generados automáticamente)
├── src/
│   ├── preprocessing.py     # Limpieza y preprocesado
│   ├── classification.py    # Modelos de clasificación
│   ├── regression.py        # Modelos de regresión
│   └── unsupervised.py      # PCA y Clustering
├── notebooks/               # Jupyter notebooks para análisis
├── results/                 # Resultados y visualizaciones
├── models/                  # Modelos entrenados guardados
├── run_pipeline.py          # Script principal
├── README.md                # Documentación completa
└── SETUP.md                 # Guía de instalación detallada
```

---

## Resultados

Después de ejecutar el pipeline, encontrarás:

**Visualizaciones:**
- `results/figures/classification/` - Matrices de confusión, curvas ROC
- `results/figures/regression/` - Gráficos de predicción, residuos
- `results/figures/unsupervised/` - PCA, clustering

**Métricas:**
- `results/metrics/classification_metrics.csv`
- `results/metrics/regression_metrics.csv`
- `results/metrics/cluster_statistics.csv`

**Modelos:**
- `models/classification/` - Modelos .joblib de clasificación
- `models/regression/` - Modelos .joblib de regresión
- `models/unsupervised/` - Modelos .joblib de PCA y K-Means

---

## Modelos Implementados

### Clasificación (¿Será exitoso el juego?)
- Regresión Logística
- Árbol de Decisión
- Random Forest
- Gradient Boosting
- SVM

### Regresión (¿Cuántos jugadores tendrá?)
- Regresión Lineal
- Random Forest Regressor
- Gradient Boosting Regressor
- SVR

### No Supervisado
- PCA (reducción de dimensionalidad)
- K-Means Clustering (segmentación)

---

## Métricas de Evaluación

**Clasificación:**
- Accuracy, Precision, Recall, F1-Score
- ROC-AUC
- Matriz de confusión

**Regresión:**
- MAE (Mean Absolute Error)
- RMSE (Root Mean Squared Error)
- R² Score

**Clustering:**
- Silhouette Score
- Varianza explicada (PCA)

---

## Problemas Comunes

**Error: "FileNotFoundError: steam_games.csv"**
→ Coloca el dataset en `data/raw/steam_games.csv`

**Error: "ModuleNotFoundError"**
→ Activa el entorno virtual: `venv\Scripts\activate`

**Jupyter no se abre**
→ Verifica instalación: `pip install jupyter`

---

## Personalización Rápida

### Cambiar umbral de éxito:
Edita `src/preprocessing.py` línea 23:
```python
success_threshold = 100000  # Cambia este valor
```

### Cambiar número de clusters:
Edita `src/unsupervised.py` línea 560:
```python
n_clusters = 4  # Cambia este valor
```

### Modificar hiperparámetros:
Edita los parámetros en:
- `src/classification.py` (líneas 40-70)
- `src/regression.py` (líneas 40-65)

---

## Siguiente Paso: Exploración

Una vez ejecutado el pipeline, abre los notebooks:

```bash
jupyter notebook
```

Navega a `notebooks/01_exploratory_analysis.ipynb` y explora los datos.

---

Para más detalles, consulta:
- [README.md](README.md) - Documentación completa
- [SETUP.md](SETUP.md) - Guía detallada de instalación
