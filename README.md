# Predicción del Éxito de Videojuegos en Steam con Machine Learning

## 📋 Descripción del Proyecto

Este proyecto aplica técnicas clásicas de **Machine Learning supervisado y no supervisado** para analizar videojuegos de la plataforma Steam y predecir su éxito basándose en características técnicas y de mercado.

### Pregunta Principal
> **¿Se puede predecir si un videojuego será exitoso a partir de sus características técnicas y de mercado?**

### Objetivos
- Clasificación: Predecir si un juego será exitoso (binario)
- Regresión: Estimar el número de propietarios de un juego
- Análisis no supervisado: PCA y Clustering para segmentación de mercado

---

## 🎯 Alcance

Este proyecto se centra en:
- ✅ Clasificación y Regresión supervisada
- ✅ Reducción de dimensionalidad (PCA)
- ✅ Clustering (K-Means)
- ❌ NO es un sistema de recomendación

### Modelos Implementados

#### Supervisados - Clasificación
- Regresión Logística
- Árbol de Decisión
- Random Forest
- Gradient Boosting
- Support Vector Machine (SVM)

#### Supervisados - Regresión
- Regresión Lineal
- Random Forest Regressor
- Gradient Boosting Regressor
- Support Vector Regression (SVR)

#### No Supervisados
- PCA (Principal Component Analysis)
- K-Means Clustering

---

## 📊 Dataset

**Fuente:** Steam Store Games Dataset / SteamDB

### Variables Principales

#### Variables Numéricas
- `price`: Precio del juego
- `positive_ratings`: Valoraciones positivas
- `negative_ratings`: Valoraciones negativas
- `positive_ratio`: Ratio de valoraciones positivas
- `playtime_forever`: Tiempo total de juego
- `owners_mid`: Número medio estimado de propietarios (variable objetivo)

#### Variables Categóricas
- `genres`: Géneros del juego (Action, RPG, Indie, Strategy, etc.)
- `categories`: Categorías (Single-player, Multiplayer, etc.)
- `platforms`: Plataformas (Windows, Mac, Linux)
- `developer`: Desarrollador
- `publisher`: Distribuidor

---

## 📁 Estructura del Proyecto

```
MMLL/
├── venv/                          # Entorno virtual Python
├── data/
│   ├── raw/                       # Datos originales sin procesar
│   └── processed/                 # Datos procesados y limpios
├── notebooks/
│   ├── 01_exploratory_analysis.ipynb
│   ├── 02_classification.ipynb
│   ├── 03_regression.ipynb
│   └── 04_unsupervised.ipynb
├── src/
│   ├── __init__.py
│   ├── preprocessing.py           # Limpieza y preprocesado
│   ├── classification.py          # Modelos de clasificación
│   ├── regression.py              # Modelos de regresión
│   ├── unsupervised.py            # PCA y Clustering
│   └── utils.py                   # Funciones auxiliares
├── models/                        # Modelos entrenados guardados
├── results/
│   ├── figures/                   # Gráficos y visualizaciones
│   └── metrics/                   # Métricas de evaluación
├── requirements.txt               # Dependencias del proyecto
├── .gitignore
└── README.md
```

---

## 🚀 Instalación y Configuración

### 1. Clonar el repositorio (o descargar)
```bash
cd MMLL
```

### 2. Crear y activar entorno virtual

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**macOS/Linux:**
```bash
python -m venv venv
source venv/bin/activate
```

### 3. Instalar dependencias
```bash
pip install -r requirements.txt
```

---

## 💻 Uso del Proyecto

### 1. Preprocesamiento de Datos
```bash
python src/preprocessing.py
```

### 2. Ejecutar Modelos de Clasificación
```bash
python src/classification.py
```

### 3. Ejecutar Modelos de Regresión
```bash
python src/regression.py
```

### 4. Análisis No Supervisado
```bash
python src/unsupervised.py
```

### 5. Exploración con Jupyter Notebooks
```bash
jupyter notebook
```

---

## 🎓 Definición del Problema

### Variable de Éxito

#### A) Clasificación (Binaria)
```python
success = 1  # Si owners_mid >= 100,000
success = 0  # Si owners_mid < 100,000
```
**Pregunta:** *¿El juego será exitoso o no?*

#### B) Regresión (Continua)
```python
target = owners_mid  # Número medio de propietarios
```
**Pregunta:** *¿Cuántos jugadores tendrá aproximadamente?*

---

## 📈 Métricas de Evaluación

### Clasificación
- Accuracy
- Precision / Recall / F1-Score
- ROC-AUC
- Matriz de Confusión

### Regresión
- MAE (Mean Absolute Error)
- RMSE (Root Mean Squared Error)
- R² (Coeficiente de determinación)

### Clustering
- Silhouette Score
- Análisis de varianza explicada (PCA)

---

## 🔬 Evitación de Data Leakage

Para garantizar la validez del modelo:

### Variables NO utilizadas como inputs:
- ❌ `owners` (es nuestra variable objetivo)
- ❌ `owners_mid` (derivada de la anterior)

### Escenarios de Predicción

**Escenario Pre-lanzamiento:**
- Precio
- Género
- Plataforma
- Developer/Publisher

**Escenario Post-lanzamiento:**
- Variables anteriores +
- Ratings
- Tiempo de juego

---

## 📊 Resultados Esperados

1. **Comparación de Modelos:** Identificar qué algoritmo predice mejor el éxito
2. **Feature Importance:** Qué variables son más influyentes
3. **Segmentación del Mercado:** Clusters interpretables de tipos de juegos
4. **Visualización PCA:** Estructura del mercado de Steam en 2D/3D

### Ejemplos de Clusters Esperables
- 🎮 Juegos AAA: caros, muy vendidos
- 🕹️ Juegos indie: baratos, alta valoración
- 🎯 Juegos de nicho: pocos jugadores, alto engagement

---

## 🛠️ Tecnologías Utilizadas

- **Python 3.8+**
- **scikit-learn:** Modelos de ML
- **pandas:** Manipulación de datos
- **numpy:** Operaciones numéricas
- **matplotlib / seaborn:** Visualización
- **jupyter:** Notebooks interactivos

---

## 📚 Contexto Académico

Este proyecto está alineado con los contenidos de:
- **PRDL** (Procesamiento de Datos y Lenguajes)
- **MMLB** (Modelos de Machine Learning Básicos)

### Temas Cubiertos
- Regresión Lineal y Logística
- Árboles de Decisión
- Random Forest
- Boosted Trees
- Support Vector Machines
- PCA
- Clustering

---

## 👥 Autores

Javier Sancho Alvarez

---

## 📝 Licencia

Este proyecto es de carácter académico.

---

## 🤝 Contribuciones

Si deseas contribuir:
1. Fork el repositorio
2. Crea una rama (`git checkout -b feature/nueva-funcionalidad`)
3. Commit tus cambios (`git commit -m 'Añade nueva funcionalidad'`)
4. Push a la rama (`git push origin feature/nueva-funcionalidad`)
5. Abre un Pull Request

---

## 📧 Contacto

Para preguntas o sugerencias sobre el proyecto, abre un issue en el repositorio.
