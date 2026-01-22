# Features Usadas en el Modelo - Lista Completa

Este documento lista **TODAS** las variables que realmente se usan en el modelo de Machine Learning.

---

## ✅ FEATURES FINALES (Aproximadamente 100-120 variables)

### 1. Variables Numéricas Originales (~8 features)

| Feature | Descripción | Ejemplo |
|---------|-------------|---------|
| `price` | Precio del juego | 19.99 |
| `positive_ratings` | Número de valoraciones positivas | 45000 |
| `negative_ratings` | Número de valoraciones negativas | 2000 |
| `average_playtime` | Tiempo medio de juego (minutos) | 890 |
| `median_playtime` | Tiempo mediano de juego | 450 |
| `achievements` | Número de logros del juego | 50 |
| `required_age` | Edad mínima requerida | 18 |
| `english` | Si está en inglés (0/1) | 1 |

**Todas estas se ESCALAN** con StandardScaler antes del modelo.

---

### 2. Features Engineered - Creadas Automáticamente (~8 features)

| Feature | Fórmula | Descripción |
|---------|---------|-------------|
| `positive_ratio` | positive / (positive + negative) | Proporción de valoraciones positivas (0-1) |
| `total_ratings` | positive + negative | Total de valoraciones |
| `is_free` | 1 si price == 0 | Indicador de juego gratis |
| `genre_count` | Número de géneros | Cuántos géneros tiene el juego |
| `platform_count` | Número de plataformas | En cuántas plataformas está disponible |
| `release_year` | Año extraído de release_date | 2015, 2020, etc. |
| `release_month` | Mes extraído de release_date | 1-12 |
| `release_quarter` | Trimestre (1-4) | Q1, Q2, Q3, Q4 |

**Importancia de estas features:**
- `positive_ratio`: Muy importante - indica calidad percibida
- `total_ratings`: Indica visibilidad/popularidad
- `release_year`: Tendencias del mercado cambian con el tiempo
- `is_free`: Juegos F2P tienen dinámica diferente

---

### 3. Géneros - One-Hot Encoded (~20 features)

**Top 20 géneros más frecuentes** (ejemplo basado en Steam):

| Feature | Juegos que lo tienen |
|---------|----------------------|
| `genres_Action` | 1 si tiene "Action" en sus géneros |
| `genres_Indie` | 1 si tiene "Indie" |
| `genres_Adventure` | 1 si tiene "Adventure" |
| `genres_Casual` | 1 si tiene "Casual" |
| `genres_Strategy` | 1 si tiene "Strategy" |
| `genres_RPG` | 1 si tiene "RPG" |
| `genres_Simulation` | 1 si tiene "Simulation" |
| `genres_Early_Access` | 1 si tiene "Early Access" |
| `genres_Free_to_Play` | 1 si tiene "Free to Play" |
| `genres_Sports` | 1 si tiene "Sports" |
| `genres_Racing` | 1 si tiene "Racing" |
| `genres_Massively_Multiplayer` | 1 si tiene "Massively Multiplayer" |
| ... | (hasta 20 total) |

**Un juego puede tener MÚLTIPLES géneros = 1 simultáneamente**

Ejemplo:
- Counter-Strike: `genres_Action=1`, `genres_FPS=1`, `genres_Multiplayer=1`
- The Witcher 3: `genres_RPG=1`, `genres_Action=1`, `genres_Adventure=1`

---

### 4. Categorías - One-Hot Encoded (~20 features)

**Top 20 categorías más frecuentes** (características de Steam):

| Feature | Descripción |
|---------|-------------|
| `categories_Single_player` | Tiene modo single-player |
| `categories_Multi_player` | Tiene modo multi-player |
| `categories_Co_op` | Tiene modo cooperativo |
| `categories_Steam_Achievements` | Tiene logros de Steam |
| `categories_Steam_Trading_Cards` | Tiene cromos coleccionables |
| `categories_Steam_Workshop` | Soporte para mods/workshop |
| `categories_Steam_Cloud` | Guardado en la nube |
| `categories_Full_controller_support` | Soporte completo de mando |
| `categories_Partial_Controller_Support` | Soporte parcial de mando |
| `categories_VR_Support` | Soporte de realidad virtual |
| `categories_Captions_available` | Subtítulos disponibles |
| `categories_In_App_Purchases` | Compras dentro del juego |
| `categories_Online_Multi_Player` | Multijugador online |
| `categories_Local_Multi_Player` | Multijugador local |
| `categories_Online_Co_op` | Cooperativo online |
| ... | (hasta 20 total) |

**Importancia:** Características técnicas que influyen en ventas
- VR games tienen nicho específico
- Multiplayer games tienen mayor longevidad
- Steam Workshop aumenta engagement

---

### 5. Plataformas - One-Hot Encoded (~3 features)

| Feature | Descripción |
|---------|-------------|
| `platforms_windows` | Disponible en Windows |
| `platforms_mac` | Disponible en macOS |
| `platforms_linux` | Disponible en Linux |

**Importancia:**
- Windows = Mayor mercado
- Mac + Linux + Windows = Mayor alcance
- Solo Linux = Nicho muy específico

---

### 6. Desarrolladores - One-Hot Encoded (~20 features)

**Top 20 desarrolladores más prolíficos:**

| Feature | Ejemplo |
|---------|---------|
| `developer_Valve` | Desarrollado por Valve |
| `developer_SEGA` | Desarrollado por SEGA |
| `developer_Ubisoft` | Desarrollado por Ubisoft |
| `developer_BANDAI_NAMCO` | Desarrollado por Bandai Namco |
| `developer_Square_Enix` | Desarrollado por Square Enix |
| `developer_Deep_Silver` | Desarrollado por Deep Silver |
| ... | (hasta 20 total) |

**Importancia:**
- Valve, Blizzard, etc. tienen reconocimiento de marca
- Desarrolladores indie pequeños no aparecen (quedan en "otros")
- Solo los top 20 más frecuentes se incluyen

---

### 7. Publishers - One-Hot Encoded (~20 features)

**Top 20 publishers más prolíficos:**

| Feature | Ejemplo |
|---------|---------|
| `publisher_Valve` | Publicado por Valve |
| `publisher_SEGA` | Publicado por SEGA |
| `publisher_Ubisoft` | Publicado por Ubisoft |
| `publisher_Electronic_Arts` | Publicado por EA |
| `publisher_Activision` | Publicado por Activision |
| `publisher_2K` | Publicado por 2K |
| `publisher_Bethesda` | Publicado por Bethesda |
| ... | (hasta 20 total) |

**Importancia:**
- Publishers grandes tienen presupuesto de marketing
- Reconocimiento de marca
- Diferentes estrategias de pricing

---

## 📊 Resumen de Dimensionalidad

| Categoría | Número de Features |
|-----------|-------------------|
| Numéricas originales | ~8 |
| Features engineered | ~8 |
| Géneros (one-hot) | ~20 |
| Categorías (one-hot) | ~20 |
| Plataformas (one-hot) | ~3 |
| Developers (one-hot) | ~20 |
| Publishers (one-hot) | ~20 |
| **TOTAL APROXIMADO** | **~99-120 features** |

---

## ❌ Lo que NO se usa (Columnas Excluidas)

### Excluidas Correctamente:

| Columna | ¿Por qué NO se usa? |
|---------|---------------------|
| `appid` | ID único, no aporta información predictiva |
| `name` | Nombre del juego, causaría overfitting |
| `owners` (original) | Es la variable objetivo (versión texto) |
| `owners_mid` | Variable objetivo para regresión |
| `success` | Variable objetivo para clasificación |
| `release_date` (original) | Reemplazada por year/month/quarter |
| `genres` (original) | Reemplazada por columnas one-hot |
| `categories` (original) | Reemplazada por columnas one-hot |
| `platforms` (original) | Reemplazada por columnas one-hot |
| `developer` (original) | Reemplazada por columnas one-hot |
| `publisher` (original) | Reemplazada por columnas one-hot |
| `price_category` (original) | Categórica creada, pero ya tenemos `price` numérica y `is_free` |
| `steamspy_tags` | Redundante con géneros (opcional, puedes activarlo) |

---

## 🎯 Ejemplo Concreto: Counter-Strike: Global Offensive

**Datos originales:**
```
name: "Counter-Strike: Global Offensive"
price: 0.0
positive_ratings: 2644404
negative_ratings: 402313
genres: "Action;Free to Play"
categories: "Multi-player;Steam Achievements;..."
platforms: "windows;mac;linux"
developer: "Valve;Hidden Path Entertainment"
publisher: "Valve"
release_date: "2012-08-21"
owners: "50000000-100000000"
```

**Features que ve el modelo (parcial):**
```python
{
    # Numéricas
    'price': 0.0,  # (después escalado: -1.2)
    'positive_ratings': 2644404,  # (después escalado: 3.8)
    'negative_ratings': 402313,  # (después escalado: 2.1)

    # Engineered
    'positive_ratio': 0.868,  # (después escalado: 1.5)
    'total_ratings': 3046717,  # (después escalado: 3.9)
    'is_free': 1,
    'genre_count': 2,
    'platform_count': 3,
    'release_year': 2012,  # (después escalado: -0.5)
    'release_month': 8,
    'release_quarter': 3,

    # Géneros (one-hot)
    'genres_Action': 1,
    'genres_Free_to_Play': 1,
    'genres_FPS': 1,
    'genres_RPG': 0,
    'genres_Indie': 0,
    # ... resto de géneros

    # Categorías (one-hot)
    'categories_Multi_player': 1,
    'categories_Steam_Achievements': 1,
    'categories_Single_player': 0,
    # ... resto de categorías

    # Plataformas (one-hot)
    'platforms_windows': 1,
    'platforms_mac': 1,
    'platforms_linux': 1,

    # Developer/Publisher (one-hot)
    'developer_Valve': 1,
    'publisher_Valve': 1,
    'developer_Ubisoft': 0,
    # ... resto
}
```

**Total:** ~100-120 números que el modelo usa para predecir.

---

## 🔍 Cómo Verificar Qué Features se Usan

Después de ejecutar el preprocesamiento, puedes ver las features exactas:

```python
# En preprocessing.py, se imprime:
print(f"Features: {feature_cols}")

# O después de cargar datos procesados:
import pandas as pd
df = pd.read_csv('data/processed/steam_games_processed.csv')
print(df.columns.tolist())
```

---

## 💡 Importancia de Features Esperada

Basándome en análisis típicos de Steam:

**Top 10 features más importantes** (estimación):

1. `positive_ratio` - Calidad percibida
2. `total_ratings` - Popularidad/visibilidad
3. `release_year` - Tendencias del mercado
4. `price` - Pricing strategy
5. `genres_Action` - Género más popular
6. `genres_Indie` - Mercado indie vs AAA
7. `categories_Multi_player` - Longevidad
8. `publisher_Valve` - Marca reconocida
9. `platforms_windows` - Alcance de mercado
10. `is_free` - Modelo de negocio F2P

Después de entrenar, verás la importancia real en los plots de feature importance.

---

## 🎓 Conclusión

**AHORA SÍ** se usan las variables importantes que mencionaste:
- ✅ Release date → Convertida a year/month/quarter
- ✅ Genres → 20 columnas binarias de géneros
- ✅ Categories → 20 columnas binarias de categorías
- ✅ Platforms → 3 columnas binarias
- ✅ Developers → Top 20 como columnas binarias
- ✅ Publishers → Top 20 como columnas binarias

**La confusión era:** El código excluye las columnas ORIGINALES de texto, pero usa las versiones TRANSFORMADAS a números.
