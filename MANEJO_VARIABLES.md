# Manejo de Variables en el Proyecto Steam Games ML

Este documento explica cómo el pipeline maneja cada tipo de variable del dataset de Steam.

---

## 📊 Tipos de Variables en el Dataset

### 1. Variables Numéricas (Listas para usar)

Estas variables ya son números y se usan directamente:

| Variable | Descripción | Uso en el modelo |
|----------|-------------|------------------|
| `price` | Precio del juego | ✅ Se usa directamente |
| `positive_ratings` | Valoraciones positivas | ✅ Se usa directamente |
| `negative_ratings` | Valoraciones negativas | ✅ Se usa directamente |
| `average_playtime` | Tiempo medio de juego | ✅ Se usa directamente |
| `median_playtime` | Tiempo mediano de juego | ✅ Se usa directamente |
| `achievements` | Número de logros | ✅ Se usa directamente |
| `required_age` | Edad requerida | ✅ Se usa directamente |

**Procesamiento adicional:**
- Se escalan usando `StandardScaler` para que todas estén en la misma escala
- Se crean features derivadas como `positive_ratio`, `total_ratings`

---

### 2. Variables Categóricas Multi-Valor (Separadas por `;`)

Estas columnas contienen múltiples valores en una sola celda:

#### **`genres` (Géneros)**

**Ejemplo de datos raw:**
```
"Action;FPS;Multiplayer"
"RPG;Strategy;Fantasy"
"Indie;Puzzle"
```

**Procesamiento (One-Hot Encoding Multi-Valor):**

El código:
1. Separa cada género individual por el `;`
2. Cuenta cuántas veces aparece cada género en todo el dataset
3. Toma los **20 géneros más frecuentes** (configurable)
4. Crea una columna binaria para cada género

**Resultado:**
```
         genres_Action  genres_FPS  genres_RPG  genres_Indie  ...
Juego 1       1            1           0            0
Juego 2       0            0           1            0
Juego 3       0            0           0            1
```

**Géneros esperables:** Action, RPG, Strategy, Indie, Adventure, Simulation, FPS, Puzzle, etc.

#### **`categories` (Categorías)**

**Ejemplo de datos raw:**
```
"Single-player;Multi-player;Steam Achievements"
"Co-op;Steam Trading Cards"
```

**Procesamiento:** Igual que genres
- Se separan por `;`
- Top 20 categorías más frecuentes
- Columnas binarias

**Categorías esperables:** Single-player, Multi-player, Co-op, Steam Achievements, Steam Cloud, etc.

#### **`platforms` (Plataformas)**

**Ejemplo de datos raw:**
```
"windows;mac;linux"
"windows"
```

**Procesamiento:** Igual que genres
- Se separan por `;`
- Normalmente habrá solo 3: windows, mac, linux
- Columnas binarias para cada una

**Resultado:**
```
         platforms_windows  platforms_mac  platforms_linux
Juego 1        1                1              1
Juego 2        1                0              0
```

#### **`steamspy_tags` (Tags)**

**Ejemplo:** Similar a genres pero con más variedad

**Decisión:** Actualmente NO se usa en el modelo porque:
- Hay demasiados tags únicos (cientos)
- Se solapa mucho con `genres`
- Añadiría demasiadas features

Si quieres usarlo, puedes añadirlo editando `preprocessing.py` línea 342:
```python
categorical_cols = ['genres', 'categories', 'platforms', 'steamspy_tags']
```

---

### 3. Variables de Identificación (NO se usan)

Estas variables se EXCLUYEN del modelo porque no aportan información predictiva:

| Variable | ¿Por qué NO se usa? |
|----------|---------------------|
| `appid` | ID único, no tiene relación con el éxito |
| `name` | Nombre del juego, causaría overfitting |
| `release_date` | Fecha, podría usarse pero requiere procesamiento especial |

**Código de exclusión** (`preprocessing.py` líneas 281-284):
```python
exclude_cols = [target_col, 'owners', 'owners_mid', 'success', 'name',
               'appid', 'release_date', 'developer', 'publisher',
               'genres', 'categories', 'platforms', 'price_category']
```

---

### 4. Variables Categóricas Únicas (Developer/Publisher)

#### **`developer` y `publisher`**

**Problema:**
- Hay miles de desarrolladores/publishers únicos
- Valve, EA, Ubisoft aparecen mucho
- Pero hay cientos que solo tienen 1-2 juegos

**Solución actual:** Se EXCLUYEN del modelo

**¿Por qué?**
- Demasiadas categorías únicas (miles)
- Crear columnas para todos causaría:
  - Dimensionalidad extrema (curse of dimensionality)
  - Overfitting
  - Problemas de memoria

**Alternativas (si quieres mejorar):**

**Opción 1:** Solo top publishers/developers
```python
# En preprocessing.py, cambiar línea 342 a:
categorical_cols = ['genres', 'categories', 'platforms', 'developer', 'publisher']

# Y ajustar max_categories a un número bajo:
max_categories=10  # Solo top 10 developers/publishers
```

**Opción 2:** Crear feature "is_big_publisher"
```python
big_publishers = ['Valve', 'Electronic Arts', 'Ubisoft', 'Activision']
df['is_big_publisher'] = df['publisher'].isin(big_publishers).astype(int)
```

---

### 5. Variable Objetivo (Target)

#### **`owners` → `owners_mid` → `success`**

**Transformación:**

1. **Raw:** `"10000000-20000000"` (rango como texto)

2. **`owners_mid`** (numérico para regresión):
   ```python
   parse_owners_range("10000000-20000000") → 15000000
   ```
   Toma el punto medio del rango

3. **`success`** (binario para clasificación):
   ```python
   owners_mid >= 100000 → 1 (éxito)
   owners_mid < 100000  → 0 (no éxito)
   ```

**Configuración del umbral:**

Puedes cambiar el umbral de éxito en `preprocessing.py` línea 23:
```python
self.success_threshold = 100000  # Cambia este valor
```

Opciones razonables:
- `50000` - Umbral bajo (más juegos exitosos)
- `100000` - Umbral medio (default)
- `500000` - Umbral alto (solo juegos muy exitosos)
- `1000000` - Umbral muy alto (solo blockbusters)

---

## 🔄 Flujo de Procesamiento Completo

### Paso 1: Carga de Datos
```
steam_games.csv (raw)
  ↓
DataFrame con todas las columnas originales
```

### Paso 2: Limpieza
```
- Eliminar duplicados
- Manejar valores nulos:
  - Numéricos → Mediana
  - Categóricos → 'Unknown'
- Parsear owners a numérico
```

### Paso 3: Feature Engineering
```
Crear nuevas features:
- positive_ratio = positive / (positive + negative)
- total_ratings = positive + negative
- is_free = 1 si price == 0
- price_category = ['Free', 'Budget', 'Standard', 'Premium', 'AAA']
- genre_count = número de géneros
- platform_count = número de plataformas
- success = 1 si owners_mid >= threshold
```

### Paso 4: Encoding Categórico
```
Para genres, categories, platforms:
1. Separar por ';'
2. Contar frecuencias
3. Tomar top 20 más comunes
4. Crear columnas binarias

Resultado: ~60-80 columnas adicionales
```

### Paso 5: Selección de Features
```
Excluir:
- Identificadores (appid, name)
- Variables objetivo (owners, owners_mid, success)
- Variables originales ya codificadas (genres, categories, platforms)
- Fechas sin procesar (release_date)
- Developer/publisher (opcionales)

Incluir:
- Todas las numéricas originales
- Features engineered
- Columnas one-hot de genres/categories/platforms
```

### Paso 6: Escalado
```
StandardScaler:
- Media = 0
- Desviación estándar = 1

Ejemplo:
  price: [0, 5.99, 19.99, 59.99]
    ↓
  price_scaled: [-0.5, -0.3, 0.2, 1.8]
```

### Paso 7: Train-Test Split
```
80% Train - 20% Test
Stratified (para clasificación)
```

---

## 📈 Dimensionalidad Esperada

Con el dataset de Steam y configuración por defecto:

**Variables numéricas originales:** ~8
- price, positive_ratings, negative_ratings, average_playtime, median_playtime, achievements, required_age

**Features engineered:** ~5
- positive_ratio, total_ratings, is_free, genre_count, platform_count

**One-hot genres:** ~20 columnas
- genres_Action, genres_RPG, genres_Strategy, ...

**One-hot categories:** ~20 columnas
- categories_Single_player, categories_Multi_player, ...

**One-hot platforms:** ~3 columnas
- platforms_windows, platforms_mac, platforms_linux

**Total esperado: ~56 features**

---

## ⚙️ Configuración Personalizable

### Cambiar número de categorías top:

En `preprocessing.py` línea 347:
```python
df_processed = self.encode_categorical_features(
    df_feat,
    categorical_cols=categorical_cols,
    method='onehot',
    max_categories=20  # ← Cambia esto (10-50 recomendado)
)
```

- `max_categories=10` → Menos features, más rápido, posible pérdida de info
- `max_categories=50` → Más features, más lento, más información

### Añadir/quitar columnas categóricas:

En `preprocessing.py` línea 342:
```python
categorical_cols = ['genres', 'categories', 'platforms']

# Opción: añadir más
categorical_cols = ['genres', 'categories', 'platforms', 'steamspy_tags']

# Opción: añadir developer/publisher (solo top)
categorical_cols = ['genres', 'categories', 'platforms', 'developer', 'publisher']
```

---

## 🎯 Resumen: ¿Qué se usa y qué no?

### ✅ SE USA EN EL MODELO

| Tipo | Variables | Cómo |
|------|-----------|------|
| Numéricas | price, ratings, playtime, achievements, age | Directamente (escaladas) |
| Features | positive_ratio, total_ratings, is_free, counts | Creadas automáticamente |
| Géneros | Top 20 géneros más frecuentes | One-hot encoding |
| Categorías | Top 20 categorías más frecuentes | One-hot encoding |
| Plataformas | windows, mac, linux | One-hot encoding |

### ❌ NO SE USA EN EL MODELO

| Variable | Razón |
|----------|-------|
| appid | Identificador único |
| name | Nombre del juego (no predictivo) |
| release_date | Fecha (requiere procesamiento especial) |
| developer | Demasiadas categorías únicas |
| publisher | Demasiadas categorías únicas |
| steamspy_tags | Redundante con genres |
| owners (original) | Es la variable objetivo |

---

## 💡 Recomendaciones

1. **Ejecuta primero con la configuración por defecto** para ver resultados base

2. **Experimenta con el umbral de éxito:**
   - Prueba 50k, 100k, 500k para ver cómo cambia el balance de clases

3. **Analiza feature importance** después de entrenar:
   - Te dirá qué géneros/categorías son más importantes
   - Puedes reducir features eliminando las menos importantes

4. **Si tienes problemas de memoria:**
   - Reduce `max_categories` de 20 a 10
   - Usa una muestra del dataset primero

5. **Para investigación más profunda:**
   - Considera añadir `release_date` procesada (año, mes)
   - Crea features como "días desde lanzamiento"
   - Agrupa developers/publishers en "indie" vs "AAA"

---

¿Alguna duda sobre el manejo de variables?
