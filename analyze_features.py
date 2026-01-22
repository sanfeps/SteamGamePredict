"""
Script para analizar qué features exactas se generarán del dataset.
Muestra la lista completa de inputs que verá el modelo.
"""

import pandas as pd
import numpy as np
from collections import Counter
import sys
import os

sys.path.append('src')
from utils import parse_owners_range

def analyze_dataset_features(csv_path):
    """
    Analiza el dataset y muestra qué features se generarán.
    """
    print("\n" + "="*80)
    print("ANÁLISIS DE FEATURES - ¿QUÉ INFORMACIÓN USARÁ EL MODELO?")
    print("="*80 + "\n")

    # Cargar datos
    print(f"Cargando dataset desde: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"OK - Dataset cargado: {df.shape[0]} juegos, {df.shape[1]} columnas\n")

    # ====================================================================
    # PARTE 1: VARIABLES NUMÉRICAS DIRECTAS
    # ====================================================================
    print("="*80)
    print("1. VARIABLES NUMÉRICAS DIRECTAS (usadas tal cual, solo escaladas)")
    print("="*80)

    numeric_cols = ['price', 'positive_ratings', 'negative_ratings',
                   'average_playtime', 'median_playtime', 'achievements',
                   'required_age', 'english']

    available_numeric = [col for col in numeric_cols if col in df.columns]

    print(f"\nTotal: {len(available_numeric)} variables numéricas\n")

    for col in available_numeric:
        non_null = df[col].notna().sum()
        missing_pct = (df[col].isna().sum() / len(df)) * 100
        min_val = df[col].min()
        max_val = df[col].max()
        median_val = df[col].median()

        print(f"  ✓ {col:25s} | Valores: {non_null:6d} | Missing: {missing_pct:5.1f}% | "
              f"Rango: [{min_val:.1f}, {max_val:.1f}] | Mediana: {median_val:.1f}")

    # ====================================================================
    # PARTE 2: FEATURES ENGINEERED
    # ====================================================================
    print("\n" + "="*80)
    print("2. FEATURES ENGINEERED (creadas automáticamente)")
    print("="*80)

    engineered_features = []

    # Positive ratio
    if 'positive_ratings' in df.columns and 'negative_ratings' in df.columns:
        total = df['positive_ratings'] + df['negative_ratings']
        ratio = df['positive_ratings'] / (total + 1e-6)
        engineered_features.append({
            'name': 'positive_ratio',
            'description': 'Proporción de ratings positivos',
            'example': f"{ratio.median():.3f}",
            'range': f"[{ratio.min():.3f}, {ratio.max():.3f}]"
        })

    # Total ratings
    if 'positive_ratings' in df.columns and 'negative_ratings' in df.columns:
        total = df['positive_ratings'] + df['negative_ratings']
        engineered_features.append({
            'name': 'total_ratings',
            'description': 'Total de valoraciones',
            'example': f"{int(total.median())}",
            'range': f"[{int(total.min())}, {int(total.max())}]"
        })

    # Is free
    if 'price' in df.columns:
        free_count = (df['price'] == 0).sum()
        free_pct = (free_count / len(df)) * 100
        engineered_features.append({
            'name': 'is_free',
            'description': 'Si es gratis (0 o 1)',
            'example': f"1 para {free_pct:.1f}% de juegos",
            'range': '[0, 1]'
        })

    # Genre count
    if 'genres' in df.columns:
        genre_counts = df['genres'].dropna().apply(lambda x: len(str(x).split(';')) if isinstance(x, str) else 1)
        engineered_features.append({
            'name': 'genre_count',
            'description': 'Número de géneros del juego',
            'example': f"{int(genre_counts.median())}",
            'range': f"[{int(genre_counts.min())}, {int(genre_counts.max())}]"
        })

    # Platform count
    if 'platforms' in df.columns:
        platform_counts = df['platforms'].dropna().apply(lambda x: len(str(x).split(';')) if isinstance(x, str) else 1)
        engineered_features.append({
            'name': 'platform_count',
            'description': 'Número de plataformas',
            'example': f"{int(platform_counts.median())}",
            'range': f"[{int(platform_counts.min())}, {int(platform_counts.max())}]"
        })

    # Release date features
    if 'release_date' in df.columns:
        dates = pd.to_datetime(df['release_date'], errors='coerce')
        years = dates.dt.year
        engineered_features.append({
            'name': 'release_year',
            'description': 'Año de lanzamiento',
            'example': f"{int(years.median())}",
            'range': f"[{int(years.min())}, {int(years.max())}]"
        })
        engineered_features.append({
            'name': 'release_month',
            'description': 'Mes de lanzamiento (1-12)',
            'example': 'Mayo = 5',
            'range': '[1, 12]'
        })
        engineered_features.append({
            'name': 'release_quarter',
            'description': 'Trimestre (1-4)',
            'example': 'Q2 = 2',
            'range': '[1, 4]'
        })

    print(f"\nTotal: {len(engineered_features)} features engineered\n")
    for feat in engineered_features:
        print(f"  ✓ {feat['name']:25s} | {feat['description']:35s} | "
              f"Ejemplo: {feat['example']:15s} | Rango: {feat['range']}")

    # ====================================================================
    # PARTE 3: GÉNEROS (ONE-HOT)
    # ====================================================================
    print("\n" + "="*80)
    print("3. GÉNEROS (One-Hot Encoding) - Top 20 más frecuentes")
    print("="*80)

    if 'genres' in df.columns:
        all_genres = []
        for genres in df['genres'].dropna():
            if isinstance(genres, str):
                all_genres.extend([g.strip() for g in genres.split(';')])

        genre_counter = Counter(all_genres)
        top_20_genres = genre_counter.most_common(20)

        print(f"\nTotal géneros únicos encontrados: {len(genre_counter)}")
        print(f"Se usarán los top 20 más frecuentes como features binarias (0/1):\n")

        for i, (genre, count) in enumerate(top_20_genres, 1):
            pct = (count / len(df)) * 100
            col_name = f"genres_{genre}".replace(' ', '_').replace('-', '_').replace('&', 'and')
            print(f"  {i:2d}. {col_name:35s} | Aparece en {count:6d} juegos ({pct:5.1f}%)")

        genres_features = len(top_20_genres)
    else:
        genres_features = 0
        print("\n⚠ Columna 'genres' no encontrada")

    # ====================================================================
    # PARTE 4: CATEGORÍAS (ONE-HOT)
    # ====================================================================
    print("\n" + "="*80)
    print("4. CATEGORÍAS (One-Hot Encoding) - Top 20 más frecuentes")
    print("="*80)

    if 'categories' in df.columns:
        all_categories = []
        for cats in df['categories'].dropna():
            if isinstance(cats, str):
                all_categories.extend([c.strip() for c in cats.split(';')])

        cat_counter = Counter(all_categories)
        top_20_cats = cat_counter.most_common(20)

        print(f"\nTotal categorías únicas encontradas: {len(cat_counter)}")
        print(f"Se usarán las top 20 más frecuentes como features binarias (0/1):\n")

        for i, (cat, count) in enumerate(top_20_cats, 1):
            pct = (count / len(df)) * 100
            col_name = f"categories_{cat}".replace(' ', '_').replace('-', '_').replace('&', 'and')
            print(f"  {i:2d}. {col_name:45s} | Aparece en {count:6d} juegos ({pct:5.1f}%)")

        categories_features = len(top_20_cats)
    else:
        categories_features = 0
        print("\n⚠ Columna 'categories' no encontrada")

    # ====================================================================
    # PARTE 5: PLATAFORMAS (ONE-HOT)
    # ====================================================================
    print("\n" + "="*80)
    print("5. PLATAFORMAS (One-Hot Encoding)")
    print("="*80)

    if 'platforms' in df.columns:
        all_platforms = []
        for plats in df['platforms'].dropna():
            if isinstance(plats, str):
                all_platforms.extend([p.strip() for p in plats.split(';')])

        plat_counter = Counter(all_platforms)

        print(f"\nPlataformas encontradas:\n")

        for i, (plat, count) in enumerate(plat_counter.most_common(), 1):
            pct = (count / len(df)) * 100
            col_name = f"platforms_{plat}".replace(' ', '_').replace('-', '_')
            print(f"  {i}. {col_name:30s} | Aparece en {count:6d} juegos ({pct:5.1f}%)")

        platforms_features = len(plat_counter)
    else:
        platforms_features = 0
        print("\n⚠ Columna 'platforms' no encontrada")

    # ====================================================================
    # PARTE 6: DEVELOPERS (ONE-HOT)
    # ====================================================================
    print("\n" + "="*80)
    print("6. DEVELOPERS (One-Hot Encoding) - Top 20 más prolíficos")
    print("="*80)

    if 'developer' in df.columns:
        dev_counter = df['developer'].value_counts()
        top_20_devs = dev_counter.head(20)

        print(f"\nTotal developers únicos: {len(dev_counter)}")
        print(f"Se usarán los top 20 más prolíficos como features binarias (0/1):\n")

        for i, (dev, count) in enumerate(top_20_devs.items(), 1):
            pct = (count / len(df)) * 100
            col_name = f"developer_{dev}".replace(' ', '_').replace('-', '_').replace('&', 'and')[:50]
            print(f"  {i:2d}. {col_name:45s} | {count:4d} juegos ({pct:5.1f}%)")

        developers_features = len(top_20_devs)
    else:
        developers_features = 0
        print("\n⚠ Columna 'developer' no encontrada")

    # ====================================================================
    # PARTE 7: PUBLISHERS (ONE-HOT)
    # ====================================================================
    print("\n" + "="*80)
    print("7. PUBLISHERS (One-Hot Encoding) - Top 20 más prolíficos")
    print("="*80)

    if 'publisher' in df.columns:
        pub_counter = df['publisher'].value_counts()
        top_20_pubs = pub_counter.head(20)

        print(f"\nTotal publishers únicos: {len(pub_counter)}")
        print(f"Se usarán los top 20 más prolíficos como features binarias (0/1):\n")

        for i, (pub, count) in enumerate(top_20_pubs.items(), 1):
            pct = (count / len(df)) * 100
            col_name = f"publisher_{pub}".replace(' ', '_').replace('-', '_').replace('&', 'and')[:50]
            print(f"  {i:2d}. {col_name:45s} | {count:4d} juegos ({pct:5.1f}%)")

        publishers_features = len(top_20_pubs)
    else:
        publishers_features = 0
        print("\n⚠ Columna 'publisher' no encontrada")

    # ====================================================================
    # PARTE 8: VARIABLE OBJETIVO
    # ====================================================================
    print("\n" + "="*80)
    print("8. VARIABLE OBJETIVO (no se usa como input, es lo que predecimos)")
    print("="*80)

    if 'owners' in df.columns:
        df['owners_mid'] = df['owners'].apply(parse_owners_range)

        print("\n📊 Distribución de propietarios:\n")
        print(f"  Mínimo:   {df['owners_mid'].min():,.0f} propietarios")
        print(f"  Mediana:  {df['owners_mid'].median():,.0f} propietarios")
        print(f"  Media:    {df['owners_mid'].mean():,.0f} propietarios")
        print(f"  Máximo:   {df['owners_mid'].max():,.0f} propietarios")

        # Clasificación con threshold 100k
        success = (df['owners_mid'] >= 100000).astype(int)
        success_rate = success.mean() * 100

        print(f"\n🎯 Clasificación (threshold = 100,000 propietarios):")
        print(f"  Exitosos (≥100k):     {success.sum():5d} juegos ({success_rate:.1f}%)")
        print(f"  No exitosos (<100k):  {(~success.astype(bool)).sum():5d} juegos ({100-success_rate:.1f}%)")

    # ====================================================================
    # RESUMEN FINAL
    # ====================================================================
    print("\n" + "="*80)
    print("📊 RESUMEN FINAL DE FEATURES")
    print("="*80 + "\n")

    total_features = (len(available_numeric) + len(engineered_features) +
                     genres_features + categories_features + platforms_features +
                     developers_features + publishers_features)

    print(f"  1. Variables numéricas directas:     {len(available_numeric):3d} features")
    print(f"  2. Features engineered:               {len(engineered_features):3d} features")
    print(f"  3. Géneros (one-hot):                 {genres_features:3d} features")
    print(f"  4. Categorías (one-hot):              {categories_features:3d} features")
    print(f"  5. Plataformas (one-hot):             {platforms_features:3d} features")
    print(f"  6. Developers (one-hot):              {developers_features:3d} features")
    print(f"  7. Publishers (one-hot):              {publishers_features:3d} features")
    print(f"  " + "-"*60)
    print(f"  TOTAL DE FEATURES (INPUT AL MODELO):  {total_features:3d} features")

    print("\n" + "="*80)
    print("✅ CONCLUSIÓN SOBRE LA INFORMACIÓN")
    print("="*80 + "\n")

    print("El modelo tiene acceso a:")
    print("  ✓ Información de precio y modelo de negocio")
    print("  ✓ Métricas de engagement (ratings, playtime)")
    print("  ✓ Características técnicas del juego (géneros, categorías)")
    print("  ✓ Alcance de plataforma")
    print("  ✓ Reputación del desarrollador/publisher")
    print("  ✓ Contexto temporal (cuándo se lanzó)")
    print("  ✓ Features derivadas que capturan calidad percibida")

    print(f"\n💡 Con {total_features} features, el modelo tiene INFORMACIÓN SUFICIENTE para:")
    print("  • Distinguir entre juegos indie y AAA")
    print("  • Capturar tendencias de género/categoría populares")
    print("  • Identificar publishers/developers exitosos")
    print("  • Considerar el modelo de negocio (F2P vs pago)")
    print("  • Evaluar la calidad percibida (positive_ratio)")
    print("  • Detectar patrones temporales (año de lanzamiento)")

    if total_features < 50:
        print("\n⚠ ADVERTENCIA: Pocas features. Considera añadir más información.")
    elif total_features > 200:
        print("\n⚠ ADVERTENCIA: Muchas features. Riesgo de overfitting. Considera reducir.")
    else:
        print(f"\n✅ {total_features} features es un número ÓPTIMO para este tipo de problema.")

    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    csv_path = os.path.join('data', 'raw', 'steam_games.csv')

    if not os.path.exists(csv_path):
        print(f"ERROR: No se encuentra el dataset en {csv_path}")
        print("Por favor, asegúrate de tener el archivo steam_games.csv en data/raw/")
    else:
        analyze_dataset_features(csv_path)
