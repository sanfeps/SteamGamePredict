"""
Script interactivo para predecir el exito de un videojuego.
Introduce los parametros de tu juego y obtén una prediccion.
"""

import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime


def load_models():
    """Carga los mejores modelos entrenados."""
    models = {}

    # Buscar el modelo de clasificacion mas reciente (Gradient Boosting)
    classification_dir = 'models/classification'
    if os.path.exists(classification_dir):
        for f in os.listdir(classification_dir):
            if 'gradient_boosting' in f and f.endswith('.joblib'):
                models['classification'] = joblib.load(os.path.join(classification_dir, f))
                print(f"OK - Modelo de clasificacion cargado: {f}")
                break

    # Buscar el modelo de regresion mas reciente (Gradient Boosting)
    regression_dir = 'models/regression'
    if os.path.exists(regression_dir):
        for f in os.listdir(regression_dir):
            if 'gradient_boosting' in f and f.endswith('.joblib'):
                models['regression'] = joblib.load(os.path.join(regression_dir, f))
                print(f"OK - Modelo de regresion cargado: {f}")
                break

    # Cargar el scaler
    if os.path.exists('models/scaler.joblib'):
        models['scaler'] = joblib.load('models/scaler.joblib')
        print("OK - Scaler cargado")

    return models


def get_feature_names():
    """Retorna la lista de features que espera el modelo."""
    # Cargar datos procesados para obtener las columnas
    processed_path = 'data/processed/steam_games_processed.csv'
    if os.path.exists(processed_path):
        df = pd.read_csv(processed_path, nrows=1)

        # Excluir columnas objetivo y de identificacion
        exclude_cols = ['owners', 'owners_mid', 'success', 'name', 'appid',
                       'release_date', 'developer', 'publisher', 'genres',
                       'categories', 'platforms', 'price_category', 'steamspy_tags']

        feature_cols = [col for col in df.columns
                       if col not in exclude_cols and df[col].dtype in ['int64', 'float64']]
        return feature_cols
    return None


def create_game_features(game_data, feature_names):
    """
    Crea el vector de features a partir de los datos del juego.
    """
    # Inicializar todas las features a 0
    features = {name: 0 for name in feature_names}

    # === VARIABLES NUMERICAS ===
    features['price'] = game_data.get('price', 0)
    features['achievements'] = game_data.get('achievements', 0)
    features['english'] = game_data.get('english', 1)  # Por defecto en ingles
    features['required_age'] = game_data.get('required_age', 0)  # Por defecto para todos

    # Para un juego nuevo, no tenemos ratings aun
    # Usamos valores estimados o promedios
    features['positive_ratings'] = game_data.get('positive_ratings', 0)
    features['negative_ratings'] = game_data.get('negative_ratings', 0)
    features['average_playtime'] = game_data.get('average_playtime', 0)
    features['median_playtime'] = game_data.get('median_playtime', 0)

    # === FEATURES ENGINEERED ===
    total_ratings = features['positive_ratings'] + features['negative_ratings']
    if total_ratings > 0:
        features['positive_ratio'] = features['positive_ratings'] / total_ratings
    else:
        features['positive_ratio'] = 0.75  # Valor neutral para juegos nuevos

    features['total_ratings'] = total_ratings
    features['is_free'] = 1 if features['price'] == 0 else 0

    # Fecha de lanzamiento
    release_year = game_data.get('release_year', datetime.now().year)
    features['release_year'] = release_year
    features['release_month'] = game_data.get('release_month', 6)
    features['release_quarter'] = (features['release_month'] - 1) // 3 + 1

    # Contar generos y plataformas
    genres = game_data.get('genres', [])
    platforms = game_data.get('platforms', [])
    features['genre_count'] = len(genres)
    features['platform_count'] = len(platforms)

    # === ONE-HOT ENCODING DE GENEROS ===
    genre_mapping = {
        'indie': 'genres_Indie',
        'action': 'genres_Action',
        'casual': 'genres_Casual',
        'adventure': 'genres_Adventure',
        'strategy': 'genres_Strategy',
        'simulation': 'genres_Simulation',
        'rpg': 'genres_RPG',
        'early access': 'genres_Early_Access',
        'free to play': 'genres_Free_to_Play',
        'sports': 'genres_Sports',
        'racing': 'genres_Racing',
        'multiplayer': 'genres_Massively_Multiplayer',
    }

    for genre in genres:
        genre_lower = genre.lower().strip()
        if genre_lower in genre_mapping:
            col_name = genre_mapping[genre_lower]
            if col_name in features:
                features[col_name] = 1

    # === ONE-HOT ENCODING DE CATEGORIAS ===
    category_mapping = {
        'single-player': 'categories_Single_player',
        'singleplayer': 'categories_Single_player',
        'multi-player': 'categories_Multi_player',
        'multiplayer': 'categories_Multi_player',
        'co-op': 'categories_Co_op',
        'coop': 'categories_Co_op',
        'steam achievements': 'categories_Steam_Achievements',
        'achievements': 'categories_Steam_Achievements',
        'steam trading cards': 'categories_Steam_Trading_Cards',
        'trading cards': 'categories_Steam_Trading_Cards',
        'steam cloud': 'categories_Steam_Cloud',
        'cloud saves': 'categories_Steam_Cloud',
        'controller support': 'categories_Full_controller_support',
        'full controller': 'categories_Full_controller_support',
        'partial controller': 'categories_Partial_Controller_Support',
        'online multiplayer': 'categories_Online_Multi_Player',
        'local multiplayer': 'categories_Local_Multi_Player',
        'online co-op': 'categories_Online_Co_op',
        'local co-op': 'categories_Local_Co_op',
        'workshop': 'categories_Steam_Workshop',
        'steam workshop': 'categories_Steam_Workshop',
        'level editor': 'categories_Includes_level_editor',
        'in-app purchases': 'categories_In_App_Purchases',
        'microtransactions': 'categories_In_App_Purchases',
    }

    categories = game_data.get('categories', [])
    for cat in categories:
        cat_lower = cat.lower().strip()
        if cat_lower in category_mapping:
            col_name = category_mapping[cat_lower]
            if col_name in features:
                features[col_name] = 1

    # === ONE-HOT ENCODING DE PLATAFORMAS ===
    platform_mapping = {
        'windows': 'platforms_windows',
        'pc': 'platforms_windows',
        'mac': 'platforms_mac',
        'macos': 'platforms_mac',
        'linux': 'platforms_linux',
    }

    for plat in platforms:
        plat_lower = plat.lower().strip()
        if plat_lower in platform_mapping:
            col_name = platform_mapping[plat_lower]
            if col_name in features:
                features[col_name] = 1

    # === ONE-HOT ENCODING DE PUBLISHER ===
    publisher = game_data.get('publisher', '').lower()
    publisher_mapping = {
        'valve': 'publisher_Valve',
        'ubisoft': 'publisher_Ubisoft',
        'square enix': 'publisher_Square_Enix',
        'sega': 'publisher_SEGA',
        'thq nordic': 'publisher_THQ_Nordic',
        'devolver digital': 'publisher_Devolver_Digital',
        'big fish games': 'publisher_Big_Fish_Games',
    }

    for pub_name, col_name in publisher_mapping.items():
        if pub_name in publisher and col_name in features:
            features[col_name] = 1
            break

    # Convertir a DataFrame con el orden correcto
    features_df = pd.DataFrame([features])

    # Asegurar que tenemos todas las columnas en el orden correcto
    for col in feature_names:
        if col not in features_df.columns:
            features_df[col] = 0

    features_df = features_df[feature_names]

    return features_df


def predict_game(game_data, models, feature_names):
    """
    Realiza la prediccion para un juego.
    """
    # Crear features
    X = create_game_features(game_data, feature_names)

    results = {}

    # Prediccion de clasificacion (exito/no exito)
    if 'classification' in models:
        clf = models['classification']
        prediction = clf.predict(X)[0]
        proba = clf.predict_proba(X)[0]

        results['success'] = bool(prediction)
        results['success_probability'] = proba[1]  # Probabilidad de exito
        results['failure_probability'] = proba[0]  # Probabilidad de fracaso

    # Prediccion de regresion (numero de owners)
    if 'regression' in models:
        reg = models['regression']
        owners_pred = reg.predict(X)[0]
        results['predicted_owners'] = max(0, owners_pred)  # No puede ser negativo

    return results


def get_user_input():
    """
    Obtiene los datos del juego del usuario de forma interactiva.
    """
    print("\n" + "="*60)
    print("PREDICTOR DE EXITO DE VIDEOJUEGOS EN STEAM")
    print("="*60)
    print("\nIntroduce los datos de tu juego:\n")

    game_data = {}

    # Precio
    while True:
        try:
            price = input("Precio del juego en USD (ej: 19.99, 0 si es gratis): ")
            game_data['price'] = float(price)
            break
        except ValueError:
            print("Por favor, introduce un numero valido.")

    # Generos
    print("\nGeneros disponibles: Indie, Action, Casual, Adventure, Strategy,")
    print("                     Simulation, RPG, Early Access, Free to Play,")
    print("                     Sports, Racing, Multiplayer")
    genres_input = input("Generos (separados por coma, ej: Indie, Action, RPG): ")
    game_data['genres'] = [g.strip() for g in genres_input.split(',') if g.strip()]

    # Categorias
    print("\nCategorias disponibles: Single-player, Multi-player, Co-op,")
    print("                        Steam Achievements, Steam Trading Cards,")
    print("                        Steam Cloud, Controller Support, Workshop,")
    print("                        Online Multiplayer, Local Multiplayer,")
    print("                        In-App Purchases")
    cats_input = input("Categorias (separadas por coma): ")
    game_data['categories'] = [c.strip() for c in cats_input.split(',') if c.strip()]

    # Plataformas
    print("\nPlataformas disponibles: Windows, Mac, Linux")
    plats_input = input("Plataformas (separadas por coma, ej: Windows, Mac): ")
    game_data['platforms'] = [p.strip() for p in plats_input.split(',') if p.strip()]

    # Achievements
    while True:
        try:
            achievements = input("\nNumero de logros/achievements (ej: 50): ")
            game_data['achievements'] = int(achievements) if achievements else 0
            break
        except ValueError:
            print("Por favor, introduce un numero entero.")

    # Publisher (opcional)
    publisher = input("\nPublisher (opcional, ej: Valve, Ubisoft, o dejar vacio): ")
    game_data['publisher'] = publisher

    # Año de lanzamiento
    while True:
        try:
            year = input(f"\nAño de lanzamiento (ej: 2024, por defecto {datetime.now().year}): ")
            game_data['release_year'] = int(year) if year else datetime.now().year
            break
        except ValueError:
            print("Por favor, introduce un año valido.")

    # Si el juego ya existe, puede tener ratings
    has_ratings = input("\nEl juego ya tiene valoraciones? (s/n): ").lower().strip()
    if has_ratings == 's':
        try:
            pos = input("  Numero de valoraciones positivas: ")
            game_data['positive_ratings'] = int(pos) if pos else 0
            neg = input("  Numero de valoraciones negativas: ")
            game_data['negative_ratings'] = int(neg) if neg else 0
        except ValueError:
            game_data['positive_ratings'] = 0
            game_data['negative_ratings'] = 0
    else:
        game_data['positive_ratings'] = 0
        game_data['negative_ratings'] = 0

    return game_data


def display_results(game_data, results):
    """
    Muestra los resultados de la prediccion de forma bonita.
    """
    print("\n" + "="*60)
    print("RESULTADOS DE LA PREDICCION")
    print("="*60)

    # Resumen del juego
    print("\n--- Tu juego ---")
    print(f"  Precio: ${game_data.get('price', 0):.2f}")
    print(f"  Generos: {', '.join(game_data.get('genres', ['No especificado']))}")
    print(f"  Categorias: {', '.join(game_data.get('categories', ['No especificado']))}")
    print(f"  Plataformas: {', '.join(game_data.get('platforms', ['No especificado']))}")
    print(f"  Achievements: {game_data.get('achievements', 0)}")
    print(f"  Año lanzamiento: {game_data.get('release_year', 'No especificado')}")
    if game_data.get('publisher'):
        print(f"  Publisher: {game_data.get('publisher')}")

    # Prediccion de exito
    print("\n--- Prediccion de Exito (>=100k owners) ---")
    if 'success' in results:
        if results['success']:
            print(f"  PREDICCION: *** EXITOSO ***")
            print(f"  Probabilidad de exito: {results['success_probability']*100:.1f}%")
        else:
            print(f"  PREDICCION: No exitoso")
            print(f"  Probabilidad de exito: {results['success_probability']*100:.1f}%")

        # Barra visual de probabilidad
        prob = results['success_probability']
        bar_length = 30
        filled = int(prob * bar_length)
        bar = "[" + "#" * filled + "-" * (bar_length - filled) + "]"
        print(f"  {bar} {prob*100:.1f}%")

    # Prediccion de owners
    print("\n--- Prediccion de Numero de Jugadores ---")
    if 'predicted_owners' in results:
        owners = results['predicted_owners']
        print(f"  Owners estimados: {owners:,.0f} jugadores")

        # Clasificacion por rango
        if owners >= 1000000:
            tier = "BLOCKBUSTER (1M+)"
        elif owners >= 500000:
            tier = "MUY EXITOSO (500k-1M)"
        elif owners >= 100000:
            tier = "EXITOSO (100k-500k)"
        elif owners >= 50000:
            tier = "MODERADO (50k-100k)"
        elif owners >= 20000:
            tier = "PEQUEÑO (20k-50k)"
        else:
            tier = "NICHO (<20k)"

        print(f"  Categoria: {tier}")

    # Recomendaciones
    print("\n--- Recomendaciones ---")

    if results.get('success_probability', 0) < 0.3:
        print("  - Considera reducir el precio para aumentar accesibilidad")
        print("  - Añade mas generos populares (Action, Indie)")
        print("  - Implementa Steam Achievements y Trading Cards")
    elif results.get('success_probability', 0) < 0.6:
        print("  - El juego tiene potencial moderado")
        print("  - Considera añadir modo multijugador si es viable")
        print("  - Asegurate de lanzar en multiples plataformas")
    else:
        print("  - El juego tiene muy buen potencial!")
        print("  - Enfocate en marketing y visibilidad")
        print("  - Considera Early Access para construir comunidad")

    print("\n" + "="*60)


def main():
    """
    Funcion principal del predictor.
    """
    print("\n" + "="*60)
    print("CARGANDO MODELOS...")
    print("="*60 + "\n")

    # Cargar modelos
    models = load_models()

    if not models:
        print("ERROR: No se encontraron modelos entrenados.")
        print("Por favor, ejecuta primero: python run_pipeline.py")
        return

    # Obtener nombres de features
    feature_names = get_feature_names()
    if not feature_names:
        print("ERROR: No se encontraron datos procesados.")
        print("Por favor, ejecuta primero: python run_pipeline.py")
        return

    print(f"\nOK - {len(feature_names)} features cargadas")

    while True:
        # Obtener datos del usuario
        game_data = get_user_input()

        # Hacer prediccion
        print("\nAnalizando tu juego...")
        results = predict_game(game_data, models, feature_names)

        # Mostrar resultados
        display_results(game_data, results)

        # Preguntar si quiere probar otro
        another = input("\nQuieres probar con otro juego? (s/n): ").lower().strip()
        if another != 's':
            print("\nGracias por usar el predictor de exito de Steam!")
            break


if __name__ == "__main__":
    main()
