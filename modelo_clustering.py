"""
modelo_clustering.py
====================
Modelo de aprendizaje NO supervisado basado en K-Means para agrupar
los viajes del Megabús de Pereira en grupos con patrones similares.

El algoritmo descubre por sí solo patrones en los datos sin necesidad
de etiquetas previas, identificando grupos como:
  - Viajes rápidos en horas valle
  - Viajes congestionados en hora pico
  - Viajes con condiciones climáticas adversas
  - etc.

Referencia: Palma Méndez, J. T. (2008). Cap. 16 — Técnicas de agrupamiento.

Autor Estudiante 1: [TU NOMBRE]
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.decomposition import PCA
import pickle
import os


# ─────────────────────────────────────────────────────────────────────────────
# CARGA Y PREPARACIÓN
# ─────────────────────────────────────────────────────────────────────────────

def cargar_datos(ruta_csv: str = "megabus_clustering.csv") -> pd.DataFrame:
    if not os.path.exists(ruta_csv):
        raise FileNotFoundError(
            f"No se encontró '{ruta_csv}'. "
            "Ejecuta primero: python3 dataset_clustering.py"
        )
    df = pd.read_csv(ruta_csv)
    print(f"Dataset cargado: {len(df)} registros, {len(df.columns)} columnas")
    return df


def preparar_features(df: pd.DataFrame):
    """
    Selecciona y escala las variables para el clustering.
    K-Means requiere escalado porque es sensible a la magnitud de las variables.
    """
    df = df.copy()

    # Codificar variables categóricas
    encoders = {}
    for col in ["dia_semana", "clima"]:
        le = LabelEncoder()
        df[col + "_cod"] = le.fit_transform(df[col])
        encoders[col] = le

    # Features para clustering
    feature_cols = [
        "tiempo_real_min",
        "velocidad_kmh",
        "indice_congestion",
        "hora",
        "hora_pico",
        "pasajeros",
        "paradas",
        "clima_cod",
        "dia_semana_cod",
        "bus_lleno",
    ]

    X = df[feature_cols].copy()

    # Escalado estándar (media=0, desv=1) — obligatorio para K-Means
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, X, feature_cols, scaler, encoders, df


# ─────────────────────────────────────────────────────────────────────────────
# MÉTODO DEL CODO — encontrar K óptimo
# ─────────────────────────────────────────────────────────────────────────────

def encontrar_k_optimo(X_scaled, k_min: int = 2, k_max: int = 10) -> dict:
    """
    Calcula inercia y silhouette score para distintos valores de K.
    El K óptimo es donde la inercia deja de bajar bruscamente (codo)
    y el silhouette score es más alto.
    """
    resultados = {"k": [], "inercia": [], "silhouette": [], "davies_bouldin": []}

    print("\nBuscando K óptimo...")
    print(f"{'K':>4} {'Inercia':>12} {'Silhouette':>12} {'Davies-Bouldin':>16}")
    print("─" * 48)

    for k in range(k_min, k_max + 1):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_scaled)

        inercia   = km.inertia_
        sil       = silhouette_score(X_scaled, labels)
        db        = davies_bouldin_score(X_scaled, labels)

        resultados["k"].append(k)
        resultados["inercia"].append(round(inercia, 2))
        resultados["silhouette"].append(round(sil, 4))
        resultados["davies_bouldin"].append(round(db, 4))

        print(f"{k:>4} {inercia:>12.2f} {sil:>12.4f} {db:>16.4f}")

    # K óptimo = mayor silhouette
    idx_optimo = resultados["silhouette"].index(max(resultados["silhouette"]))
    k_optimo   = resultados["k"][idx_optimo]
    print(f"\n→ K óptimo recomendado: {k_optimo} "
          f"(Silhouette: {max(resultados['silhouette'])})")

    return resultados, k_optimo


# ─────────────────────────────────────────────────────────────────────────────
# ENTRENAMIENTO K-MEANS
# ─────────────────────────────────────────────────────────────────────────────

def entrenar_kmeans(X_scaled, k: int = 4):
    """
    Entrena el modelo K-Means con K grupos.
    """
    modelo = KMeans(n_clusters=k, random_state=42, n_init=10, max_iter=300)
    labels = modelo.fit_predict(X_scaled)

    sil = silhouette_score(X_scaled, labels)
    db  = davies_bouldin_score(X_scaled, labels)

    print(f"\nModelo K-Means entrenado con K={k}")
    print(f"  Inercia          : {modelo.inertia_:.2f}")
    print(f"  Silhouette Score : {sil:.4f}  (más alto = mejor, máx=1)")
    print(f"  Davies-Bouldin   : {db:.4f}   (más bajo = mejor)")

    return modelo, labels


# ─────────────────────────────────────────────────────────────────────────────
# ANÁLISIS DE GRUPOS
# ─────────────────────────────────────────────────────────────────────────────

def analizar_grupos(df_original: pd.DataFrame, labels: np.ndarray,
                    feature_cols: list) -> pd.DataFrame:
    """
    Calcula estadísticas por grupo y les asigna un nombre descriptivo.
    """
    df = df_original.copy()
    df["cluster"] = labels

    print("\n" + "═" * 65)
    print("  ANÁLISIS DE GRUPOS ENCONTRADOS")
    print("═" * 65)

    resumen = df.groupby("cluster").agg(
        n_viajes         =("cluster",           "count"),
        tiempo_promedio  =("tiempo_real_min",    "mean"),
        velocidad_prom   =("velocidad_kmh",      "mean"),
        congestion_prom  =("indice_congestion",  "mean"),
        hora_promedio    =("hora",               "mean"),
        pasajeros_prom   =("pasajeros",          "mean"),
        pct_hora_pico    =("hora_pico",          "mean"),
        pct_bus_lleno    =("bus_lleno",          "mean"),
    ).round(2)

    # Nombres descriptivos automáticos basados en características
    nombres = {}
    for idx, row in resumen.iterrows():
        if row["congestion_prom"] > 0.55:
            nombres[idx] = "🔴 Alta congestión"
        elif row["tiempo_promedio"] < 28:
            nombres[idx] = "🟢 Viaje rápido"
        elif row["pct_hora_pico"] > 0.55:
            nombres[idx] = "🟡 Hora pico"
        else:
            nombres[idx] = "🔵 Condición normal"

    resumen["nombre_grupo"] = [nombres[i] for i in resumen.index]

    for idx, row in resumen.iterrows():
        print(f"\n  Grupo {idx} — {row['nombre_grupo']}")
        print(f"    Viajes          : {int(row['n_viajes'])}")
        print(f"    Tiempo promedio : {row['tiempo_promedio']} min")
        print(f"    Velocidad prom  : {row['velocidad_prom']} km/h")
        print(f"    Congestión prom : {row['congestion_prom']}")
        print(f"    % en hora pico  : {row['pct_hora_pico']*100:.1f}%")
        print(f"    % bus lleno     : {row['pct_bus_lleno']*100:.1f}%")

    return df, resumen


# ─────────────────────────────────────────────────────────────────────────────
# PCA PARA VISUALIZACIÓN
# ─────────────────────────────────────────────────────────────────────────────

def reducir_pca(X_scaled) -> np.ndarray:
    """Reduce a 2 dimensiones para poder graficar los clusters."""
    pca = PCA(n_components=2, random_state=42)
    return pca.fit_transform(X_scaled)


# ─────────────────────────────────────────────────────────────────────────────
# GUARDAR MODELO
# ─────────────────────────────────────────────────────────────────────────────

def guardar_modelo(modelo, scaler, encoders, feature_cols,
                   path="modelo_clustering.pkl"):
    with open(path, "wb") as f:
        pickle.dump({
            "modelo": modelo,
            "scaler": scaler,
            "encoders": encoders,
            "feature_cols": feature_cols,
        }, f)
    print(f"\nModelo guardado en '{path}'")


# ─────────────────────────────────────────────────────────────────────────────
# EJECUCIÓN PRINCIPAL
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # 1. Cargar datos
    df = cargar_datos("megabus_clustering.csv")

    # 2. Preparar features
    X_scaled, X, feature_cols, scaler, encoders, df_enc = preparar_features(df)
    print(f"Features usadas: {feature_cols}")

    # 3. Encontrar K óptimo
    resultados_k, k_optimo = encontrar_k_optimo(X_scaled, k_min=2, k_max=8)

    # 4. Entrenar con K óptimo
    modelo, labels = entrenar_kmeans(X_scaled, k=k_optimo)

    # 5. Analizar grupos
    df_clusterizado, resumen = analizar_grupos(df, labels, feature_cols)
    df_clusterizado.to_csv("megabus_clusterizado.csv", index=False)
    print(f"\nDataset con clusters guardado en 'megabus_clusterizado.csv'")

    # 6. Guardar modelo
    guardar_modelo(modelo, scaler, encoders, feature_cols)

    print("\n✅ Proceso completado exitosamente")
