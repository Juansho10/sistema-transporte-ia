"""
visualizaciones_clustering.py
==============================
Genera gráficas del modelo de clustering K-Means.

Autor Estudiante 2: [NOMBRE DEL COMPAÑERO]
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import os, sys

sys.path.insert(0, os.path.dirname(__file__))
from dataset_clustering import generar_dataset_clustering
from modelo_clustering import (
    preparar_features, encontrar_k_optimo,
    entrenar_kmeans, analizar_grupos, reducir_pca
)

os.makedirs("graficas_clustering", exist_ok=True)

COLORES = ["#2196F3", "#FF5722", "#4CAF50", "#9C27B0", "#FF9800", "#00BCD4"]


def grafica_codo(resultados_k):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Método del Codo — Selección del K óptimo", fontsize=14, fontweight="bold")

    axes[0].plot(resultados_k["k"], resultados_k["inercia"],
                 marker="o", color=COLORES[0], linewidth=2.5)
    axes[0].set_xlabel("Número de clusters (K)")
    axes[0].set_ylabel("Inercia (WCSS)")
    axes[0].set_title("Inercia por K")
    axes[0].grid(alpha=0.3)

    axes[1].plot(resultados_k["k"], resultados_k["silhouette"],
                 marker="s", color=COLORES[2], linewidth=2.5)
    idx_max = resultados_k["silhouette"].index(max(resultados_k["silhouette"]))
    k_opt   = resultados_k["k"][idx_max]
    axes[1].axvline(k_opt, color="red", linestyle="--",
                    label=f"K óptimo = {k_opt}")
    axes[1].set_xlabel("Número de clusters (K)")
    axes[1].set_ylabel("Silhouette Score")
    axes[1].set_title("Silhouette Score por K")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("graficas_clustering/1_metodo_codo.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  ✓ graficas_clustering/1_metodo_codo.png")


def grafica_clusters_pca(X_scaled, labels, k):
    X_pca = reducir_pca(X_scaled)
    fig, ax = plt.subplots(figsize=(9, 7))
    for i in range(k):
        mask = labels == i
        ax.scatter(X_pca[mask, 0], X_pca[mask, 1],
                   c=COLORES[i], label=f"Grupo {i}",
                   alpha=0.6, s=25, edgecolors="white", linewidth=0.3)
    ax.set_xlabel("Componente Principal 1")
    ax.set_ylabel("Componente Principal 2")
    ax.set_title("Clusters de Viajes — Visualización PCA 2D", fontweight="bold")
    ax.legend()
    ax.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig("graficas_clustering/2_clusters_pca.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  ✓ graficas_clustering/2_clusters_pca.png")


def grafica_perfiles_grupos(resumen):
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    fig.suptitle("Perfil de cada Grupo — K-Means", fontsize=14, fontweight="bold")

    grupos = resumen.index.tolist()
    colores_g = [COLORES[i] for i in range(len(grupos))]
    etiquetas = [f"G{g}" for g in grupos]

    # Tiempo promedio
    axes[0,0].bar(etiquetas, resumen["tiempo_promedio"], color=colores_g, edgecolor="white")
    axes[0,0].set_title("Tiempo promedio (min)")
    axes[0,0].set_ylabel("Minutos")
    for i, v in enumerate(resumen["tiempo_promedio"]):
        axes[0,0].text(i, v+0.5, f"{v:.1f}", ha="center", fontsize=9)

    # Velocidad promedio
    axes[0,1].bar(etiquetas, resumen["velocidad_prom"], color=colores_g, edgecolor="white")
    axes[0,1].set_title("Velocidad promedio (km/h)")
    axes[0,1].set_ylabel("km/h")
    for i, v in enumerate(resumen["velocidad_prom"]):
        axes[0,1].text(i, v+0.2, f"{v:.1f}", ha="center", fontsize=9)

    # Congestión promedio
    axes[1,0].bar(etiquetas, resumen["congestion_prom"], color=colores_g, edgecolor="white")
    axes[1,0].set_title("Índice de congestión promedio")
    axes[1,0].set_ylabel("Índice (0-1)")
    axes[1,0].set_ylim(0, 1)
    for i, v in enumerate(resumen["congestion_prom"]):
        axes[1,0].text(i, v+0.02, f"{v:.3f}", ha="center", fontsize=9)

    # % hora pico
    axes[1,1].bar(etiquetas, resumen["pct_hora_pico"]*100, color=colores_g, edgecolor="white")
    axes[1,1].set_title("% de viajes en hora pico")
    axes[1,1].set_ylabel("Porcentaje (%)")
    for i, v in enumerate(resumen["pct_hora_pico"]*100):
        axes[1,1].text(i, v+0.5, f"{v:.1f}%", ha="center", fontsize=9)

    plt.tight_layout()
    plt.savefig("graficas_clustering/3_perfiles_grupos.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  ✓ graficas_clustering/3_perfiles_grupos.png")


def grafica_distribucion_clusters(labels, k):
    conteos = [np.sum(labels == i) for i in range(k)]
    fig, ax = plt.subplots(figsize=(8, 6))
    wedges, texts, autotexts = ax.pie(
        conteos,
        labels=[f"Grupo {i}\n({c} viajes)" for i, c in enumerate(conteos)],
        colors=COLORES[:k],
        autopct="%1.1f%%",
        startangle=90,
        wedgeprops=dict(edgecolor="white", linewidth=2)
    )
    ax.set_title("Distribución de viajes por cluster", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig("graficas_clustering/4_distribucion_clusters.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  ✓ graficas_clustering/4_distribucion_clusters.png")


def grafica_tiempo_vs_congestion(df_clusterizado, k):
    fig, ax = plt.subplots(figsize=(10, 7))
    for i in range(k):
        mask = df_clusterizado["cluster"] == i
        ax.scatter(
            df_clusterizado[mask]["indice_congestion"],
            df_clusterizado[mask]["tiempo_real_min"],
            c=COLORES[i], label=f"Grupo {i}",
            alpha=0.5, s=20, edgecolors="white", linewidth=0.2
        )
    ax.set_xlabel("Índice de congestión")
    ax.set_ylabel("Tiempo real (minutos)")
    ax.set_title("Tiempo real vs Congestión por grupo", fontweight="bold")
    ax.legend()
    ax.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig("graficas_clustering/5_tiempo_vs_congestion.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  ✓ graficas_clustering/5_tiempo_vs_congestion.png")


if __name__ == "__main__":
    print("Generando dataset...")
    df = generar_dataset_clustering(1200)
    df.to_csv("megabus_clustering.csv", index=False)

    print("Preparando features y entrenando modelo...")
    X_scaled, X, feature_cols, scaler, encoders, df_enc = preparar_features(df)
    resultados_k, k_optimo = encontrar_k_optimo(X_scaled, k_min=2, k_max=8)
    modelo, labels = entrenar_kmeans(X_scaled, k=k_optimo)
    df_clusterizado, resumen = analizar_grupos(df, labels, feature_cols)

    print("\nGenerando gráficas...")
    grafica_codo(resultados_k)
    grafica_clusters_pca(X_scaled, labels, k_optimo)
    grafica_perfiles_grupos(resumen)
    grafica_distribucion_clusters(labels, k_optimo)
    grafica_tiempo_vs_congestion(df_clusterizado, k_optimo)

    print("\n✅ Todas las gráficas en 'graficas_clustering/'")
