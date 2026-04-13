"""
visualizaciones.py
==================
Genera gráficas del dataset y del modelo de árbol de decisión.
Produce imágenes PNG para incluir en el documento de pruebas.

Autor Estudiante 2: [NOMBRE DEL COMPAÑERO]
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")  # Para entornos sin pantalla
from sklearn.tree import plot_tree
import os, sys

sys.path.insert(0, os.path.dirname(__file__))
from dataset_generator import generar_dataset
from modelo_arbol_decision import (
    cargar_datos, preparar_features, entrenar_modelo, evaluar_modelo
)

os.makedirs("graficas", exist_ok=True)

COLORES = ["#2196F3", "#FF5722", "#4CAF50", "#9C27B0", "#FF9800"]


def grafica_distribucion_tiempos(df):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Distribución de Tiempos de Viaje — Megabús Pereira", fontsize=14, fontweight="bold")

    axes[0].hist(df["tiempo_real_min"], bins=30, color=COLORES[0], edgecolor="white", alpha=0.85)
    axes[0].set_xlabel("Tiempo real (minutos)")
    axes[0].set_ylabel("Frecuencia")
    axes[0].set_title("Distribución general")
    axes[0].axvline(df["tiempo_real_min"].mean(), color="red", linestyle="--", label=f"Media: {df['tiempo_real_min'].mean():.1f} min")
    axes[0].legend()

    clima_means = df.groupby("clima")["tiempo_real_min"].mean().sort_values(ascending=False)
    axes[1].bar(clima_means.index, clima_means.values, color=COLORES[:4], edgecolor="white")
    axes[1].set_xlabel("Clima")
    axes[1].set_ylabel("Tiempo promedio (minutos)")
    axes[1].set_title("Tiempo promedio por clima")
    for i, v in enumerate(clima_means.values):
        axes[1].text(i, v + 0.3, f"{v:.1f}", ha="center", fontsize=10)

    plt.tight_layout()
    plt.savefig("graficas/1_distribucion_tiempos.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  ✓ graficas/1_distribucion_tiempos.png")


def grafica_hora_vs_tiempo(df):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Impacto de la Hora en el Tiempo de Viaje", fontsize=14, fontweight="bold")

    hora_means = df.groupby("hora")["tiempo_real_min"].mean()
    axes[0].plot(hora_means.index, hora_means.values, color=COLORES[0], linewidth=2.5, marker="o", markersize=5)
    axes[0].fill_between(hora_means.index, hora_means.values, alpha=0.15, color=COLORES[0])
    axes[0].axvspan(6, 9, alpha=0.12, color="red", label="Hora pico mañana")
    axes[0].axvspan(17, 19, alpha=0.12, color="orange", label="Hora pico tarde")
    axes[0].set_xlabel("Hora del día")
    axes[0].set_ylabel("Tiempo promedio (minutos)")
    axes[0].set_title("Tiempo promedio por hora")
    axes[0].legend()
    axes[0].set_xticks(range(5, 23))

    dia_order = ["lunes","martes","miercoles","jueves","viernes","sabado","domingo"]
    dia_means = df.groupby("dia_semana")["tiempo_real_min"].mean().reindex(dia_order)
    bars = axes[1].bar(dia_means.index, dia_means.values, color=COLORES, edgecolor="white")
    axes[1].set_xlabel("Día de la semana")
    axes[1].set_ylabel("Tiempo promedio (minutos)")
    axes[1].set_title("Tiempo promedio por día")
    axes[1].tick_params(axis="x", rotation=30)
    for bar, v in zip(bars, dia_means.values):
        axes[1].text(bar.get_x() + bar.get_width()/2, v + 0.2, f"{v:.1f}", ha="center", fontsize=9)

    plt.tight_layout()
    plt.savefig("graficas/2_hora_dia_vs_tiempo.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  ✓ graficas/2_hora_dia_vs_tiempo.png")


def grafica_prediccion_vs_real(y_test, y_pred_test):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Evaluación del Modelo — Predicción vs Valor Real", fontsize=14, fontweight="bold")

    y_test_arr = np.array(y_test)
    axes[0].scatter(y_test_arr, y_pred_test, alpha=0.4, color=COLORES[0], s=20)
    lims = [min(y_test_arr.min(), y_pred_test.min()) - 2,
            max(y_test_arr.max(), y_pred_test.max()) + 2]
    axes[0].plot(lims, lims, "r--", linewidth=1.5, label="Predicción perfecta")
    axes[0].set_xlabel("Tiempo real (minutos)")
    axes[0].set_ylabel("Tiempo predicho (minutos)")
    axes[0].set_title("Predicho vs Real")
    axes[0].legend()

    errores = y_pred_test - y_test_arr
    axes[1].hist(errores, bins=30, color=COLORES[2], edgecolor="white", alpha=0.85)
    axes[1].axvline(0, color="red", linestyle="--", linewidth=1.5)
    axes[1].set_xlabel("Error (predicho − real) en minutos")
    axes[1].set_ylabel("Frecuencia")
    axes[1].set_title("Distribución del error")
    axes[1].text(0.98, 0.95, f"MAE: {np.abs(errores).mean():.2f} min",
                 transform=axes[1].transAxes, ha="right", va="top",
                 bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

    plt.tight_layout()
    plt.savefig("graficas/3_prediccion_vs_real.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  ✓ graficas/3_prediccion_vs_real.png")


def grafica_importancia_features(modelo, feature_cols):
    fig, ax = plt.subplots(figsize=(9, 5))
    importancias = pd.Series(modelo.feature_importances_, index=feature_cols).sort_values()
    labels = {
        "paradas": "Número de paradas",
        "tiempo_base_min": "Tiempo base (min)",
        "hora": "Hora del día",
        "hora_pico": "¿Hora pico?",
        "dia_semana_cod": "Día de la semana",
        "clima_cod": "Clima",
        "bus_lleno": "Bus lleno",
    }
    importancias.index = [labels.get(i, i) for i in importancias.index]
    bars = importancias.plot(kind="barh", ax=ax, color=COLORES[0], edgecolor="white")
    ax.set_xlabel("Importancia (Gini)")
    ax.set_title("Importancia de Variables — Árbol de Decisión", fontweight="bold")
    for i, v in enumerate(importancias.values):
        ax.text(v + 0.002, i, f"{v:.4f}", va="center", fontsize=9)
    plt.tight_layout()
    plt.savefig("graficas/4_importancia_features.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  ✓ graficas/4_importancia_features.png")


def grafica_arbol(modelo, feature_cols):
    fig, ax = plt.subplots(figsize=(20, 8))
    labels = {
        "paradas": "Paradas",
        "tiempo_base_min": "T.base",
        "hora": "Hora",
        "hora_pico": "H.pico",
        "dia_semana_cod": "Dia",
        "clima_cod": "Clima",
        "bus_lleno": "Lleno",
    }
    feat_names = [labels.get(f, f) for f in feature_cols]
    plot_tree(modelo, feature_names=feat_names, filled=True, rounded=True,
              max_depth=3, fontsize=8, ax=ax,
              impurity=False, precision=1)
    ax.set_title("Árbol de Decisión (primeros 3 niveles) — Predicción Tiempo de Viaje",
                 fontsize=13, fontweight="bold", pad=15)
    plt.tight_layout()
    plt.savefig("graficas/5_arbol_decision.png", dpi=120, bbox_inches="tight")
    plt.close()
    print("  ✓ graficas/5_arbol_decision.png")


if __name__ == "__main__":
    print("Generando dataset...")
    df = generar_dataset(1000)
    df.to_csv("megabus_viajes.csv", index=False)

    print("Entrenando modelo...")
    X, y, encoders, feature_cols = preparar_features(df)
    modelo, X_train, X_test, y_train, y_test = entrenar_modelo(X, y)
    metricas = evaluar_modelo(modelo, X_train, X_test, y_train, y_test, feature_cols)

    print("\nGenerando gráficas...")
    grafica_distribucion_tiempos(df)
    grafica_hora_vs_tiempo(df)
    grafica_prediccion_vs_real(metricas["y_test"], metricas["y_pred_test"])
    grafica_importancia_features(modelo, feature_cols)
    grafica_arbol(modelo, feature_cols)

    print("\n✅ Todas las gráficas generadas en la carpeta 'graficas/'")
