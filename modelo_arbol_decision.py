"""
modelo_arbol_decision.py
========================
Modelo de aprendizaje supervisado basado en Árbol de Decisión (Decision Tree)
para predecir el tiempo real de viaje en el Megabús de Pereira.

Fundamentado en el capítulo 17 del libro:
  Palma Méndez, J. T. (2008). Inteligencia artificial: métodos, técnicas
  y aplicaciones. Madrid: McGraw-Hill España.

El árbol de decisión aprende reglas automáticamente desde los datos,
complementando el sistema basado en reglas manuales de la actividad anterior.

Autor Estudiante 1: Fernando Perez Florez
Autor Estudiante 2: Juan Pablo Ayala
"""

import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor, export_text
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pickle
import os

# ─────────────────────────────────────────────────────────────────────────────
# CARGA Y PREPARACIÓN DE DATOS
# ─────────────────────────────────────────────────────────────────────────────


def cargar_datos(ruta_csv: str = "megabus_viajes.csv") -> pd.DataFrame:
    """Carga el dataset y verifica su estructura."""
    if not os.path.exists(ruta_csv):
        raise FileNotFoundError(
            f"No se encontró '{ruta_csv}'. "
            "Ejecuta primero: python dataset_generator.py"
        )
    df = pd.read_csv(ruta_csv)
    print(f"Dataset cargado: {len(df)} registros, {len(df.columns)} columnas")
    return df


def preparar_features(df: pd.DataFrame):
    """
    Codifica variables categóricas y selecciona features para el modelo.

    Features de entrada (X):
      - paradas          : número de paradas de la ruta
      - tiempo_base_min  : tiempo estimado sin factores externos
      - hora             : hora del día (5-22)
      - hora_pico        : 1 si es hora pico, 0 si no
      - dia_semana       : día codificado numéricamente
      - clima            : clima codificado numéricamente
      - bus_lleno        : 1 si el bus va lleno, 0 si no

    Variable objetivo (y):
      - tiempo_real_min  : tiempo real del viaje en minutos
    """
    df = df.copy()

    # Codificación de variables categóricas con LabelEncoder
    encoders = {}

    for col in ["dia_semana", "clima", "origen", "destino"]:
        le = LabelEncoder()
        df[col + "_cod"] = le.fit_transform(df[col])
        encoders[col] = le

    # Features seleccionadas
    feature_cols = [
        "paradas",
        "tiempo_base_min",
        "hora",
        "hora_pico",
        "dia_semana_cod",
        "clima_cod",
        "bus_lleno",
    ]

    X = df[feature_cols]
    y = df["tiempo_real_min"]

    return X, y, encoders, feature_cols


# ─────────────────────────────────────────────────────────────────────────────
# ENTRENAMIENTO DEL MODELO
# ─────────────────────────────────────────────────────────────────────────────

def entrenar_modelo(X, y, max_depth: int = 6):
    """
    Entrena el árbol de decisión regresor.

    Parámetros:
        X         : features de entrada
        y         : variable objetivo
        max_depth : profundidad máxima del árbol (controla sobreajuste)

    Retorna:
        modelo, X_train, X_test, y_train, y_test
    """
    # División 80% entrenamiento / 20% prueba
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    modelo = DecisionTreeRegressor(
        max_depth=max_depth,
        min_samples_split=10,   # mínimo muestras para dividir un nodo
        min_samples_leaf=5,     # mínimo muestras en hoja
        random_state=42
    )

    modelo.fit(X_train, y_train)
    print(f"Modelo entrenado con {len(X_train)} registros")
    print(f"Profundidad del árbol: {modelo.get_depth()}")
    print(f"Número de hojas: {modelo.get_n_leaves()}")

    return modelo, X_train, X_test, y_train, y_test


# ─────────────────────────────────────────────────────────────────────────────
# EVALUACIÓN DEL MODELO
# ─────────────────────────────────────────────────────────────────────────────

def evaluar_modelo(modelo, X_train, X_test, y_train, y_test, feature_cols):
    """
    Evalúa el modelo con métricas estándar de regresión y muestra resultados.
    """
    y_pred_train = modelo.predict(X_train)
    y_pred_test = modelo.predict(X_test)

    print("\n" + "═" * 55)
    print("  EVALUACIÓN DEL MODELO — ÁRBOL DE DECISIÓN")
    print("═" * 55)

    print(f"\n{'Métrica':<35} {'Entrenamiento':>10} {'Prueba':>10}")
    print("─" * 55)

    mae_train = mean_absolute_error(y_train, y_pred_train)
    mae_test = mean_absolute_error(y_test,  y_pred_test)
    print(
        f"  Error Absoluto Medio (MAE) min   {mae_train:>10.2f} {mae_test:>10.2f}")

    rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
    rmse_test = np.sqrt(mean_squared_error(y_test,  y_pred_test))
    print(
        f"  Error Cuadrático Medio (RMSE)    {rmse_train:>10.2f} {rmse_test:>10.2f}")

    r2_train = r2_score(y_train, y_pred_train)
    r2_test = r2_score(y_test,  y_pred_test)
    print(
        f"  Coeficiente R²                   {r2_train:>10.4f} {r2_test:>10.4f}")

    print("\n  Importancia de features:")
    print("─" * 55)
    importancias = sorted(
        zip(feature_cols, modelo.feature_importances_),
        key=lambda x: x[1], reverse=True
    )
    for feat, imp in importancias:
        barra = "█" * int(imp * 40)
        print(f"  {feat:<25} {imp:.4f}  {barra}")

    return {
        "mae_test": mae_test,
        "rmse_test": rmse_test,
        "r2_test": r2_test,
        "y_pred_test": y_pred_test,
        "y_test": y_test,
    }


# ─────────────────────────────────────────────────────────────────────────────
# GUARDAR Y CARGAR MODELO
# ─────────────────────────────────────────────────────────────────────────────

def guardar_modelo(modelo, encoders, feature_cols, path="modelo_megabus.pkl"):
    """Guarda el modelo entrenado para uso posterior."""
    with open(path, "wb") as f:
        pickle.dump({
            "modelo": modelo,
            "encoders": encoders,
            "feature_cols": feature_cols,
        }, f)
    print(f"\nModelo guardado en '{path}'")


def cargar_modelo(path="modelo_megabus.pkl"):
    """Carga un modelo previamente entrenado."""
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data["modelo"], data["encoders"], data["feature_cols"]


# ─────────────────────────────────────────────────────────────────────────────
# PREDICCIÓN INDIVIDUAL
# ─────────────────────────────────────────────────────────────────────────────

def predecir_viaje(modelo, encoders, feature_cols,
                   paradas, tiempo_base, hora, dia, clima, bus_lleno):
    """
    Predice el tiempo real de un viaje dado sus características.

    Parámetros:
        paradas      : número de paradas de la ruta
        tiempo_base  : tiempo base estimado en minutos
        hora         : hora de salida (5-22)
        dia          : día de la semana en español (ej: 'lunes')
        clima        : 'soleado', 'nublado', 'lluvia', 'tormenta'
        bus_lleno    : 1 si va lleno, 0 si no

    Retorna:
        float — tiempo estimado en minutos
    """
    hora_pico = 1 if (6 <= hora <= 9 or 17 <= hora <= 19) else 0

    dia_cod = encoders["dia_semana"].transform([dia])[0]
    clima_cod = encoders["clima"].transform([clima])[0]

    entrada = pd.DataFrame([{
        "paradas":         paradas,
        "tiempo_base_min": tiempo_base,
        "hora":            hora,
        "hora_pico":       hora_pico,
        "dia_semana_cod":  dia_cod,
        "clima_cod":       clima_cod,
        "bus_lleno":       bus_lleno,
    }])[feature_cols]

    return round(modelo.predict(entrada)[0], 1)


# ─────────────────────────────────────────────────────────────────────────────
# EJECUCIÓN PRINCIPAL
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # 1. Cargar datos
    df = cargar_datos("megabus_viajes.csv")
    print(f"\nResumen del dataset:")
    print(df.describe())

    # 2. Preparar features
    X, y, encoders, feature_cols = preparar_features(df)
    print(f"\nFeatures: {feature_cols}")

    # 3. Entrenar modelo
    modelo, X_train, X_test, y_train, y_test = entrenar_modelo(
        X, y, max_depth=6)

    # 4. Evaluar
    metricas = evaluar_modelo(modelo, X_train, X_test,
                              y_train, y_test, feature_cols)

    # 5. Mostrar reglas del árbol (primeros niveles)
    print("\n" + "═" * 55)
    print("  REGLAS APRENDIDAS POR EL ÁRBOL (primeros 3 niveles)")
    print("═" * 55)
    reglas = export_text(modelo, feature_names=feature_cols, max_depth=3)
    print(reglas)

    # 6. Guardar modelo
    guardar_modelo(modelo, encoders, feature_cols)

    # 7. Ejemplos de predicción
    print("\n" + "═" * 55)
    print("  EJEMPLOS DE PREDICCIÓN")
    print("═" * 55)

    ejemplos = [
        (12, 36, 8,  "lunes",    "lluvia",   1,
         "Cuba → Dosquebradas, lunes 8am, lluvia, lleno"),
        (12, 36, 14, "sabado",   "soleado",  0,
         "Cuba → Dosquebradas, sábado 2pm, sol, vacío"),
        (8,  21, 17, "viernes",  "nublado",  1,
         "Estadio → Terminal, viernes 5pm, nublado, lleno"),
        (8,  21, 10, "domingo",  "soleado",  0,
         "Estadio → Terminal, domingo 10am, sol, vacío"),
        (10, 30, 7,  "miercoles", "tormenta", 1,
         "Cuba → Terminal, miércoles 7am, tormenta, lleno"),
    ]

    print(f"\n  {'Escenario':<50} {'Predicción':>12}")
    print("─" * 65)
    for paradas, t_base, hora, dia, clima, lleno, desc in ejemplos:
        pred = predecir_viaje(modelo, encoders, feature_cols,
                              paradas, t_base, hora, dia, clima, lleno)
        print(f"  {desc:<50} {pred:>8.1f} min")
