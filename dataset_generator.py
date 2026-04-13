"""
dataset_generator.py
====================
Genera el dataset de viajes del Megabús de Pereira.
Como no existen datos públicos oficiales del sistema, se construye un dataset
sintético pero realista basado en:
  - Tiempos base reales de las rutas del sistema anterior
  - Variables que afectan el tiempo real de viaje (hora, clima, día, etc.)
  - Ruido aleatorio controlado para simular variabilidad real

El dataset resultante (megabus_viajes.csv) sirve como fuente de datos
para entrenar el modelo de aprendizaje supervisado (árbol de decisión).

Autor Estudiante 1: [TU NOMBRE]
"""

import pandas as pd
import numpy as np
import random

random.seed(42)
np.random.seed(42)

# ─────────────────────────────────────────────────────────────────────────────
# DATOS BASE DEL SISTEMA (heredados del proyecto anterior)
# ─────────────────────────────────────────────────────────────────────────────

RUTAS = [
    # (origen, destino, paradas, tiempo_base_minutos)
    ("Cuba",            "Dosquebradas",    12, 36),
    ("Cuba",            "Terminal",        10, 30),
    ("Cuba",            "Centro",          9,  27),
    ("Villa Santana",   "Nacederos",       11, 33),
    ("Estadio",         "Terminal",        8,  21),
    ("Galicia",         "Boston",          10, 28),
    ("Kennedy",         "Dosquebradas",    10, 30),
    ("Centro",          "Cuba",            9,  27),
    ("Terminal",        "Cuba",            10, 30),
    ("Nacederos",       "Galicia",         9,  26),
    ("Boston",          "Kennedy",         8,  23),
    ("Río Otún",        "El Plumón",       8,  22),
    ("Simón Bolívar",   "Villa Santana",   7,  20),
    ("Av. 30 de Agosto","Dosquebradas",    9,  25),
    ("La Roma",         "Boston",          11, 32),
]

HORAS = list(range(5, 23))  # 5am a 10pm

HORA_FACTOR = {
    5:  0.85, 6:  1.30, 7:  1.65, 8:  1.70, 9:  1.40,
    10: 1.10, 11: 1.05, 12: 1.20, 13: 1.25, 14: 1.10,
    15: 1.10, 16: 1.30, 17: 1.70, 18: 1.75, 19: 1.40,
    20: 1.15, 21: 1.00, 22: 0.90,
}

CLIMA_FACTOR = {
    "soleado": 1.00,
    "nublado": 1.05,
    "lluvia":  1.25,
    "tormenta":1.40,
}

DIA_FACTOR = {
    "lunes":    1.10,
    "martes":   1.05,
    "miercoles":1.05,
    "jueves":   1.08,
    "viernes":  1.20,
    "sabado":   0.90,
    "domingo":  0.80,
}

DIAS = list(DIA_FACTOR.keys())
CLIMAS = list(CLIMA_FACTOR.keys())
CLIMA_PROB = [0.50, 0.25, 0.20, 0.05]  # Pereira: mucho sol y algo de lluvia


def generar_dataset(n_registros: int = 1000) -> pd.DataFrame:
    """
    Genera n_registros filas de viajes simulados con sus variables y
    el tiempo real resultante (variable objetivo).
    """
    registros = []

    for _ in range(n_registros):
        ruta = random.choice(RUTAS)
        origen, destino, paradas, tiempo_base = ruta

        hora     = random.choice(HORAS)
        dia      = random.choice(DIAS)
        clima    = random.choices(CLIMAS, weights=CLIMA_PROB)[0]
        pasajeros_llenos = random.randint(0, 1)  # 1 = bus lleno

        # Calcular tiempo real con factores
        t = tiempo_base
        t *= HORA_FACTOR[hora]
        t *= CLIMA_FACTOR[clima]
        t *= DIA_FACTOR[dia]
        if pasajeros_llenos:
            t *= 1.10  # bus lleno demora más en paradas
        # Ruido aleatorio ±10%
        t *= random.uniform(0.90, 1.10)
        tiempo_real = round(t, 1)

        # Hora pico: 6-9am y 5-7pm
        hora_pico = 1 if (6 <= hora <= 9 or 17 <= hora <= 19) else 0

        registros.append({
            "origen":           origen,
            "destino":          destino,
            "paradas":          paradas,
            "tiempo_base_min":  tiempo_base,
            "hora":             hora,
            "hora_pico":        hora_pico,
            "dia_semana":       dia,
            "clima":            clima,
            "bus_lleno":        pasajeros_llenos,
            "tiempo_real_min":  tiempo_real,  # VARIABLE OBJETIVO
        })

    return pd.DataFrame(registros)


if __name__ == "__main__":
    df = generar_dataset(1000)
    df.to_csv("megabus_viajes.csv", index=False)
    print(f"Dataset generado: {len(df)} registros")
    print(f"Columnas: {list(df.columns)}")
    print(f"\nPrimeras filas:")
    print(df.head())
    print(f"\nEstadísticas del tiempo real:")
    print(df["tiempo_real_min"].describe())
