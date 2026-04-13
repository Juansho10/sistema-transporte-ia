"""
dataset_clustering.py
=====================
Genera el dataset de viajes del Megabús de Pereira para el modelo
de aprendizaje no supervisado (clustering / agrupamiento).

Reutiliza la lógica del dataset anterior pero agrega variables
adicionales útiles para el agrupamiento:
  - velocidad_promedio_kmh
  - indice_congestion (0-1)
  - costo_pasaje

Como no existen datos públicos oficiales del Megabús de Pereira,
se construye un dataset sintético realista basado en las características
reales del sistema.

Referencia: Palma Méndez, J. T. (2008). Cap. 16 — Técnicas de agrupamiento.

Autor Estudiante 1: [TU NOMBRE]
"""

import pandas as pd
import numpy as np
import random

random.seed(42)
np.random.seed(42)

RUTAS = [
    ("Cuba",            "Dosquebradas",    12, 36, 18.5),
    ("Cuba",            "Terminal",        10, 30, 15.0),
    ("Cuba",            "Centro",           9, 27, 13.5),
    ("Villa Santana",   "Nacederos",       11, 33, 16.5),
    ("Estadio",         "Terminal",         8, 21, 10.5),
    ("Galicia",         "Boston",          10, 28, 14.0),
    ("Kennedy",         "Dosquebradas",    10, 30, 15.0),
    ("Centro",          "Cuba",             9, 27, 13.5),
    ("Terminal",        "Cuba",            10, 30, 15.0),
    ("Nacederos",       "Galicia",          9, 26, 13.0),
    ("Boston",          "Kennedy",          8, 23, 11.5),
    ("Río Otún",        "El Plumón",        8, 22, 11.0),
    ("Simón Bolívar",   "Villa Santana",    7, 20,  9.5),
    ("Av. 30 de Agosto","Dosquebradas",     9, 25, 12.5),
    ("La Roma",         "Boston",          11, 32, 16.0),
]

HORA_FACTOR    = {5:0.85,6:1.30,7:1.65,8:1.70,9:1.40,10:1.10,11:1.05,
                  12:1.20,13:1.25,14:1.10,15:1.10,16:1.30,17:1.70,
                  18:1.75,19:1.40,20:1.15,21:1.00,22:0.90}
CLIMA_FACTOR   = {"soleado":1.00,"nublado":1.05,"lluvia":1.25,"tormenta":1.40}
DIA_FACTOR     = {"lunes":1.10,"martes":1.05,"miercoles":1.05,"jueves":1.08,
                  "viernes":1.20,"sabado":0.90,"domingo":0.80}
DIAS           = list(DIA_FACTOR.keys())
CLIMAS         = list(CLIMA_FACTOR.keys())
CLIMA_PROB     = [0.50, 0.25, 0.20, 0.05]


def generar_dataset_clustering(n: int = 1200) -> pd.DataFrame:
    registros = []
    for _ in range(n):
        ruta = random.choice(RUTAS)
        origen, destino, paradas, t_base, distancia_km = ruta

        hora    = random.choice(list(range(5, 23)))
        dia     = random.choice(DIAS)
        clima   = random.choices(CLIMAS, weights=CLIMA_PROB)[0]
        bus_lleno = random.randint(0, 1)
        pasajeros = random.randint(15, 80)

        # Tiempo real
        t = t_base * HORA_FACTOR[hora] * CLIMA_FACTOR[clima] * DIA_FACTOR[dia]
        if bus_lleno:
            t *= 1.10
        t *= random.uniform(0.90, 1.10)
        tiempo_real = round(t, 1)

        # Variables derivadas
        velocidad = round((distancia_km / (tiempo_real / 60)), 1)  # km/h
        hora_pico = 1 if (6 <= hora <= 9 or 17 <= hora <= 19) else 0

        # Índice de congestión (0=fluido, 1=muy congestionado)
        congestion = round(
            (HORA_FACTOR[hora] - 0.85) / (1.75 - 0.85) *
            (CLIMA_FACTOR[clima]) *
            (1.1 if bus_lleno else 1.0) *
            random.uniform(0.85, 1.15), 3
        )
        congestion = min(max(congestion, 0), 1)

        registros.append({
            "origen":             origen,
            "destino":            destino,
            "paradas":            paradas,
            "distancia_km":       distancia_km,
            "tiempo_base_min":    t_base,
            "hora":               hora,
            "hora_pico":          hora_pico,
            "dia_semana":         dia,
            "clima":              clima,
            "bus_lleno":          bus_lleno,
            "pasajeros":          pasajeros,
            "tiempo_real_min":    tiempo_real,
            "velocidad_kmh":      velocidad,
            "indice_congestion":  congestion,
        })

    return pd.DataFrame(registros)


if __name__ == "__main__":
    df = generar_dataset_clustering(1200)
    df.to_csv("megabus_clustering.csv", index=False)
    print(f"Dataset generado: {len(df)} registros")
    print(f"Columnas: {list(df.columns)}")
    print(f"\nPrimeras filas:")
    print(df.head())
    print(f"\nEstadísticas:")
    print(df[["tiempo_real_min","velocidad_kmh","indice_congestion","pasajeros"]].describe())
