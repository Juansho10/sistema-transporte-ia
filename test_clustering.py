"""
tests/test_clustering.py
========================
Pruebas unitarias del modelo de aprendizaje no supervisado (K-Means).

Autor Estudiante 2: [NOMBRE DEL COMPAÑERO]
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import unittest
import numpy as np
import pandas as pd
from dataset_clustering import generar_dataset_clustering
from modelo_clustering import (
    preparar_features, encontrar_k_optimo,
    entrenar_kmeans, analizar_grupos, guardar_modelo
)
import pickle


class TestDatasetClustering(unittest.TestCase):

    def setUp(self):
        self.df = generar_dataset_clustering(200)

    def test_columnas_correctas(self):
        cols = ["origen","destino","paradas","distancia_km","tiempo_base_min",
                "hora","hora_pico","dia_semana","clima","bus_lleno",
                "pasajeros","tiempo_real_min","velocidad_kmh","indice_congestion"]
        for col in cols:
            self.assertIn(col, self.df.columns)

    def test_sin_nulos(self):
        self.assertEqual(self.df.isnull().sum().sum(), 0)

    def test_congestion_entre_0_y_1(self):
        self.assertTrue((self.df["indice_congestion"] >= 0).all())
        self.assertTrue((self.df["indice_congestion"] <= 1).all())

    def test_velocidad_positiva(self):
        self.assertTrue((self.df["velocidad_kmh"] > 0).all())

    def test_pasajeros_en_rango(self):
        self.assertTrue(self.df["pasajeros"].between(15, 80).all())

    def test_tamaño(self):
        self.assertEqual(len(self.df), 200)


class TestModeloClustering(unittest.TestCase):

    def setUp(self):
        df = generar_dataset_clustering(300)
        self.X_scaled, self.X, self.feature_cols, self.scaler, \
            self.encoders, self.df_enc = preparar_features(df)
        self.df = df

    def test_escalado_media_cero(self):
        medias = np.abs(self.X_scaled.mean(axis=0))
        self.assertTrue((medias < 0.1).all(), "Media debe ser ~0 tras escalado")

    def test_kmeans_entrena(self):
        modelo, labels = entrenar_kmeans(self.X_scaled, k=3)
        self.assertIsNotNone(modelo)
        self.assertEqual(len(labels), len(self.X_scaled))

    def test_labels_en_rango(self):
        k = 4
        modelo, labels = entrenar_kmeans(self.X_scaled, k=k)
        self.assertTrue(set(labels).issubset(set(range(k))))

    def test_silhouette_positivo(self):
        from sklearn.metrics import silhouette_score
        modelo, labels = entrenar_kmeans(self.X_scaled, k=3)
        sil = silhouette_score(self.X_scaled, labels)
        self.assertGreater(sil, 0, "Silhouette debe ser positivo")

    def test_n_clusters_correctos(self):
        k = 4
        modelo, labels = entrenar_kmeans(self.X_scaled, k=k)
        self.assertEqual(modelo.n_clusters, k)

    def test_analisis_grupos(self):
        modelo, labels = entrenar_kmeans(self.X_scaled, k=3)
        df_c, resumen = analizar_grupos(self.df, labels, self.feature_cols)
        self.assertIn("cluster", df_c.columns)
        self.assertEqual(len(resumen), 3)

    def test_guardar_y_cargar(self):
        modelo, labels = entrenar_kmeans(self.X_scaled, k=3)
        guardar_modelo(modelo, self.scaler, self.encoders,
                       self.feature_cols, "test_clustering.pkl")
        with open("test_clustering.pkl", "rb") as f:
            data = pickle.load(f)
        self.assertIn("modelo", data)
        self.assertIn("scaler", data)
        os.remove("test_clustering.pkl")

    def test_k_optimo_en_rango(self):
        resultados, k_opt = encontrar_k_optimo(self.X_scaled, k_min=2, k_max=6)
        self.assertGreaterEqual(k_opt, 2)
        self.assertLessEqual(k_opt, 6)

    def test_reproducibilidad(self):
        _, labels1 = entrenar_kmeans(self.X_scaled, k=3)
        _, labels2 = entrenar_kmeans(self.X_scaled, k=3)
        self.assertTrue(np.array_equal(labels1, labels2))


if __name__ == "__main__":
    print("═" * 60)
    print("  PRUEBAS DEL MODELO DE CLUSTERING K-MEANS")
    print("═" * 60)
    unittest.main(verbosity=2)
