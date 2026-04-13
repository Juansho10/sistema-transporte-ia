"""
tests/test_modelo.py
====================
Pruebas unitarias del modelo de aprendizaje automático.

Autor Estudiante 2: [NOMBRE DEL COMPAÑERO]
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import unittest
import pandas as pd
import numpy as np
from dataset_generator import generar_dataset
from modelo_arbol_decision import (
    preparar_features, entrenar_modelo, evaluar_modelo,
    predecir_viaje, guardar_modelo, cargar_modelo
)


class TestDataset(unittest.TestCase):

    def setUp(self):
        self.df = generar_dataset(200)

    def test_columnas_correctas(self):
        cols = ["origen","destino","paradas","tiempo_base_min","hora",
                "hora_pico","dia_semana","clima","bus_lleno","tiempo_real_min"]
        for col in cols:
            self.assertIn(col, self.df.columns)

    def test_sin_valores_nulos(self):
        self.assertEqual(self.df.isnull().sum().sum(), 0)

    def test_tiempo_real_mayor_que_cero(self):
        self.assertTrue((self.df["tiempo_real_min"] > 0).all())

    def test_hora_en_rango(self):
        self.assertTrue(self.df["hora"].between(5, 22).all())

    def test_hora_pico_binario(self):
        self.assertTrue(self.df["hora_pico"].isin([0, 1]).all())

    def test_tamaño_dataset(self):
        self.assertEqual(len(self.df), 200)


class TestModelo(unittest.TestCase):

    def setUp(self):
        df = generar_dataset(500)
        X, y, encoders, feature_cols = preparar_features(df)
        self.modelo, self.X_train, self.X_test, self.y_train, self.y_test = \
            entrenar_modelo(X, y)
        self.encoders = encoders
        self.feature_cols = feature_cols
        self.X = X
        self.y = y

    def test_modelo_entrena(self):
        self.assertIsNotNone(self.modelo)

    def test_r2_positivo(self):
        from sklearn.metrics import r2_score
        y_pred = self.modelo.predict(self.X_test)
        r2 = r2_score(self.y_test, y_pred)
        self.assertGreater(r2, 0.5, "R² debe ser mayor a 0.5")

    def test_mae_razonable(self):
        from sklearn.metrics import mean_absolute_error
        y_pred = self.modelo.predict(self.X_test)
        mae = mean_absolute_error(self.y_test, y_pred)
        self.assertLess(mae, 10, "MAE debe ser menor a 10 minutos")

    def test_prediccion_individual(self):
        pred = predecir_viaje(
            self.modelo, self.encoders, self.feature_cols,
            paradas=12, tiempo_base=36, hora=8,
            dia="lunes", clima="lluvia", bus_lleno=1
        )
        self.assertIsInstance(pred, float)
        self.assertGreater(pred, 0)

    def test_hora_pico_mayor_tiempo(self):
        """Hora pico debe predecir más tiempo que hora valle."""
        pred_pico = predecir_viaje(
            self.modelo, self.encoders, self.feature_cols,
            paradas=10, tiempo_base=30, hora=8,
            dia="lunes", clima="soleado", bus_lleno=0
        )
        pred_valle = predecir_viaje(
            self.modelo, self.encoders, self.feature_cols,
            paradas=10, tiempo_base=30, hora=14,
            dia="lunes", clima="soleado", bus_lleno=0
        )
        self.assertGreaterEqual(pred_pico, pred_valle)

    def test_lluvia_mayor_que_sol(self):
        """Lluvia debe predecir más tiempo que sol."""
        pred_lluvia = predecir_viaje(
            self.modelo, self.encoders, self.feature_cols,
            paradas=10, tiempo_base=30, hora=12,
            dia="miercoles", clima="lluvia", bus_lleno=0
        )
        pred_sol = predecir_viaje(
            self.modelo, self.encoders, self.feature_cols,
            paradas=10, tiempo_base=30, hora=12,
            dia="miercoles", clima="soleado", bus_lleno=0
        )
        self.assertGreaterEqual(pred_lluvia, pred_sol)

    def test_guardar_y_cargar(self):
        guardar_modelo(self.modelo, self.encoders, self.feature_cols, "test_model.pkl")
        modelo2, encoders2, cols2 = cargar_modelo("test_model.pkl")
        pred1 = predecir_viaje(self.modelo, self.encoders, self.feature_cols,
                               10, 30, 10, "martes", "soleado", 0)
        pred2 = predecir_viaje(modelo2, encoders2, cols2,
                               10, 30, 10, "martes", "soleado", 0)
        self.assertEqual(pred1, pred2)
        os.remove("test_model.pkl")

    def test_profundidad_arbol(self):
        self.assertLessEqual(self.modelo.get_depth(), 6)

    def test_n_features(self):
        self.assertEqual(self.modelo.n_features_in_, len(self.feature_cols))


if __name__ == "__main__":
    print("═" * 60)
    print("  PRUEBAS DEL MODELO DE APRENDIZAJE AUTOMÁTICO")
    print("═" * 60)
    unittest.main(verbosity=2)
