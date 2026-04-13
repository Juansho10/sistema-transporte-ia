"""
tests/test_routes.py
====================
Pruebas unitarias del sistema inteligente de transporte masivo.
Evidencia las pruebas realizadas para el entregable del proyecto.

"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import unittest
from knowledge_base import KnowledgeBase
from search_algorithms import a_star, bfs, dfs
from transport_system import TransportSystem


class TestKnowledgeBase(unittest.TestCase):
    """Pruebas sobre la base de conocimiento y motor de reglas."""

    def setUp(self):
        self.kb = KnowledgeBase()

    def test_stations_loaded(self):
        """Verifica que las estaciones se cargaron correctamente."""
        self.assertGreater(len(self.kb.stations), 0)
        self.assertIn("cuba", self.kb.stations)
        self.assertIn("dosquebradas", self.kb.stations)

    def test_graph_bidirectional(self):
        """Verifica que el grafo es bidireccional."""
        neighbors_cuba = [n["to"] for n in self.kb.get_neighbors("cuba")]
        neighbors_villa = [n["to"] for n in self.kb.get_neighbors("villa_santana")]
        self.assertIn("villa_santana", neighbors_cuba)
        self.assertIn("cuba", neighbors_villa)

    def test_inference_same_station(self):
        """REGLA-5: origen == destino debe activarse."""
        inferences = self.kb.infer("cuba", "cuba")
        self.assertTrue(any("misma estación" in i.lower() for i in inferences))

    def test_inference_invalid_station(self):
        """REGLA-4: estación inexistente debe detectarse."""
        inferences = self.kb.infer("cuba", "xxxx_no_existe")
        self.assertTrue(any("ERROR" in i for i in inferences))

    def test_inference_same_zone(self):
        """REGLA-2: misma zona debe detectarse."""
        inferences = self.kb.infer("cuba", "estadio")
        self.assertTrue(any("zona" in i.lower() for i in inferences))

    def test_inference_different_zones(self):
        """REGLA-3: zonas distintas deben sugerir transbordo."""
        inferences = self.kb.infer("cuba", "dosquebradas")
        self.assertTrue(any("transbordo" in i.lower() for i in inferences))


class TestSearchAlgorithms(unittest.TestCase):
    """Pruebas de los algoritmos de búsqueda."""

    def setUp(self):
        self.kb = KnowledgeBase()

    def test_astar_finds_route(self):
        """A* debe encontrar una ruta válida."""
        result = a_star(self.kb, "cuba", "dosquebradas")
        self.assertIsNotNone(result)
        self.assertEqual(result.path[0], "cuba")
        self.assertEqual(result.path[-1], "dosquebradas")

    def test_bfs_finds_route(self):
        """BFS debe encontrar una ruta válida."""
        result = bfs(self.kb, "cuba", "dosquebradas")
        self.assertIsNotNone(result)
        self.assertEqual(result.path[-1], "dosquebradas")

    def test_dfs_finds_route(self):
        """DFS debe encontrar una ruta."""
        result = dfs(self.kb, "cuba", "terminal")
        self.assertIsNotNone(result)

    def test_astar_optimality(self):
        """A* debe retornar tiempo menor o igual que BFS para misma ruta."""
        r_astar = a_star(self.kb, "cuba", "dosquebradas")
        r_bfs   = bfs(self.kb,   "cuba", "dosquebradas")
        self.assertIsNotNone(r_astar)
        self.assertIsNotNone(r_bfs)
        self.assertLessEqual(r_astar.total_time, r_bfs.total_time)

    def test_no_route_isolated(self):
        """Sin conexión posible debe retornar None."""
        # Forzamos origen inválido artificialmente accediendo al grafo
        self.kb.graph["isla_falsa"] = []
        result = a_star(self.kb, "isla_falsa", "cuba")
        self.assertIsNone(result)

    def test_same_station(self):
        """Ruta de estación a sí misma."""
        # El sistema filtra esto antes en TransportSystem, pero el algoritmo
        # puede retornar camino de longitud 1
        result = a_star(self.kb, "cuba", "cuba")
        # A* retorna el nodo origen como destino directamente
        if result:
            self.assertEqual(result.path[0], result.path[-1])

    def test_short_route(self):
        """Cuba a Villa Santana (adyacentes) debe ser 1 conexión."""
        result = a_star(self.kb, "cuba", "villa_santana")
        self.assertIsNotNone(result)
        self.assertEqual(len(result.path), 2)
        self.assertEqual(result.total_time, 3)


class TestTransportSystem(unittest.TestCase):
    """Pruebas de integración del sistema completo."""

    def setUp(self):
        self.ts = TransportSystem()

    def test_find_best_route_astar(self):
        """Sistema completo encuentra ruta con A*."""
        response = self.ts.find_best_route("cuba", "dosquebradas", "A*")
        self.assertIsNone(response["error"])
        self.assertIsNotNone(response["result"])
        self.assertGreater(len(response["route_info"]), 0)

    def test_error_on_same_station(self):
        """Sistema reporta error cuando origen == destino."""
        response = self.ts.find_best_route("cuba", "cuba", "A*")
        self.assertIsNotNone(response["error"])

    def test_error_on_invalid_station(self):
        """Sistema reporta error cuando la estación no existe."""
        response = self.ts.find_best_route("cuba", "no_existe", "A*")
        self.assertIsNotNone(response["error"])

    def test_compare_all_algorithms(self):
        """Comparación retorna resultados para los 3 algoritmos."""
        results = self.ts.compare_routes("cuba", "dosquebradas")
        self.assertIn("A*", results)
        self.assertIn("BFS", results)
        self.assertIn("DFS", results)

    def test_inference_present(self):
        """Las inferencias del motor de reglas siempre se incluyen."""
        response = self.ts.find_best_route("cuba", "terminal", "A*")
        self.assertGreater(len(response["inference"]), 0)


if __name__ == "__main__":
    print("═" * 60)
    print("  PRUEBAS DEL SISTEMA INTELIGENTE DE TRANSPORTE")
    print("═" * 60)
    unittest.main(verbosity=2)
