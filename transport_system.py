"""
transport_system.py
===================
Sistema inteligente de transporte masivo.
Integra la base de conocimiento (reglas lógicas) con los algoritmos de búsqueda
para producir rutas óptimas con explicación razonada.

Autor Estudiante 1: [TU NOMBRE]
"""

from knowledge_base import KnowledgeBase
from search_algorithms import a_star, bfs, dfs, compare_algorithms, SearchResult


class TransportSystem:
    """
    Sistema experto de transporte masivo.
    Combina:
      - Base de conocimiento con reglas lógicas (KnowledgeBase)
      - Motor de búsqueda heurística A* como algoritmo principal
      - Encadenamiento de reglas para razonar sobre la ruta
    """

    def __init__(self):
        self.kb = KnowledgeBase()

    # ──────────────────────────────────────────────────────────────
    # CONSULTA PRINCIPAL
    # ──────────────────────────────────────────────────────────────

    def find_best_route(self, origin_id: str, destination_id: str,
                        algorithm: str = "A*") -> dict:
        """
        Encuentra la mejor ruta entre dos estaciones.

        Parámetros:
            origin_id      : id de estación origen
            destination_id : id de estación destino
            algorithm      : "A*" (defecto) | "BFS" | "DFS"

        Retorna dict con:
            - inference   : conclusiones del motor de reglas
            - result      : SearchResult con la ruta
            - route_info  : descripción legible de la ruta
            - error       : mensaje de error si aplica
        """
        response = {
            "origin": origin_id,
            "destination": destination_id,
            "algorithm": algorithm,
            "inference": [],
            "result": None,
            "route_info": [],
            "error": None,
        }

        # Paso 1: Motor de inferencia (reglas lógicas)
        inferences = self.kb.infer(origin_id, destination_id)
        response["inference"] = inferences

        # Verificar si el motor de reglas bloqueó la búsqueda
        for inf in inferences:
            if "ERROR" in inf or "no se requiere ruta" in inf.lower() or "no se necesita ruta" in inf.lower():
                response["error"] = inf
                return response

        # Paso 2: Algoritmo de búsqueda
        algo_map = {"A*": a_star, "BFS": bfs, "DFS": dfs}
        search_fn = algo_map.get(algorithm.upper(), a_star)
        result = search_fn(self.kb, origin_id, destination_id)
        response["result"] = result

        if result is None:
            response["error"] = "No se encontró ruta entre las estaciones indicadas."
            return response

        # Paso 3: Construir descripción de la ruta
        response["route_info"] = self._build_route_info(result)
        return response

    def _build_route_info(self, result: SearchResult) -> list[dict]:
        """Construye pasos legibles de la ruta encontrada."""
        steps = []
        path = result.path
        lines = result.lines_used

        for i, station_id in enumerate(path):
            name = self.kb.get_station_name(station_id)
            step = {
                "step": i + 1,
                "station_id": station_id,
                "station_name": name,
            }
            if i < len(lines):
                step["line"] = lines[i]
                # Detectar transbordo
                if i > 0 and lines[i] != lines[i - 1]:
                    step["transfer"] = True
            steps.append(step)
        return steps

    # ──────────────────────────────────────────────────────────────
    # COMPARACIÓN DE ALGORITMOS
    # ──────────────────────────────────────────────────────────────

    def compare_routes(self, origin_id: str, destination_id: str) -> dict:
        """Compara A*, BFS y DFS para la misma ruta."""
        return compare_algorithms(self.kb, origin_id, destination_id)

    # ──────────────────────────────────────────────────────────────
    # UTILIDADES
    # ──────────────────────────────────────────────────────────────

    def list_stations(self):
        return self.kb.list_stations()

    def station_exists(self, station_id: str) -> bool:
        return station_id in self.kb.stations
