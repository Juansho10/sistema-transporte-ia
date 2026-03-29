from __future__ import annotations

"""
search_algorithms.py
====================
Implementación de algoritmos de búsqueda para encontrar la mejor ruta
en el sistema de transporte masivo.

Incluye:
  - BFS  (Búsqueda en Anchura)        — búsqueda no informada
  - DFS  (Búsqueda en Profundidad)    — búsqueda no informada
  - A*   (A-estrella)                 — búsqueda heurística (cap. 9, Benítez 2014)

La heurística de A* usa la distancia euclidiana entre coordenadas GPS,
lo que convierte este sistema en un buscador informado basado en conocimiento
geográfico del dominio.

Autor Estudiante 2: [NOMBRE DEL COMPAÑERO]
"""

import heapq
import math
from collections import deque
from knowledge_base import KnowledgeBase


# ─────────────────────────────────────────────────────────────────────────────
# HEURÍSTICA PARA A*
# ─────────────────────────────────────────────────────────────────────────────

def heuristic(kb: KnowledgeBase, station_a: str, station_b: str) -> float:
    """
    Distancia euclidiana entre dos estaciones usando sus coordenadas GPS.
    Se usa como heurística admisible h(n) en A*.

    Es admisible porque nunca sobreestima el costo real
    (la distancia en línea recta ≤ distancia real de viaje).

    Parámetros:
        kb          : instancia de KnowledgeBase
        station_a   : id de estación actual
        station_b   : id de estación destino

    Retorna:
        float — distancia aproximada escalada a minutos
    """
    lat1, lon1 = kb.get_coords(station_a)
    lat2, lon2 = kb.get_coords(station_b)
    # Distancia euclidiana en grados, escalada para representar minutos
    dist = math.sqrt((lat2 - lat1) ** 2 + (lon2 - lon1) ** 2)
    # Factor de escala empírico: ~111 km por grado, velocidad ~20 km/h → ~333 min/grado
    return dist * 333


# ─────────────────────────────────────────────────────────────────────────────
# RESULTADO ESTÁNDAR DE BÚSQUEDA
# ─────────────────────────────────────────────────────────────────────────────

class SearchResult:
    def __init__(self, algorithm: str, path: list, total_time: int,
                 nodes_explored: int, lines_used: list):
        self.algorithm = algorithm
        self.path = path                    # lista de station_ids
        self.total_time = total_time        # minutos totales
        self.nodes_explored = nodes_explored
        self.lines_used = lines_used        # líneas usadas en el recorrido

    def __repr__(self):
        return (f"SearchResult(algo={self.algorithm}, "
                f"paradas={len(self.path)}, tiempo={self.total_time} min, "
                f"explorados={self.nodes_explored})")


# ─────────────────────────────────────────────────────────────────────────────
# ALGORITMO A* (búsqueda heurística informada)
# ─────────────────────────────────────────────────────────────────────────────

def a_star(kb: KnowledgeBase, origin: str, destination: str) -> SearchResult | None:
    """
    A* (A-estrella): búsqueda heurística que minimiza f(n) = g(n) + h(n)
      g(n) = costo real acumulado desde el origen (tiempo en minutos)
      h(n) = heurística admisible (distancia euclidiana a destino)

    Garantiza la ruta óptima si h(n) es admisible.

    Complejidad: O(b^d) en peor caso, pero guiada por heurística.
    """
    # Cola de prioridad: (f, g, station, path, lines)
    start_h = heuristic(kb, origin, destination)
    open_set = [(start_h, 0, origin, [origin], [])]
    # g_score: menor costo conocido desde origen a cada nodo
    g_score = {origin: 0}
    nodes_explored = 0

    while open_set:
        f, g, current, path, lines = heapq.heappop(open_set)
        nodes_explored += 1

        if current == destination:
            return SearchResult("A*", path, g, nodes_explored, lines)

        for neighbor_info in kb.get_neighbors(current):
            neighbor = neighbor_info["to"]
            cost = neighbor_info["time"]
            line = neighbor_info["line"]
            tentative_g = g + cost

            if neighbor not in g_score or tentative_g < g_score[neighbor]:
                g_score[neighbor] = tentative_g
                h = heuristic(kb, neighbor, destination)
                f_new = tentative_g + h
                heapq.heappush(open_set, (
                    f_new, tentative_g, neighbor,
                    path + [neighbor],
                    lines + [line]
                ))

    return None  # Sin ruta


# ─────────────────────────────────────────────────────────────────────────────
# ALGORITMO BFS (Búsqueda en Anchura)
# ─────────────────────────────────────────────────────────────────────────────

def bfs(kb: KnowledgeBase, origin: str, destination: str) -> SearchResult | None:
    """
    BFS: explora nivel por nivel, garantiza ruta con MENOS PARADAS (no mínimo tiempo).
    Búsqueda no informada — no usa heurística.

    Complejidad: O(V + E) donde V=estaciones, E=conexiones.
    """
    queue = deque([(origin, [origin], 0, [])]
                  )  # (nodo, camino, tiempo, lineas)
    visited = {origin}
    nodes_explored = 0

    while queue:
        current, path, total_time, lines = queue.popleft()
        nodes_explored += 1

        if current == destination:
            return SearchResult("BFS", path, total_time, nodes_explored, lines)

        for neighbor_info in kb.get_neighbors(current):
            neighbor = neighbor_info["to"]
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((
                    neighbor,
                    path + [neighbor],
                    total_time + neighbor_info["time"],
                    lines + [neighbor_info["line"]]
                ))

    return None


# ─────────────────────────────────────────────────────────────────────────────
# ALGORITMO DFS (Búsqueda en Profundidad)
# ─────────────────────────────────────────────────────────────────────────────

def dfs(kb: KnowledgeBase, origin: str, destination: str) -> SearchResult | None:
    """
    DFS: explora en profundidad hasta encontrar el destino.
    NO garantiza ruta óptima. Incluido como comparación de estrategias.

    Complejidad: O(V + E).
    """
    stack = [(origin, [origin], 0, [])]
    visited = set()
    nodes_explored = 0

    while stack:
        current, path, total_time, lines = stack.pop()

        if current in visited:
            continue
        visited.add(current)
        nodes_explored += 1

        if current == destination:
            return SearchResult("DFS", path, total_time, nodes_explored, lines)

        for neighbor_info in kb.get_neighbors(current):
            neighbor = neighbor_info["to"]
            if neighbor not in visited:
                stack.append((
                    neighbor,
                    path + [neighbor],
                    total_time + neighbor_info["time"],
                    lines + [neighbor_info["line"]]
                ))

    return None


# ─────────────────────────────────────────────────────────────────────────────
# COMPARADOR DE ALGORITMOS
# ─────────────────────────────────────────────────────────────────────────────

def compare_algorithms(kb: KnowledgeBase, origin: str, destination: str) -> dict:
    """
    Ejecuta los tres algoritmos y retorna comparación de resultados.
    Útil para el análisis del proyecto.
    """
    results = {}
    for name, fn in [("A*", a_star), ("BFS", bfs), ("DFS", dfs)]:
        result = fn(kb, origin, destination)
        results[name] = result
    return results
