"""
knowledge_base.py
=================
Base de conocimiento del sistema de transporte masivo (Megabús - Pereira).
Representa el conocimiento mediante REGLAS LÓGICAS (Hechos + Reglas de inferencia).

"""

# ─────────────────────────────────────────────────────────────────────────────
# HECHOS: Estaciones del sistema (nodos del grafo)
# Representan proposiciones verdaderas en el dominio
# ─────────────────────────────────────────────────────────────────────────────
STATIONS = {
    # id_estacion: {"name": nombre, "zone": zona, "coords": (lat, lon)}
    "cuba":            {"name": "Cuba",                    "zone": "sur",    "coords": (4.7800, -75.7100)},
    "villa_santana":   {"name": "Villa Santana",           "zone": "sur",    "coords": (4.7840, -75.7070)},
    "galicia":         {"name": "Galicia",                 "zone": "sur",    "coords": (4.7870, -75.7040)},
    "estadio":         {"name": "Estadio",                 "zone": "centro", "coords": (4.7900, -75.7010)},
    "dosquebradas":    {"name": "Dosquebradas",            "zone": "norte",  "coords": (4.8320, -75.6700)},
    "terminal":        {"name": "Terminal",                "zone": "norte",  "coords": (4.8100, -75.6900)},
    "centro":          {"name": "Centro / El Lago",        "zone": "centro", "coords": (4.8130, -75.6960)},
    "corocito":        {"name": "Corocito",                "zone": "centro", "coords": (4.8060, -75.6980)},
    "san_nicolas":     {"name": "San Nicolás",             "zone": "centro", "coords": (4.8090, -75.7000)},
    "rio_otun":        {"name": "Río Otún",                "zone": "centro", "coords": (4.8050, -75.7020)},
    "boston":          {"name": "Boston",                  "zone": "norte",  "coords": (4.8180, -75.6950)},
    "villa_del_prado": {"name": "Villa del Prado",         "zone": "norte",  "coords": (4.8220, -75.6920)},
    "nacederos":       {"name": "Nacederos",               "zone": "norte",  "coords": (4.8270, -75.6870)},
    "el_plumón":       {"name": "El Plumón",               "zone": "norte",  "coords": (4.8300, -75.6800)},
    "simón_bolivar":   {"name": "Simón Bolívar",           "zone": "centro", "coords": (4.8110, -75.6970)},
    "av_30_agosto":    {"name": "Av. 30 de Agosto",        "zone": "centro", "coords": (4.8070, -75.6990)},
    "kennedy":         {"name": "Kennedy",                  "zone": "sur",    "coords": (4.7950, -75.7030)},
    "la_roma":         {"name": "La Roma",                  "zone": "sur",    "coords": (4.7920, -75.7020)},
}

# ─────────────────────────────────────────────────────────────────────────────
# HECHOS: Conexiones directas entre estaciones
# Formato: (estacion_a, estacion_b, tiempo_minutos, linea)
# Representa relaciones bidireccionales en el grafo de conocimiento
# ─────────────────────────────────────────────────────────────────────────────
CONNECTIONS = [
    # Línea troncal principal (eje norte-sur)
    ("cuba",            "villa_santana",   3,  "troncal"),
    ("villa_santana",   "galicia",         3,  "troncal"),
    ("galicia",         "estadio",         4,  "troncal"),
    ("estadio",         "kennedy",         3,  "troncal"),
    ("kennedy",         "la_roma",         3,  "troncal"),
    ("la_roma",         "rio_otun",        4,  "troncal"),
    ("rio_otun",        "av_30_agosto",    3,  "troncal"),
    ("av_30_agosto",    "corocito",        2,  "troncal"),
    ("corocito",        "san_nicolas",     2,  "troncal"),
    ("san_nicolas",     "simón_bolivar",   2,  "troncal"),
    ("simón_bolivar",   "centro",          3,  "troncal"),
    ("centro",          "terminal",        4,  "troncal"),
    ("terminal",        "boston",          5,  "troncal"),
    ("boston",          "villa_del_prado", 4,  "troncal"),
    ("villa_del_prado", "nacederos",       3,  "troncal"),
    ("nacederos",       "el_plumón",       3,  "troncal"),
    ("el_plumón",       "dosquebradas",    4,  "troncal"),

    # Conexiones alimentadoras (transbordos)
    ("centro",          "simón_bolivar",   2,  "alimentadora"),
    ("terminal",        "dosquebradas",    6,  "alimentadora"),
    ("estadio",         "rio_otun",        5,  "alimentadora"),
]

# ─────────────────────────────────────────────────────────────────────────────
# REGLAS LÓGICAS de inferencia
# Cada regla tiene la forma: SI <condición> ENTONCES <conclusión>
# Se expresan como funciones predicado que retornan True/False
# ─────────────────────────────────────────────────────────────────────────────

class KnowledgeBase:
    """
    Motor de reglas lógicas.
    Implementa un encadenamiento hacia adelante (forward chaining) simple
    sobre los hechos del dominio de transporte.
    """

    def __init__(self):
        self.stations = STATIONS
        self.connections = CONNECTIONS
        self._build_graph()
        self.rules = self._define_rules()

    def _build_graph(self):
        """Construye el grafo de adyacencia desde los hechos de conexión."""
        self.graph = {s: [] for s in self.stations}
        for a, b, tiempo, linea in self.connections:
            self.graph[a].append({"to": b, "time": tiempo, "line": linea})
            self.graph[b].append({"to": a, "time": tiempo, "line": linea})  # bidireccional

    def _define_rules(self):
        """
        Define las reglas lógicas del sistema experto.
        Formato: {"nombre": str, "condicion": callable, "conclusion": callable}
        """
        return [
            {
                "nombre": "REGLA-1: Ruta directa posible",
                "descripcion": "SI existe conexión directa ENTONCES no hay transbordo",
                "condicion": lambda a, b: b in [n["to"] for n in self.graph.get(a, [])],
                "conclusion": lambda a, b: f"Ruta directa disponible de {a} a {b} sin transbordo."
            },
            {
                "nombre": "REGLA-2: Misma zona, tiempo reducido",
                "descripcion": "SI origen y destino están en la misma zona ENTONCES tiempo estimado menor a 15 min",
                "condicion": lambda a, b: (
                    a in self.stations and b in self.stations and
                    self.stations[a]["zone"] == self.stations[b]["zone"]
                ),
                "conclusion": lambda a, b: f"Origen y destino en zona '{self.stations[a]['zone']}': viaje corto esperado (< 15 min)."
            },
            {
                "nombre": "REGLA-3: Requiere transbordo",
                "descripcion": "SI origen y destino están en zonas distintas ENTONCES se requiere al menos un transbordo en el centro",
                "condicion": lambda a, b: (
                    a in self.stations and b in self.stations and
                    self.stations[a]["zone"] != self.stations[b]["zone"] and
                    b not in [n["to"] for n in self.graph.get(a, [])]
                ),
                "conclusion": lambda a, b: "Zonas distintas detectadas: se recomienda transbordo en Centro o Terminal."
            },
            {
                "nombre": "REGLA-4: Estación no existe",
                "descripcion": "SI la estación no está en la base de conocimiento ENTONCES reportar error",
                "condicion": lambda a, b: a not in self.stations or b not in self.stations,
                "conclusion": lambda a, b: f"ERROR: Estación '{a if a not in self.stations else b}' no existe en la base de conocimiento."
            },
            {
                "nombre": "REGLA-5: Misma estación origen y destino",
                "descripcion": "SI origen == destino ENTONCES no se necesita ruta",
                "condicion": lambda a, b: a == b,
                "conclusion": lambda a, b: "Origen y destino son la misma estación. No se requiere ruta."
            },
        ]

    def infer(self, origin: str, destination: str) -> list[str]:
        """
        Motor de inferencia: aplica reglas por encadenamiento hacia adelante.
        Retorna lista de conclusiones disparadas.
        """
        conclusions = []
        for rule in self.rules:
            if rule["condicion"](origin, destination):
                conclusion = rule["conclusion"](origin, destination)
                conclusions.append(f"[{rule['nombre']}] → {conclusion}")
        return conclusions

    def get_neighbors(self, station: str) -> list[dict]:
        """Retorna vecinos de una estación (hecho del grafo)."""
        return self.graph.get(station, [])

    def get_station_name(self, station_id: str) -> str:
        return self.stations.get(station_id, {}).get("name", station_id)

    def get_coords(self, station_id: str) -> tuple:
        return self.stations.get(station_id, {}).get("coords", (0, 0))

    def list_stations(self) -> list[tuple]:
        """Retorna lista de (id, nombre) de todas las estaciones."""
        return [(sid, info["name"]) for sid, info in self.stations.items()]
