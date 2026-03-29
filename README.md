# 🚌 Sistema Inteligente de Transporte Masivo — Megabús Pereira

**Actividad 3 — Inteligencia Artificial**  
Sistema basado en conocimiento con búsqueda heurística A* para encontrar rutas óptimas en el transporte masivo de Pereira.

---

## 📌 Descripción

Este proyecto implementa un **sistema experto** que combina:

- **Base de conocimiento con reglas lógicas** (cap. 2 y 3, Benítez 2014): hechos sobre estaciones y conexiones, reglas de inferencia hacia adelante.
- **Algoritmo de búsqueda heurística A\*** (cap. 9, Benítez 2014): encuentra la ruta de menor tiempo usando distancia GPS como heurística admisible.
- Comparación con BFS y DFS para análisis de estrategias.

---

## 🗂️ Estructura del proyecto

```
transporte-ia/
├── knowledge_base.py      # Base de conocimiento y motor de reglas lógicas
├── search_algorithms.py   # A*, BFS, DFS
├── transport_system.py    # Sistema integrador
├── main.py               # Interfaz de usuario CLI
├── tests/
│   └── test_routes.py    # Pruebas unitarias
└── README.md
```

---

## ▶️ Instalación y ejecución

### Requisitos
- Python 3.10 o superior (sin dependencias externas)

### Clonar el repositorio
```bash
git clone <URL_DEL_REPOSITORIO>
cd transporte-ia
```

### Ejecutar el sistema interactivo
```bash
python main.py
```

### Ejecutar modo demo (casos de prueba automáticos)
```bash
python main.py --demo
```

### Ejecutar pruebas unitarias
```bash
python -m pytest tests/ -v
# o sin pytest:
python tests/test_routes.py
```

---

## 💡 Ejemplo de uso

```
ESTACIÓN ORIGEN   → cuba
ESTACIÓN DESTINO  → dosquebradas
ALGORITMO         → A*

🧠 MOTOR DE INFERENCIA:
  [REGLA-3] → Zonas distintas: se recomienda transbordo en Centro
  [REGLA-1] → No existe conexión directa

🗺️ RUTA ENCONTRADA:
   1. 🚉 Cuba                [troncal]
   2. 🚉 Villa Santana       [troncal]
   ...
  15. 🚉 Dosquebradas        [troncal]

  ⏱️  Tiempo total: 54 minutos
  🚏 Paradas: 17
```

---

## 🔬 Fundamento teórico

| Componente | Referencia |
|---|---|
| Representación con reglas lógicas | Benítez (2014), Cap. 2-3 |
| Motor de inferencia (forward chaining) | Benítez (2014), Cap. 3 |
| Búsqueda A* con heurística admisible | Benítez (2014), Cap. 9 |
| Comparación BFS / DFS | Benítez (2014), Cap. 9 |

---

## 👥 Autores

| Estudiante | Módulos desarrollados |
|---|---|
| [TU NOMBRE] | `knowledge_base.py`, `transport_system.py`, `README.md` |
| [NOMBRE COMPAÑERO] | `search_algorithms.py`, `main.py`, `tests/test_routes.py` |
