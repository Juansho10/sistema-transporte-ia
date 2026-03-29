"""
main.py
=======
Interfaz de usuario por consola para el sistema inteligente de transporte masivo.
Permite consultar rutas de forma interactiva, comparar algoritmos y ver inferencias.

Autor Estudiante 2: [NOMBRE DEL COMPAÑERO]
"""

import sys
from transport_system import TransportSystem


# ─────────────────────────────────────────────────────────────────────────────
# UTILIDADES DE PRESENTACIÓN
# ─────────────────────────────────────────────────────────────────────────────

COLORS = {
    "reset":  "\033[0m",
    "bold":   "\033[1m",
    "cyan":   "\033[96m",
    "green":  "\033[92m",
    "yellow": "\033[93m",
    "red":    "\033[91m",
    "blue":   "\033[94m",
    "magenta":"\033[95m",
}

def c(color: str, text: str) -> str:
    """Envuelve texto en color ANSI."""
    return f"{COLORS.get(color,'')}{text}{COLORS['reset']}"

def header():
    print()
    print(c("cyan",    "╔══════════════════════════════════════════════════════════╗"))
    print(c("cyan",    "║") + c("bold", "    🚌  SISTEMA INTELIGENTE DE TRANSPORTE MASIVO         ") + c("cyan", "║"))
    print(c("cyan",    "║") + "       Megabús Pereira — Búsqueda Heurística A*           " + c("cyan", "║"))
    print(c("cyan",    "║") + "       Inteligencia Artificial — UNAD                     " + c("cyan", "║"))
    print(c("cyan",    "╚══════════════════════════════════════════════════════════╝"))
    print()

def print_stations(stations: list):
    print(c("bold", "\n📍 ESTACIONES DISPONIBLES:"))
    print(c("yellow", f"{'ID':<20} {'NOMBRE':<30}"))
    print("─" * 50)
    for sid, name in sorted(stations, key=lambda x: x[1]):
        print(f"  {c('cyan', sid):<30} {name}")
    print()

def print_inference(inferences: list):
    print(c("bold", "\n🧠 MOTOR DE INFERENCIA (Reglas lógicas disparadas):"))
    print("─" * 60)
    for inf in inferences:
        print(f"  {c('magenta', '▸')} {inf}")
    print()

def print_route(route_info: list, result, algorithm: str):
    print(c("bold", f"\n🗺️  RUTA ENCONTRADA ({algorithm}):"))
    print("─" * 60)

    for step in route_info:
        icon = "🚉"
        line_info = ""
        if "transfer" in step and step["transfer"]:
            icon = "🔄"
            line_info = c("yellow", f"  [TRANSBORDO → línea {step['line']}]")
        elif "line" in step:
            line_info = c("blue", f"  [{step['line']}]")

        print(f"  {step['step']:>2}. {icon} {c('green', step['station_name'])}{line_info}")

    print("─" * 60)
    print(f"  ⏱️  Tiempo total estimado : {c('bold', str(result.total_time))} minutos")
    print(f"  🚏 Paradas               : {c('bold', str(len(result.path)))}")
    print(f"  🔍 Nodos explorados      : {c('bold', str(result.nodes_explored))}")
    print()

def print_comparison(results: dict, kb_stations: dict):
    print(c("bold", "\n📊 COMPARACIÓN DE ALGORITMOS:"))
    print("─" * 70)
    print(c("yellow", f"  {'Algoritmo':<10} {'Paradas':>8} {'Tiempo (min)':>14} {'Nodos explorados':>18}"))
    print("─" * 70)
    for algo, res in results.items():
        if res:
            print(f"  {c('cyan', algo):<20} {len(res.path):>8} {res.total_time:>14} {res.nodes_explored:>18}")
        else:
            print(f"  {c('red', algo):<20} {'Sin ruta':>8}")
    print()

def print_error(msg: str):
    print(c("red", f"\n❌ {msg}\n"))


# ─────────────────────────────────────────────────────────────────────────────
# MENÚ PRINCIPAL
# ─────────────────────────────────────────────────────────────────────────────

def select_station(system: TransportSystem, label: str) -> str:
    """Solicita al usuario seleccionar una estación."""
    stations = system.list_stations()
    print_stations(stations)
    while True:
        sid = input(c("bold", f"  Ingrese el ID de la estación {label}: ")).strip().lower()
        if system.station_exists(sid):
            return sid
        print(c("red", f"  La estación '{sid}' no existe. Intente de nuevo."))

def select_algorithm() -> str:
    """Solicita el algoritmo de búsqueda."""
    print(c("bold", "\n🔧 Seleccione el algoritmo de búsqueda:"))
    print(f"  {c('cyan', '1')}. A* (Heurística — RECOMENDADO)")
    print(f"  {c('cyan', '2')}. BFS (Anchura)")
    print(f"  {c('cyan', '3')}. DFS (Profundidad)")
    opt = input(c("bold", "  Opción [1-3]: ")).strip()
    return {"1": "A*", "2": "BFS", "3": "DFS"}.get(opt, "A*")

def menu(system: TransportSystem):
    """Bucle principal del menú."""
    while True:
        print(c("bold", "\n═══ MENÚ PRINCIPAL ═══"))
        print(f"  {c('cyan','1')}. Buscar ruta entre dos estaciones")
        print(f"  {c('cyan','2')}. Comparar los tres algoritmos")
        print(f"  {c('cyan','3')}. Listar estaciones")
        print(f"  {c('cyan','0')}. Salir")
        opt = input(c("bold", "\n  Seleccione una opción: ")).strip()

        if opt == "0":
            print(c("green", "\n¡Hasta luego! 🚌\n"))
            sys.exit(0)

        elif opt == "1":
            origin = select_station(system, "ORIGEN")
            destination = select_station(system, "DESTINO")
            algorithm = select_algorithm()

            response = system.find_best_route(origin, destination, algorithm)
            print_inference(response["inference"])

            if response["error"]:
                print_error(response["error"])
            else:
                print_route(response["route_info"], response["result"], algorithm)

        elif opt == "2":
            origin = select_station(system, "ORIGEN")
            destination = select_station(system, "DESTINO")
            results = system.compare_routes(origin, destination)

            # Mostrar inferencias del primero disponible
            inferences = system.kb.infer(origin, destination)
            print_inference(inferences)
            print_comparison(results, system.kb.stations)

            # Mostrar detalle de la mejor ruta (A*)
            best = results.get("A*")
            if best:
                route_info = system._build_route_info(best)
                print_route(route_info, best, "A* (mejor ruta)")

        elif opt == "3":
            print_stations(system.list_stations())

        else:
            print(c("red", "  Opción inválida."))


# ─────────────────────────────────────────────────────────────────────────────
# MODO DEMO (para ejecución automática de pruebas)
# ─────────────────────────────────────────────────────────────────────────────

def run_demo(system: TransportSystem):
    """Ejecuta casos de prueba predefinidos para demostración."""
    demos = [
        ("cuba", "dosquebradas", "A*"),
        ("estadio", "terminal", "BFS"),
        ("villa_santana", "nacederos", "DFS"),
        ("centro", "centro", "A*"),          # misma estación
        ("cuba", "inexistente", "A*"),        # estación no existe
    ]

    print(c("bold", "\n🎬 MODO DEMO — Casos de prueba\n"))
    print("═" * 70)

    for origin, dest, algo in demos:
        print(c("yellow", f"\n▶ Ruta: {origin.upper()} → {dest.upper()} | Algoritmo: {algo}"))
        response = system.find_best_route(origin, dest, algo)
        print_inference(response["inference"])
        if response["error"]:
            print_error(response["error"])
        else:
            print_route(response["route_info"], response["result"], algo)
        print("─" * 70)


# ─────────────────────────────────────────────────────────────────────────────
# PUNTO DE ENTRADA
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    system = TransportSystem()
    header()

    if "--demo" in sys.argv:
        run_demo(system)
    else:
        menu(system)
