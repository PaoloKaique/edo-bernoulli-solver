"""EDO Bernoulli Solver: clasifica, linealiza y resuelve y' + P(x)y = Q(x)y^n."""

from __future__ import annotations

import argparse
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.clasificador import analizar_y_clasificar
from src.resolvedor import resolver_edo
from src.transformador import transformar_a_lineal
from src.utils import exportar_latex, imprimir_reporte

CASOS_PRUEBA = [
    "y' + (1/x)*y = x * y**2",        # Bernoulli (n=2)
    "dy/dx - y = 5 * x * y**(1/2)",   # Bernoulli (n=1/2)
    "y' + y/x = x*y^3",               # Bernoulli (n=3)
    "x*y' + y = x**2",                # Lineal común (rechazar)
    "y'' + 2*y' + y = 0",             # Orden 2 (rechazar)
    "(y')**2 + y = x",                # Grado 2 (rechazar)
]


def ejecutar(entrada: str, mostrar_latex: bool = False, archivo_tex: str | None = None) -> bool:
    """Pipeline completo. Devuelve True si la EDO fue resuelta."""
    print(f"\n>>> Entrada: {entrada}")
    datos_clasif = analizar_y_clasificar(entrada)
    if not datos_clasif["valido"]:
        print(f"[ERROR] {datos_clasif['mensaje']}")
        return False
    try:
        datos_transf = transformar_a_lineal(datos_clasif)
        datos_resol = resolver_edo(datos_transf)
    except Exception as e:  # noqa: BLE001
        print(f"[ERROR] Falló la resolución: {e}")
        return False

    imprimir_reporte(datos_clasif, datos_transf, datos_resol)
    if mostrar_latex or archivo_tex:
        tex = exportar_latex(datos_clasif, datos_transf, datos_resol,
                             documento_completo=bool(archivo_tex))
        if mostrar_latex:
            print("\n--- LaTeX ---\n" + tex)
        if archivo_tex:
            with open(archivo_tex, "w", encoding="utf-8") as f:
                f.write(tex)
            print(f"\nLaTeX guardado en: {archivo_tex}")
    return True


def modo_interactivo(mostrar_latex: bool) -> None:
    print("EDO de Bernoulli: y' + P(x)y = Q(x)y^n   (línea vacía o 'salir' para terminar)")
    while True:
        try:
            entrada = input("\nEDO> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if entrada.lower() in ("", "salir", "exit", "q"):
            break
        ejecutar(entrada, mostrar_latex)


def main() -> None:
    ap = argparse.ArgumentParser(description="Solver de EDOs de Bernoulli")
    ap.add_argument("-e", "--edo", help="Resuelve una EDO dada y termina")
    ap.add_argument("-c", "--casos", action="store_true", help="Ejecuta los casos de prueba")
    ap.add_argument("-l", "--latex", action="store_true", help="Imprime también el LaTeX")
    ap.add_argument("-o", "--salida", metavar="ARCHIVO.tex", help="Guarda un .tex compilable")
    args = ap.parse_args()

    if args.casos:
        for caso in CASOS_PRUEBA:
            ejecutar(caso, args.latex)
    elif args.edo:
        ok = ejecutar(args.edo, args.latex, args.salida)
        sys.exit(0 if ok else 1)
    else:
        modo_interactivo(args.latex)


if __name__ == "__main__":
    main()