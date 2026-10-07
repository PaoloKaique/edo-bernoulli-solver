"""Utilidades de presentación: reporte en consola y exportación a LaTeX."""

from __future__ import annotations

from typing import Any

import sympy as sp

from .clasificador import x, y_func

_L = sp.latex


def _tex_y(expr: sp.Expr) -> str:
    """LaTeX de una expresión en y(x), con notación compacta y(x) -> y."""
    return _L(expr).replace(r"y{\left(x \right)}", "y")


def _par(expr: sp.Expr) -> str:
    """Paréntesis solo cuando hacen falta (sumas)."""
    return r"\left(" + _L(expr) + r"\right)" if isinstance(expr, sp.Add) else _L(expr)


def _sumando(coef: sp.Expr, simbolo: str) -> str:
    """Devuelve ' + c simbolo' o ' - c simbolo' con signo limpio."""
    coef = sp.sympify(coef)
    if coef == 0:
        return ""
    signo = " - " if coef.could_extract_minus_sign() else " + "
    mag = -coef if signo == " - " else coef
    cuerpo = simbolo if mag == 1 else _par(mag) + r"\," + simbolo
    return signo + cuerpo


def exportar_latex(
    datos_clasif: dict[str, Any],
    datos_transf: dict[str, Any],
    datos_resol: dict[str, Any],
    documento_completo: bool = False,
) -> str:
    """
    Genera el desarrollo completo en LaTeX.
    Con documento_completo=True devuelve un .tex compilable; si no, solo el cuerpo.
    """
    px, qx, n = datos_clasif["P_x"], datos_clasif["Q_x"], datos_clasif["n"]
    k = 1 - n
    pv, qv = datos_transf["P_v"], datos_transf["Q_v"]
    mu, integral = datos_resol["factor_integrante"], datos_resol["integral_rhs"]
    sol_v, sol_y = datos_resol["solucion_v"], datos_resol["solucion_y"]

    L: list[str] = []
    a = L.append

    a(r"\section*{Resolución de la EDO de Bernoulli}")
    a(r"\subsection*{1. Forma canónica}")
    a(r"\[ y' + P(x)\,y = Q(x)\,y^{n} \]")
    a(r"\[ y'" + _sumando(px, "y") + " = " + _par(qx) + r"\,y^{" + _L(n) + r"} \]")
    a(r"\[ P(x) = " + _L(px) + r",\qquad Q(x) = " + _L(qx) + r",\qquad n = " + _L(n) + r" \]")

    a(r"\subsection*{2. Cambio de variable}")
    a(r"\[ v = y^{1-n} = y^{" + _L(k) + r"} \]")
    a(r"\[ v' = (1-n)\,y^{-n}\,y' = " + _par(k) + r"\cdot y^{" + _L(-n) + r"}\,y' \]")
    a(r"Sustituyendo $y' = -P\,y + Q\,y^{n}$ y simplificando:")
    a(r"\[ v' + (1-n)P(x)\,v = (1-n)Q(x) \]")

    a(r"\subsection*{3. EDO lineal obtenida}")
    a(r"\[ v'" + _sumando(pv, "v") + " = " + _L(qv) + r" \]")

    a(r"\subsection*{4. Factor integrante}")
    a(r"\[ \mu(x) = \exp\!\left(\int " + _par(pv) + r"\,dx\right) = " + _L(mu) + r" \]")
    a(r"\[ \int \mu(x)\,Q_v(x)\,dx = " + _L(integral) + r" \]")
    a(r"\[ v(x) = \frac{1}{\mu(x)}\left(\int \mu\,Q_v\,dx + C_1\right) = " + _L(sol_v) + r" \]")

    a(r"\subsection*{5. Solución general}")
    a(r"\[ y = v^{\frac{1}{1-n}} = v^{" + _L(1 / k) + r"} \]")
    a(r"\[ \boxed{\,y(x) = " + _L(sol_y).replace("C1", "C_{1}") + r"\,} \]")

    for nota in datos_resol.get("notas", []):
        a(r"\textit{Nota: " + nota.replace("_", r"\_").replace("≥", r"$\geq$")
          .replace("±", r"$\pm$").replace("μ", r"$\mu$").replace("∫", r"$\int$").replace("C1", "$C_1$") + "}")

    cuerpo = "\n".join(L)
    if not documento_completo:
        return cuerpo
    return (
        "\\documentclass[11pt]{article}\n"
        "\\usepackage[utf8]{inputenc}\n\\usepackage[T1]{fontenc}\n"
        "\\usepackage{amsmath,amssymb}\n\\usepackage[spanish]{babel}\n"
        "\\begin{document}\n" + cuerpo + "\n\\end{document}\n"
    )


def _fmt(expr: Any) -> str:
    return sp.pretty(expr, use_unicode=True) if isinstance(expr, sp.Basic) else str(expr)


def imprimir_reporte(
    datos_clasif: dict[str, Any],
    datos_transf: dict[str, Any],
    datos_resol: dict[str, Any],
) -> None:
    """Imprime el procedimiento paso a paso en consola."""
    linea = "=" * 64

    def titulo(t: str) -> None:
        print(f"\n{linea}\n{t}\n{linea}")

    titulo("1. CLASIFICACIÓN")
    print("Forma canónica:")
    print(_fmt(datos_clasif["expr_canonica"]))
    print(f"\nP(x) = {datos_clasif['P_x']}\nQ(x) = {datos_clasif['Q_x']}\nn    = {datos_clasif['n']}")

    titulo("2. CAMBIO DE VARIABLE")
    for paso in datos_transf["sustitucion"]["pasos"]:
        print(paso)
    sust = datos_transf["sustitucion"]
    print("\n" + _fmt(sust["definicion"]))
    print(_fmt(sust["derivada_v"]))
    print("\nEDO lineal resultante:")
    print(_fmt(datos_transf["ecuacion_lineal"]))

    titulo("3. RESOLUCIÓN DE LA EDO LINEAL")
    for i, p in enumerate(datos_resol["pasos"], 1):
        print(f"\n[{i}] {p['titulo']}")
        if p.get("detalle"):
            print(f"    {p['detalle']}")
        if p.get("expresion") is not None:
            print(_fmt(p["expresion"]))

    titulo("4. SOLUCIÓN GENERAL")
    print(_fmt(datos_resol["ecuacion_solucion"]))
    print("\n(C1 = constante arbitraria)")
    ver = datos_resol.get("verificacion")
    estado = {True: "✓ verificada por sustitución", False: "✗ NO verificada", None: "no determinada"}[ver]
    print(f"Verificación en la EDO original: {estado}")
    for nota in datos_resol.get("notas", []):
        print(f"* {nota}")