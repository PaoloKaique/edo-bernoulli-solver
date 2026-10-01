"""
Módulo III: Resolvedor analítico.

Resuelve la EDO lineal v' + P_v(x) v = Q_v(x) por factor integrante,
despeja v(x) y revierte la sustitución v = y^(1-n) para obtener y(x).
"""

from __future__ import annotations

from typing import Any

import sympy as sp

from .clasificador import x, y_func

C1 = sp.Symbol('C1')


def _verificar_lineal(pv: sp.Expr, qv: sp.Expr, v_expr: sp.Expr) -> bool:
    """Comprueba que v(x) satisface v' + P_v v - Q_v = 0."""
    residuo = sp.diff(v_expr, x) + pv * v_expr - qv
    return sp.simplify(residuo) == 0


def _verificar_bernoulli(pv: sp.Expr, qv: sp.Expr, n: sp.Expr, y_expr: sp.Expr, v_expr: sp.Expr):
    """
    Comprueba y' + P y - Q y^n = 0 con P = P_v/(1-n), Q = Q_v/(1-n).
    Devuelve True/False, o None si no se pudo decidir.

    Con potencias fraccionarias la solución solo vale en la rama donde
    v = y^(1-n) > 0, así que la prueba numérica se hace solo en esos puntos.
    """
    k = 1 - n
    px, qx = pv / k, qv / k
    residuo = sp.diff(y_expr, x) + px * y_expr - qx * y_expr**n
    try:
        if sp.simplify(residuo) == 0:
            return True
    except Exception:
        pass
    try:
        validos = 0
        for xv in (0.7, 1.3, 2.1):
            for cv in (-3.9, 2.7, 5.2, 12.0, 50.0):
                sub = {x: xv, C1: cv}
                vv = complex(sp.N(v_expr.subs(sub)))
                if abs(vv.imag) > 1e-12 or vv.real <= 0:
                    continue
                validos += 1
                if abs(complex(sp.N(residuo.subs(sub)))) > 1e-7:
                    return False
        return True if validos else None
    except Exception:
        return None


def resolver_edo(datos_transformados: dict[str, Any]) -> dict[str, Any]:
    """
    Resuelve la EDO lineal producida por `transformar_a_lineal`.

    Retorna un dict con: factor_integrante, integral_rhs, solucion_v,
    solucion_y, ecuacion_solucion, constante, notas, verificacion, pasos.
    Cada paso es un dict {"titulo", "detalle", "expresion"}.
    """
    for clave in ("P_v", "Q_v", "v_func", "n"):
        if clave not in datos_transformados:
            raise ValueError(f"Falta la clave '{clave}' en los datos del transformador.")

    pv: sp.Expr = sp.sympify(datos_transformados["P_v"])
    qv: sp.Expr = sp.sympify(datos_transformados["Q_v"])
    v_func: sp.Expr = datos_transformados["v_func"]
    n: sp.Expr = sp.sympify(datos_transformados["n"])
    sust = datos_transformados.get("sustitucion", {})
    k: sp.Expr = sp.sympify(sust.get("exponente", 1 - n))
    if k == 0:
        raise ValueError("Exponente 1-n = 0: no es una EDO de Bernoulli.")

    pasos: list[dict[str, Any]] = []
    notas: list[str] = []

    # 0. EDO lineal de partida
    pasos.append({
        "titulo": "EDO lineal en v(x)",
        "detalle": "Forma canónica v' + P_v(x) v = Q_v(x).",
        "expresion": datos_transformados.get("ecuacion_lineal"),
    })

    # 1. Factor integrante
    int_p = sp.integrate(pv, x)
    mu = sp.simplify(sp.exp(int_p))
    pasos.append({
        "titulo": "Factor integrante",
        "detalle": f"∫P_v dx = {int_p};  μ(x) = exp(∫P_v dx).",
        "expresion": sp.Eq(sp.Symbol('mu(x)'), mu),
    })

    # 2. Integral del lado derecho
    integrando = sp.simplify(mu * qv)
    integral = sp.integrate(integrando, x)
    if integral.has(sp.Integral):
        notas.append("La integral de μ(x)·Q_v(x) no tiene forma elemental; se deja indicada.")
    pasos.append({
        "titulo": "Integral del lado derecho",
        "detalle": "(μ v)' = μ Q_v  ⇒  μ v = ∫ μ Q_v dx + C1.",
        "expresion": sp.Eq(sp.Symbol('∫ μ·Q_v dx'), integral),
    })

    # 3. Despeje de v(x)
    sol_v = sp.simplify((integral + C1) / mu)
    pasos.append({
        "titulo": "Solución general de v(x)",
        "detalle": "v(x) = (1/μ(x)) (∫ μ Q_v dx + C1).",
        "expresion": sp.Eq(v_func, sol_v),
    })

    # 4. Sustitución inversa
    sol_y = sp.simplify(sol_v ** (1 / k))
    pasos.append({
        "titulo": "Sustitución inversa",
        "detalle": f"y = v^(1/(1-n)) = v^({1 / k}).",
        "expresion": sp.Eq(y_func, sol_y),
    })

    # Notas sobre ramas y soluciones singulares
    inv = sp.nsimplify(1 / k)
    if isinstance(inv, sp.Rational) and inv.q % 2 == 0:
        notas.append(
            f"Como 1/(1-n) = {inv} tiene denominador par, la solución real también admite "
            "la rama negativa: y = ± v^(1/(1-n)) (donde v ≥ 0)."
        )
    if n.is_number and n > 0:
        notas.append("Para n > 0, y(x) = 0 es además una solución (singular) que no se obtiene de C1.")

    verificacion = _verificar_bernoulli(pv, qv, n, sol_y, sol_v)
    solucion_ok = _verificar_lineal(pv, qv, sol_v)
    if solucion_ok is False:
        notas.append("Advertencia: v(x) no satisfizo la verificación simbólica de la EDO lineal.")

    return {
        "factor_integrante": mu,
        "integral_rhs": integral,
        "solucion_v": sol_v,
        "solucion_y": sol_y,
        "ecuacion_solucion": sp.Eq(y_func, sol_y),
        "constante": C1,
        "notas": notas,
        "verificacion": verificacion,
        "pasos": pasos,
    }