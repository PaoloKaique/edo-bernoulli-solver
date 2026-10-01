"""
Módulo II: Transformador.

Recibe el diccionario validado por el clasificador (P_x, Q_x, n) y aplica
la sustitución v = y^(1-n), obteniendo la EDO lineal

    v' + (1-n) P(x) v = (1-n) Q(x)

Además devuelve el desglose algebraico de la sustitución (memoria intermedia)
para que el resolvedor pueda revertir el cambio de variable.
"""

import sympy as sp

from .clasificador import x, y_func

# Variable dependiente nueva: v(x)
v_func = sp.Function('v')(x)


def _validar_entrada(datos: dict):
    """Verifica que el diccionario provenga de una clasificación válida."""
    if not isinstance(datos, dict) or not datos.get("valido", True):
        raise ValueError("La EDO no fue validada como Bernoulli por el clasificador.")
    for clave in ("P_x", "Q_x", "n"):
        if clave not in datos or datos[clave] is None:
            raise ValueError(f"Falta la clave '{clave}' en los datos del clasificador.")
    n = sp.sympify(datos["n"])
    if n in (0, 1):
        raise ValueError("n debe ser distinto de 0 y 1 para una EDO de Bernoulli.")
    return sp.sympify(datos["P_x"]), sp.sympify(datos["Q_x"]), n


def _verificar_derivacion(px, qx, n) -> bool:
    """
    Comprueba simbólicamente que, sustituyendo y' = -P y + Q y^n en
    v' = (1-n) y^(-n) y', se obtiene -(1-n) P y^(1-n) + (1-n) Q.
    """
    y = sp.Symbol('y', positive=True)
    yp_expr = -px * y + qx * y**n
    vp_directa = (1 - n) * y**(-n) * yp_expr
    vp_esperada = -(1 - n) * px * y**(1 - n) + (1 - n) * qx
    return sp.simplify(sp.expand(vp_directa - vp_esperada)) == 0


def transformar_a_lineal(datos: dict) -> dict:
    """
    Aplica v = y^(1-n) a la EDO de Bernoulli y' + P y = Q y^n.

    Parámetros
    ----------
    datos : dict
        Salida de `analizar_y_clasificar` con claves "P_x", "Q_x", "n".

    Retorna
    -------
    dict con:
        - "ecuacion_lineal": sp.Eq  ->  v' + P_v v = Q_v
        - "P_v", "Q_v":      coeficientes de la EDO lineal
        - "v_func":          v(x) simbólica
        - "n":               exponente original
        - "sustitucion":     desglose algebraico (memoria intermedia)
    """
    px, qx, n = _validar_entrada(datos)

    k = 1 - n                      # exponente de la sustitución
    pv = sp.simplify(k * px)       # P_v(x) = (1-n) P(x)
    qv = sp.simplify(k * qx)       # Q_v(x) = (1-n) Q(x)

    # EDO lineal canónica: v' + P_v v = Q_v
    ecuacion_lineal = sp.Eq(sp.Derivative(v_func, x) + pv * v_func, qv)

    # Desglose de la sustitución
    yp_despejada = -px * y_func + qx * y_func**n            # y' = -P y + Q y^n
    sustitucion = {
        "definicion":      sp.Eq(v_func, y_func**k),                           # v = y^(1-n)
        "inversa":         sp.Eq(y_func, v_func**(1 / k)),                     # y = v^(1/(1-n))
        "derivada_v":      sp.Eq(sp.Derivative(v_func, x),
                                 k * y_func**(-n) * sp.Derivative(y_func, x)), # v' = (1-n) y^-n y'
        "y_prima":         sp.Eq(sp.Derivative(y_func, x), yp_despejada),      # y' original
        "exponente":       k,
        "derivacion_ok":   _verificar_derivacion(px, qx, n),
        "pasos": [
            "1. Ecuación original: y' + P(x) y = Q(x) y^n",
            f"2. Sustitución: v = y^(1-n) = y^({k})",
            "3. Derivada: v' = (1-n) y^(-n) y'",
            "4. Reemplazar y' = -P y + Q y^n y simplificar:",
            "   v' = -(1-n) P(x) v + (1-n) Q(x)",
            "5. Forma lineal: v' + (1-n) P(x) v = (1-n) Q(x)",
        ],
    }

    return {
        "ecuacion_lineal": ecuacion_lineal,
        "P_v": pv,
        "Q_v": qv,
        "v_func": v_func,
        "n": n,
        "sustitucion": sustitucion,
    }