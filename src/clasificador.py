import re
import sympy as sp
from sympy.parsing.sympy_parser import (
    parse_expr,
    standard_transformations,
    implicit_multiplication_application,
    convert_xor,
)

x = sp.Symbol('x')
y_func = sp.Function('y')(x)
yp = sp.Derivative(y_func, x)

# Marcadores internos (con guion bajo: el parser nunca los divide en letras).
_Y, _YP, _YPP = sp.symbols('_Y_ _YP_ _YPP_')

_TRANSF = standard_transformations + (convert_xor, implicit_multiplication_application)


def _tokenizar(s: str) -> str:
    """
    Sustituye y, y', y'', dy/dx, d2y/dx2 por marcadores simples ANTES de parsear.
    Así la multiplicación implícita nunca ve 'y(x)' ni 'Derivative(...)'.
    """
    s = s.strip().replace("’", "'").replace("′", "'")
    s = re.sub(r"d\^?2\s*y\s*/\s*dx\^?2", " _YPP_ ", s)
    s = re.sub(r"y'\s*'\s*(\(\s*x\s*\))?", " _YPP_ ", s)
    s = re.sub(r"dy\s*/\s*dx", " _YP_ ", s)
    s = re.sub(r"y'\s*(\(\s*x\s*\))?", " _YP_ ", s)
    s = re.sub(r"\by\s*\(\s*x\s*\)", " _Y_ ", s)
    s = re.sub(r"\by\b", " _Y_ ", s)
    return s


def parsear_edo(entrada_str: str) -> sp.Eq:
    cadena = _tokenizar(entrada_str)
    ld = {
        'x': x, '_Y_': _Y, '_YP_': _YP, '_YPP_': _YPP,
        'sin': sp.sin, 'cos': sp.cos, 'tan': sp.tan,
        'exp': sp.exp, 'log': sp.log, 'ln': sp.log, 'sqrt': sp.sqrt,
        'e': sp.E, 'pi': sp.pi,
    }

    def p(t):
        return parse_expr(t, local_dict=ld, transformations=_TRANSF)

    if "=" in cadena:
        lhs, rhs = cadena.split("=", 1)
        expr = p(lhs) - p(rhs)
    else:
        expr = p(cadena)
    # Marcadores -> objetos reales de SymPy (YPP primero)
    # (sp.Symbol('y') cubre casos como '2xy', que el parser divide en x*y)
    expr = expr.subs({_YPP: sp.Derivative(y_func, (x, 2)),
                      _YP: yp, _Y: y_func, sp.Symbol('y'): y_func})
    return sp.Eq(expr, 0)


def obtener_orden_y_grado(eq: sp.Eq):
    expr = sp.simplify(eq.lhs - eq.rhs)
    derivadas = [d for d in expr.atoms(sp.Derivative) if d.expr == y_func]
    if not derivadas:
        return 0, None, expr
    orden = max(len(d.variables) for d in derivadas)
    dmax = sp.Derivative(y_func, (x, orden))
    try:
        grado = sp.Poly(expr, dmax).degree()
    except Exception:
        grado = None
    return orden, grado, expr


def extraer_componentes_bernoulli(expr: sp.Expr):
    d = sp.Dummy('d')
    sols = sp.solve(expr.subs(yp, d), d)
    if not sols:
        return False, None, None, None
    f = sp.expand(sols[0])
    terminos = sp.Add.make_args(f)
    if len(terminos) != 2:
        return False, None, None, None
    t_lin, t_pow, n_val = None, None, None
    for t in terminos:
        _, fac_y = t.as_independent(y_func, as_Add=False)
        if fac_y == y_func:
            t_lin = t
        elif isinstance(fac_y, sp.Pow) and fac_y.base == y_func:
            t_pow, n_val = t, fac_y.exp
        elif fac_y == 1:
            t_pow, n_val = t, 0
    if t_lin is not None and t_pow is not None and n_val not in (None, 0, 1):
        px = sp.simplify(-t_lin / y_func)
        qx = sp.simplify(t_pow / (y_func**n_val))
        if not px.has(y_func) and not qx.has(y_func):
            return True, px, qx, n_val
    return False, None, None, None


def analizar_y_clasificar(entrada_str: str):
    try:
        eq = parsear_edo(entrada_str)
    except Exception as e:
        return {"valido": False, "mensaje": f"Error de parseo: {e}"}
    orden, grado, expr = obtener_orden_y_grado(eq)
    if orden != 1 or grado != 1:
        return {"valido": False, "mensaje": f"Rechazada: orden={orden}, grado={grado}"}
    es_bern, px, qx, n = extraer_componentes_bernoulli(expr)
    if not es_bern:
        return {"valido": False, "mensaje": "No cumple forma Bernoulli y' + P(x)y = Q(x)y^n"}
    return {"valido": True, "P_x": px, "Q_x": qx, "n": n,
            "expr_canonica": sp.Eq(yp + px*y_func, qx*(y_func**n))}