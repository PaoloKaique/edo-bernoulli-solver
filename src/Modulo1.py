import sympy as sp
from sympy.parsing.sympy_parser import (
    parse_expr,
    standard_transformations,
    implicit_multiplication_application,
)

# 1. Definición de símbolos canónicos
x = sp.Symbol('x')
y = sp.Function('y')(x)
yp = sp.Derivative(y, x)      # y'
ypp = sp.Derivative(y, x, 2)   # y''


def interpretar_entrada(entrada_str: str) -> sp.Eq:
    """
    Convierte la cadena del usuario en una ecuación simbólica SymPy.
    Permite notaciones como "y'" o "dy/dx" sustituyéndolas antes del parsing.
    """
    transformaciones = standard_transformations + (implicit_multiplication_application,)
    
    # Preprocesamiento de sintaxis común
    cadena = entrada_str.replace("y''", "Derivative(y(x), x, 2)")
    cadena = cadena.replace("y'", "Derivative(y(x), x)")
    cadena = cadena.replace("dy/dx", "Derivative(y(x), x)")
    cadena = cadena.replace("y", "y(x)") if "y(x)" not in cadena else cadena
    
    # Manejar si el usuario ingresó una igualdad con '='
    local_dict = {
        'x': x,
        'y': sp.Function('y'),
        'Derivative': sp.Derivative,
        'sin': sp.sin, 'cos': sp.cos, 'tan': sp.tan,
        'exp': sp.exp, 'log': sp.log, 'sqrt': sp.sqrt
    }
    
    if "=" in cadena:
        lhs_str, rhs_str = cadena.split("=", 1)
        lhs = parse_expr(lhs_str, local_dict=local_dict, transformations=transformaciones)
        rhs = parse_expr(rhs_str, local_dict=local_dict, transformations=transformaciones)
        eq = sp.Eq(lhs - rhs, 0)
    else:
        expr = parse_expr(cadena, local_dict=local_dict, transformations=transformaciones)
        eq = sp.Eq(expr, 0)
        
    return eq


def obtener_orden_y_grado(eq: sp.Eq, funcion_y=y):
    """
    Calcula el orden y grado de la ecuación diferencial.
    """
    expr = sp.simplify(eq.lhs - eq.rhs)
    
    # Extraer todas las derivadas presentes de y(x)
    derivadas = [
        arg for arg in expr.atoms(sp.Derivative)
        if arg.expr == funcion_y
    ]
    
    if not derivadas:
        return 0, None, expr  # No es una EDO (es algebraica pura)
    
    # El orden es la derivada más alta
    orden = max(len(d.variables) for d in derivadas)
    derivada_maxima = sp.Derivative(funcion_y, x, orden)
    
    # El grado es la potencia a la que está elevada la derivada de mayor orden
    # Convertimos la expresión en polinomio respecto a esa derivada si es posible
    try:
        poli = sp.Poly(expr, derivada_maxima)
        grado = poli.degree()
    except (sp.PolynomialError, sp.GeneratorsError):
        # Si la derivada está dentro de una trascendente (ej: sin(y')), el grado no está definido
        grado = None
        
    return orden, grado, expr


def clasificar_primer_orden(expr: sp.Expr) -> list:
    """
    Evalúa las formas estándar de primer orden: F(x, y, y') = 0.
    Retorna las categorías compatibles del temario.
    """
    categorias = []
    
    # Despejar y' en la forma: y' = f(x, y)
    soluciones_yp = sp.solve(expr, yp)
    if not soluciones_yp:
        return ["No resuelto explícitamente en y'"]
    
    f = sp.simplify(soluciones_yp[0])  # dy/dx = f(x, y)
    
    # 1. Variables Separables: f(x, y) = g(x) * h(y)
    # Condición: diff(log(f), x, y) == 0 (cuando aplica) o verificación directa de factores
    try:
        factores = sp.factor(f)
        x_terms = factores.as_independent(y)[0]
        y_terms = factores.as_independent(x)[0]
        if sp.simplify(f - (x_terms * y_terms)) == 0:
            categorias.append("Variables Separables")
    except Exception:
        pass

    # 2. Homogénea: f(tx, ty) == f(x, y) -> Grado 0
    t = sp.Symbol('t', positive=True)
    f_escalada = f.subs({x: t*x, y: t*y})
    if sp.simplify(f_escalada - f) == 0:
        categorias.append("Homogénea de primer orden")

    # 3. Lineal de primer orden: y' + P(x)*y = Q(x)
    # y' - f(x, y) = 0 debe ser lineal en y (grado 1 respecto a y)
    try:
        poli_y = sp.Poly(expr, [yp, y])
        if poli_y.degree(yp) == 1 and poli_y.degree(y) == 1:
            categorias.append("Lineal de primer orden")
    except (sp.PolynomialError, sp.GeneratorsError):
        pass

    # 4. Bernoulli: y' + P(x)*y = Q(x)*y^n
    # Si no es lineal pero se puede factorizar y' + P(x)y
    # Se evalúa analizando si f(x, y) = -P(x)y + Q(x)y^n
    n = sp.Symbol('n')
    # Chequeo simplificado de Bernoulli:
    if "Lineal de primer orden" not in categorias:
        terminos = sp.Add.make_args(f)
        if len(terminos) == 2:
            categorias.append("Candidata a Bernoulli (verificar potencia n)")

    # 5. Exacta: M(x, y) dx + N(x, y) dy = 0  =>  dM/dy == dN/dx
    # Como y' = -M/N, podemos tomar M = -numerador y N = denominador
    num, den = sp.fraction(f)
    M = -num
    N = den
    if sp.simplify(sp.diff(M, y) - sp.diff(N, x)) == 0:
        categorias.append("Exacta")

    return categorias if categorias else ["Ecuación de 1er orden no catalogada"]


def clasificador_edo(entrada_usuario: str, orden_max: int = 1, grado_max: int = 1):
    """
    Función principal de ejecución y validación.
    """
    try:
        eq = interpretar_entrada(entrada_usuario)
    except Exception as e:
        return f"Error de sintaxis simbólica: {e}"

    orden, grado, expr = obtener_orden_y_grado(eq)

    print(f"\n--- Análisis de la EDO ---")
    print(f"Ecuación interpretada: {eq.lhs} = 0")
    print(f"Orden: {orden}")
    print(f"Grado: {grado if grado is not None else 'No definido'}")

    # Rechazar si supera los límites establecidos
    if orden > orden_max:
        return f"[RECHAZADO] La ecuación es de orden {orden}. Límite permitido: {orden_max}."
    if grado is not None and grado > grado_max:
        return f"[RECHAZADO] La ecuación es de grado {grado}. Límite permitido: {grado_max}."

    # Enrutamiento al temario
    if orden == 1:
        categorias = clasificar_primer_orden(expr)
        return f"Categoría detectada: {', '.join(categorias)}"
    else:
        return "No es una ecuación diferencial ordinaria válida."


# --- Bloque de prueba interactiva ---
if __name__ == "__main__":
    casos = [
        "y' = x * y",                       # Separable, Lineal
        "x*y' + y = x**2",                  # Lineal de primer orden
        "y' = (x**2 + y**2) / (x*y)",       # Homogénea
        "(2*x*y) + (x**2 + 1)*y' = 0",      # Exacta
        "(y')**2 + y = x",                  # Grado 2 (debe ser rechazada si grado_max=1)
        "y'' + 3*y' + 2*y = 0"              # Orden 2 (debe ser rechazada si orden_max=1)
    ]

    for caso in casos:
        print(f"\nEntrada: {caso}")
        resultado = clasificador_edo(caso, orden_max=1, grado_max=1)
        print(f"Resultado: {resultado}")