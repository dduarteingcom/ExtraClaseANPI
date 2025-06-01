import sympy as sp
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------------
# Función exacta: f(x) = sin(6 - x) / (sin(5) * sqrt(x))
# ---------------------------------------------------------------
def y_exacta(x):
    """
    Función exacta.
    f(x) = sin(6 - x) / (sin(5) * sqrt(x))

    Args:
        x (float o np.array): Punto(s) donde se evalúa la función.

    Returns:
        float o np.array: Valor(es) de f(x).
    """
    return np.sin(6 - x) / (np.sin(5) * np.sqrt(x))

# ---------------------------------------------------------------
# Kernel de Lagrange L_k(x): producto de los factores de interpolación
# Se utiliza para construir el polinomio de Lagrange
# ---------------------------------------------------------------
def Lk(x, k, x_val):
    """
    Calcula el kernel del polinomio de interpolación L_k(x).

    Args:
        x (sympy.Symbol): Variable simbólica.
        k (int): Índice.
        x_val (np.array): x_i.

    Returns:
         Expresión simbólica de L_k(x).
    """
    L_k = 1
    xk = x_val[k]
    for j in range(len(x_val)):
        if j != k:
            L_k *= (x - x_val[j]) / (xk - x_val[j])
    return L_k

# ---------------------------------------------------------------
# Construcción del polinomio de interpolación de Lagrange
# P(x) = sum_{k=0}^{n} y_k * L_k(x)
# ---------------------------------------------------------------
def polinomio_lagrange(x_val, y_val):
    """
    Calcula el polinomio de interpolación de Lagrange.

    Args:
        x_val ( np.array): x_i.
        y_val (or np.array): y_i = f(x_i).

    Returns:
        Polinomio simbólico de interpolación P(x).
    """
    x = sp.Symbol('x')
    p_x = 0
    for k in range(len(x_val)):
        p_x += y_val[k] * Lk(x, k, x_val)
    return sp.simplify(p_x)

# ---------------------------------------------------------------
# Datos obtenidos con h=1
# ---------------------------------------------------------------
x_val = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
y_val = [1.00000000, 0.69696245, -0.00758192, -0.50450963, -0.44956304, 0.00000000]

# ---------------------------------------------------------------
# Cálculo del polinomio de interpolación simbólico de Lagrange
# ---------------------------------------------------------------
P_sym = polinomio_lagrange(x_val, y_val)

# Imprimir el polinomio simbólico simplificado
print("\nPolinomio de interpolación de Lagrange simplificado:")
print("P(x) =", str(P_sym))

# ---------------------------------------------------------------
# Conversión del polinomio simbólico a función numérica evaluable
# ---------------------------------------------------------------
Px = sp.lambdify(sp.Symbol('x'), P_sym, 'numpy')

# ---------------------------------------------------------------
# Gráfico: función exacta, puntos dados e interpolación
# ---------------------------------------------------------------
x_plot = np.linspace(1, 6, 800)

plt.figure(figsize=(9, 6))

# Curva del polinomio de Lagrange
plt.plot(x_plot, Px(x_plot), label='Polinomio de Lagrange', color='blue')

# Puntos dados
plt.plot(x_val, y_val, 'ko', label='Puntos dados (h = 1)', markersize=6)

# Curva de la función exacta
plt.plot(x_plot, y_exacta(x_plot), '--', label='f(x) exacta', linewidth=2, color='green')

# Configuración del gráfico
plt.title("Interpolación de Lagrange vs. función exacta")
plt.xlabel("x")
plt.ylabel("y(x)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
