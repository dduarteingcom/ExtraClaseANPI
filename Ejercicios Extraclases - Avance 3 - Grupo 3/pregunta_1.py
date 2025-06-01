import numpy as np
"""
Estudiantes:
Ana Melissa Vásquez Rojas
Daniel Duarte Cordero
"""

"""
Este script implementa y compara el método de Runge-Kutta de 6to orden para resolver el problema de valor inicial (PVI) de una ecuación diferencial ordinaria (EDO):
    dy/dx = (x + y) / x
con la condición inicial y(2) = 4, para el intervalo [2, 10].
Funciones:
----------
f(x, y):
    Evalúa el lado derecho de la EDO: (x + y) / x.
runge_kutta_6(a, b, y0, m):
    Numéricamente resuelve la EDO usando el método de 6to orden Runge-Kutta.
        Parametros:
        a (float): Valor inicial de x.
        b (float): Valor final de x.
        y0 (float): Valor inicial de y en x = a.
        m (int): Número de puntos con la discretización.
    Returns:
        x_points (np.ndarray): Arreglo de valores de x.
        y_points (np.ndarray): Array de valores aproximados de x.
exact_solution(x):
    Calcula la solución analítica de la EDO comparando:
        y(x) = x * ln(x / 2) + 2 * x
A nivel general:
---------------
- Resuelve la EDO para diferentes valores de m.
- Grafica las soluciones numéricas obtenidas con el método Runge-Kutta de 6to orden.
- Grafica la solución analítica exacta para compararla con la solución obtenida.
- En la gráfica se identifican las etiquetas para las funciones correspondientes a los diferentes valores de m.
"""
import matplotlib.pyplot as plt
"""
Estudiantes:
Ana Melissa Vásquez Rojas
Daniel Duarte Cordero
"""

def f(x, y):
    return (x + y) / x
# Implementación del método de Runge-Kutta de orden 6
def runge_kutta_6(a, b, y0, m):
    h = (b - a) / (m - 1)
    x_points = np.linspace(a, b, m)
    y_points = np.zeros(m)
    y_points[0] = y0

    for n in range(1, m):
        
        x_old = x_points[n - 1]
        y_old = y_points[n - 1]
        # Cálculo de los coeficientes intermedios
        k1 = h * f(x_old, y_old)
        k2 = h * f(x_old + h/3, y_old + k1/3)
        k3 = h * f(x_old + 2*h/5, y_old + (4*k1 + 6*k2)/25)
        k4 = h * f(x_old + h, y_old + (k1 - 12*k2 + 15*k3)/4)
        k5 = h * f(x_old + 2*h/3, y_old + (6*k1 + 90*k2 - 50*k3 + 8*k4)/81)
        k6 = h * f(x_old + 4*h/5, y_old + (6*k1 + 36*k2 + 10*k3 + 8*k4)/75)
         # Actualización del valor de y según la fórmula del método RK6
        y_points[n] = y_old + (23*k1 + 125*k2 - 81*k5 + 125*k6)/192

    return x_points, y_points
# Solución exacta
def exact_solution(x):
    return x * np.log(x / 2) + 2 * x

if __name__ == "__main__":

    m_values = [10, 20, 50, 100, 250]
    solutions = [runge_kutta_6(2, 10, 4, m) for m in m_values]
    # Solución exacta evaluada en muchos puntos
    x_exact = np.linspace(2, 10, 1000)
    y_exact = exact_solution(x_exact)
    # Graficar cada solución numérica obtenida
    for (x, y), m in zip(solutions, m_values):
        plt.plot(x, y, label=f'RK6 m={m}')
    # Graficar la solución analítica
    plt.plot(x_exact, y_exact, 'k--', label='Solución exacta')
    plt.title("Runge-Kutta 6to orden vs Solución Exacta")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
