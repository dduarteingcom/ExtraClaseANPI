import numpy as np
import sympy as sp
import matplotlib.pyplot as plt


def thomas(A, d):
    """
    Resuelve un sistema tridiagonal de ecuaciones lineales usando el metodo de Thomas.

    Args:
        A (np.array): Matriz tridiagonal del sistema (n x n).
        d (np.array): Vector del lado derecho (n).

    Returns:
        np.array: Solución del sistema (vector x).
    """
    n = len(d)
    a = np.diag(A, -1)  # Subdiagonal
    b = np.diag(A, 0)  # Diagonal principal
    c = np.diag(A, 1)  # Superdiagonal

    p = np.zeros(n - 1)
    q = np.zeros(n)

    p[0] = c[0] / b[0]
    q[0] = d[0] / b[0]

    for i in range(1, n - 1):
        denom = b[i] - a[i - 1] * p[i - 1]
        p[i] = c[i] / denom
        q[i] = (d[i] - a[i - 1] * q[i - 1]) / denom

    q[n - 1] = (d[n - 1] - a[n - 2] * q[n - 2]) / (b[n - 1] - a[n - 2] * p[n - 2])

    x = np.zeros(n)
    x[n - 1] = q[n - 1]
    for i in range(n - 2, -1, -1):
        x[i] = q[i] - p[i] * x[i + 1]

    return x


def trazador_cubico(x_val, y_val):
    """
    Calcula los polinomios de interpolación por trazadores cúbicos.

    Args:
        x_val (np.array): Coordenadas x de los puntos de interpolación.
        y_val (np.array): Coordenadas y de los puntos de interpolación.

    Returns:
        tuple: Lista de funciones simbólicas de los trazadores y el símbolo simbólico x.
    """
    n = len(x_val)

    # -------------------------------------------------------------
    # Paso 1: Calcular h_j = x_{j+1} - x_j para j = 0,...,n-2
    # -------------------------------------------------------------
    h = np.zeros(n - 1)
    for j in range(n - 1):
        h[j] = x_val[j + 1] - x_val[j]

    # -------------------------------------------------------------
    # Paso 2: Construcción del sistema lineal A·m = u
    # A es la matriz tridiagonal y u el vector del lado derecho
    # -------------------------------------------------------------
    A = np.zeros((n - 2, n - 2))    # Matriz tridiagonal de tamaño (n-2) x (n-2)
    u = np.zeros(n - 2)          # Vector lado derecho con valores u_j

    for j in range(n - 2):
        # Diagonal principal
        A[j, j] = 2 * (h[j] + h[j + 1])
        # Subdiagonal
        if j > 0:
            A[j, j - 1] = h[j]
        # Diagonal superior
        if j < n - 3:
            A[j, j + 1] = h[j + 1]
        # Vector u
        u[j] = 6 * ((y_val[j + 2] - y_val[j + 1]) / h[j + 1] - (y_val[j + 1] - y_val[j]) / h[j])

    # -------------------------------------------------------------
    # Paso 2.5: Resolver el sistema tridiagonal con el metodo de Thomas
    # m_0 = m_n = 0
    # -------------------------------------------------------------
    m_interna = thomas(A, u)  # Soluciona el sistema para m_1 a m_{n-1}
    m = np.zeros(n)
    for j in range(1, n - 1):
        m[j] = m_interna[j - 1]
    # m[0] y m[n-1] ya están en cero por condiciones naturales

    # -------------------------------------------------------------
    # Paso 3: Calcular coeficientes a_j, b_j, c_j, d_j para j = 0,...,n-2
    # -------------------------------------------------------------
    a = np.zeros(n - 1)
    b = np.zeros(n - 1)
    c = np.zeros(n - 1)
    d = np.zeros(n - 1)

    for j in range(n - 1):
        a[j] = (m[j + 1] - m[j]) / (6 * h[j])
        b[j] = m[j] / 2
        c[j] = (y_val[j + 1] - y_val[j]) / h[j] - (h[j] / 6) * (2 * m[j] + m[j + 1])
        d[j] = y_val[j]

    # -------------------------------------------------------------
    # Paso 4: Construcción simbólica de cada trazador cúbico s_j(x)
    # s_j(x) = a_j*(x - x_j)^3 + b_j*(x - x_j)^2 + c_j*(x - x_j) + d_j
    # -------------------------------------------------------------
    x = sp.Symbol('x')
    funciones_trazador = [0] * (n - 1)

    for j in range(n - 1):
        s_j = a[j] * (x - x_val[j]) ** 3 + b[j] * (x - x_val[j]) ** 2 + c[j] * (x - x_val[j]) + d[j]
        funciones_trazador[j] = sp.expand(sp.simplify(s_j))  # Se simplifica y expande

    return funciones_trazador, x



def f_original(x):
    """
    Función original f(x) = sin(6 - x) / (sin(5) * sqrt(x))

    Args:
        x (float or np.array): Valor(es) donde evaluar la función.

    Returns:
        float or np.array: Valor evaluado.
    """
    return np.sin(6 - x) / (np.sin(5) * np.sqrt(x))


if __name__ == "__main__":
    # Puntos dados
    x_val = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    y_val = np.array([1.0, 0.69696245, -0.00758192, -0.50450963, -0.44956304, 0.0])

    # Obtener funciones simbólicas de los trazadores
    trazador, x_sym = trazador_cubico(x_val, y_val)

    # Imprimir funciones simbólicas
    print("Trazadores cúbicos por tramos:")
    for j in range(len(trazador)):
        print(f"s_{j}(x) = {trazador[j]}")

    # Graficar función original
    x_plot = np.linspace(x_val[0], x_val[-1], 1000)
    y_real = f_original(x_plot)

    plt.figure(figsize=(8, 6))
    plt.plot(x_plot, y_real, 'k--', label="Función original")

    # Graficar cada trazador en su intervalo
    for j in range(len(trazador)):
        f_j = sp.lambdify(x_sym, trazador[j], modules='numpy')
        x_l= x_plot[(x_plot >= x_val[j]) & (x_plot <= x_val[j + 1])]
        y_l = f_j(x_l)
        plt.plot(x_l, y_l, label=f"Trazador s_{j}(x)")

    plt.plot(x_val, y_val, 'ro', label="Puntos dados")
    plt.title("Interpolación por Trazadores Cúbicos")
    plt.xlabel("x")
    plt.ylabel("y(x)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
