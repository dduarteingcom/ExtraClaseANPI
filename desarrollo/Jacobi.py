import numpy as np

# Calcula el ángulo θ necesario para anular el elemento A[i, j] mediante rotación
def theta(A, i, j):
    a_ij = A[i, j]
    denominator = A[i, i] - A[j, j]
    # Evita divisiones por cero o valores numéricamente inestables
    if abs(denominator) > 1e-16:
        return 0.5 * np.arctan(2 * a_ij / denominator)
    else:
        return 0.0

# Construye la matriz de rotación G que afecta solo a las filas/columnas i y j
def matriz_rotacion(i, j, m, theta):
    I = np.eye(m)       # Matriz identidad m x m
    Z = np.zeros((m, m))  # Matriz de ajuste para insertar coseno/seno

    c = np.cos(theta)
    s = np.sin(theta)

    # Ajustes diagonales
    Z[i, i] = c - 1
    Z[j, j] = c - 1

    # Ajustes fuera de la diagonal para los índices i y j
    if i != j:
        Z[i, j] = -s
        Z[j, i] = s

    G = I + Z  # Matriz de rotación G = I + Z
    return G

# Método de Jacobi para calcular los valores propios de una matriz simétrica
def jacobi_valores_propios(A, iterMax, tol):
    A0 = A.copy()
    m = A0.shape[0]
    Ak = A0.copy()
    xk = np.diag(Ak)  # Inicializa con los elementos diagonales (estimación inicial de valores propios)

    for _ in range(iterMax):
        Bk = Ak.copy()

        # Aplicar una rotación a cada par (i, j)
        for i in range(m):
            for j in range(m):
                if i >= j:  # Evita duplicar o repetir rotaciones simétricas
                    continue
                theta_ij = theta(Bk, i, j)        # Calcula el ángulo óptimo
                G = matriz_rotacion(i, j, m, theta_ij)  # Genera la matriz de rotación
                Bk = G.T @ Bk @ G                 # Aplica la transformación de similitud

        Ak_next = Bk
        xk_next = np.diag(Ak_next)               # Extrae la nueva estimación de los valores propios
        ek = np.linalg.norm(xk_next - xk)        # Criterio de convergencia: cambio entre iteraciones

        if ek < tol:
            break

        Ak = Ak_next
        xk = xk_next

    return np.diag(Ak)  # Retorna los valores propios aproximados

# Prueba del algoritmo con una matriz definida de 15x15
def prueba_jacobi():
    A = np.zeros((15,15))
    for i in range(15):
        for j in range(15):
            A[i,j] = 0.5 * ((i+1) + (j+1))  # Elementos A_ij = 0.5 * (i + j + 2)

    valores_propios = jacobi_valores_propios(A, iterMax=1000, tol=1e-10)
    print("Valores propios calculados :", np.sort(valores_propios))

# Ejecuta la prueba
prueba_jacobi()
