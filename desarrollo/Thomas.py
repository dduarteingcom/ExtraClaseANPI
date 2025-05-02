import numpy as np

def extract_diagonal(matrix, diag_index):
    """Extrae una diagonal específica de una matriz.
    
    Args:
        matrix (np.array): Matriz de entrada.
        diag_index (int): Índice de la diagonal (0: principal, 1: superior, -1: inferior, etc.).
    
    Returns:
        np.array: Vector con los elementos de la diagonal solicitada.
    """
    return np.diag(matrix, k=diag_index)


def thomas(A, b):
    """Resuelve un sistema tridiagonal de ecuaciones lineales usando el algoritmo de Thomas.
    
    Args:
        A (np.array): Matriz tridiagonal del sistema (n x n).
        b (np.array): Vector del lado derecho (n).
    
    Returns:
        np.array: Solución del sistema (vector x).
    """
    n = len(b)  # Tamaño del sistema (número de ecuaciones)

    # Extracción de las diagonales:
    a = extract_diagonal(A, -1)  # Subdiagonal (elementos bajo la diagonal principal)
    c = extract_diagonal(A, 1)    # Superdiagonal (elementos sobre la diagonal principal)
    d = b.copy()                 # Copia del vector del lado derecho (para no modificar el original)
    b = extract_diagonal(A, 0)    # Diagonal principal (sobrescribe b, ahora es la diagonal)

    # Inicialización de vectores para el algoritmo:
    p = np.zeros(n-1)  # Coeficientes modificados (forward sweep)
    q = np.zeros(n)    # Términos independientes modificados (forward sweep)

    # Paso 1: Forward sweep (cálculo de p y q)
    # Primer elemento (caso especial i=0)
    p[0] = c[0] / b[0]
    q[0] = d[0] / b[0]

    # Elementos intermedios (i=1 hasta i=n-2)
    for i in range(1, n-1):
        denom = b[i] - a[i-1] * p[i-1]  # Denominador común
        p[i] = c[i] / denom
        q[i] = (d[i] - a[i-1] * q[i-1]) / denom

    # Último elemento (i=n-1)
    denom = b[n-1] - a[n-2] * p[n-2]
    q[n-1] = (d[n-1] - a[n-2] * q[n-2]) / denom

    # Paso 2: Backward substitution (solución hacia atrás)
    x = np.zeros(n)
    x[n-1] = q[n-1]  # Último valor de x (xn = qn)

    # Iteramos desde el penúltimo elemento hasta el primero (i=n-2 hasta i=0)
    for i in range(n-2, -1, -1):
        x[i] = q[i] - p[i] * x[i+1]
    return x

A = np.array([
    [2, -1,  0,  0],
    [-1, 2, -1,  0],
    [0, -1, 2, -1],
    [0,  0, -1, 2]
], dtype=float)

b = np.array([1, 0, 0, 1], dtype=float)

# Resolver el sistema
x = thomas(A, b)
print("Solución:", x) 