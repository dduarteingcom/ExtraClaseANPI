import numpy as np

def extract_diagonal(matrix, diag_index):
    #Extrae una diagonal específica de una matriz.
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
    return x.reshape(-1, 1)

# Función para generar la matriz tridiagonal y el vector b solicitada
def matrix_generator():
    A = np.zeros((100,100))
    A[0,0] = 5
    A[0,1] = 1

    for k in range(1,99):
        A[k,k]= 5
        A[k,k-1] = 1
        A[k,k+1] = 1
    
    A[99,99] = 5 
    A[99,98] = 1
    b = np.zeros((100,1))
    b[0] = -12
    b[-1] = -12
    
    for j in range(1,99):
        b[j] = -14
    return A, b

# Resolver el sistema
A, b = matrix_generator()
x = thomas(A, b)
print("Solucion:", x) 