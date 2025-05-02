import numpy as np

def theta(A, i, j):
    a_ij = A[i, j]
    denominator = A[i, i] - A[j, j]
    if abs(denominator) > 1e-16:
        return 0.5 * np.arctan(2 * a_ij / denominator)
    else:
        return 0.0
        

def matriz_rotacion(i, j, m, theta):
    I = np.eye(m)
    Z = np.zeros((m, m))
    c = np.cos(theta)
    s = np.sin(theta)

    Z[i, i] = c - 1
    Z[j, j] = c - 1

    if i != j:
        Z[i, j] = -s
        Z[j, i] = s

    G = I + Z
    return G

def jacobi_valores_propios(A, iterMax, tol):
    A0 = A.copy()
    m = A0.shape[0]
    Ak = A0.copy()
    xk = np.diag(Ak)
    for _ in range(iterMax):
        Bk = Ak.copy()
        for i in range(m):
            for j in range(m):
                if i >= j:  # Evitar procesar pares redundantes
                    continue
                theta_ij = theta(Bk, i, j)
                G = matriz_rotacion(i, j, m, theta_ij)
                Bk = G.T @ Bk @ G
        Ak_next = Bk
        xk_next = np.diag(Ak_next)
        ek = np.linalg.norm(xk_next - xk)
        if ek < tol:
            break
        Ak = Ak_next
        xk = xk_next
    return np.diag(Ak)

def prueba_jacobi():
    A = np.zeros((15,15))
    for i in range(15):
        for j in range(15):
            A[i,j] = 0.5 * ((i+1)+ (j+1))
    valores_propios = jacobi_valores_propios(A, iterMax=200, tol=1e-10)
    print("Valores propios calculados :", np.sort(valores_propios))

prueba_jacobi()