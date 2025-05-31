import numpy as np
import matplotlib.pyplot as plt

def extract_diagonal(matrix, diag_index):
    return np.diag(matrix, k=diag_index)

def thomas(A, b):
    """
    Resuelve un sistema tridiagonal de ecuaciones lineales usando el algoritmo de Thomas.
    
    Args:
        A (np.array): Matriz tridiagonal del sistema (n x n).
        b (np.array): Vector del lado derecho (n).
    
    Returns:
        np.array: Solución del sistema (vector x).
    """
    n = len(b) # Tamaño del sistema

    a = extract_diagonal(A, -1) # Subdiagonal
    c = extract_diagonal(A,  1)  # Superdiagonal
    d = b.copy()                    
    b = extract_diagonal(A, 0)   # Diagonal principal

    p = np.zeros(n-1) # Coeficientes modificados
    q = np.zeros(n) # Términos independientes modificados

    p[0] = c[0] / b[0]
    q[0] = d[0] / b[0]

    for i in range(1, n-1):
        denom = b[i] - a[i-1]*p[i-1]
        p[i]  = c[i] / denom
        q[i]  = (d[i] - a[i-1]*q[i-1]) / denom

    denom  = b[n-1] - a[n-2]*p[n-2]
    q[n-1] = (d[n-1] - a[n-2]*q[n-2]) / denom

    x = np.zeros(n)
    x[n-1] = q[n-1]
    for i in range(n-2, -1, -1):
        x[i] = q[i] - p[i]*x[i+1]
    return x.flatten()  # vector 1-D

#Ecuaciones
def p(x): 
    return -1/x
def q(x): 
    return 1/(4*x**2) - 1
def r(x): 
    return 0


def edo2(p, q, r, h, a, b, y0, y_n):
    # Número de puntos intermedios
    n  = int(round((b-a)/h))
    #Vector incluyendo extremos
    x  = np.linspace(a, b, n+1)
    # Inicialización de vector solución
    y  = np.zeros(n+1)
    y[0], y[-1] = y0, y_n
     # Matriz tridiagonal
    A     = np.zeros((n-1, n-1))
    b_vec = np.zeros(n-1)

    for j in range(1, n):
        pj, qj, rj = p(x[j]), q(x[j]), r(x[j])
        c1 = -(h/2)*pj - 1          # coef. y_{j-1}
        c2 =  2 + h**2 * qj         # coef. y_j
        c3 =  (h/2)*pj - 1          # coef. y_{j+1}

        if j == 1:     c1_first = c1
        if j == n-1:   c3_last  = c3

        if j > 1:      A[j-1, j-2] = c1
        A[j-1, j-1] = c2
        if j < n-1:    A[j-1, j]   = c3

        b_vec[j-1] = -h**2 * rj     # lado derecho

    b_vec[0]  -= c1_first * y0       # incorporar condiciones de frontera
    b_vec[-1] -= c3_last  * y_n

    y[1:-1] = thomas(A, b_vec)      # resolver parte interna
    return x, y



# Solución exacta que EL ENUNCIADO provee
def y_exacta(x):
    return np.sin(6 - x) / (np.sin(5) * np.sqrt(x))

# ------------------  PRUEBA Y GRÁFICA FINAL  -----------------------
h_vals = [1, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01]

plt.figure(figsize=(9,6))
for h in h_vals:
    xs, ys = edo2(p, q, r, h, a=1, b=6, y0=1, y_n=0)
    plt.plot(xs, ys, label=f"Aprox.  h = {h}")
    if h == 1:
        print("\nValores de x e y cuando h = 1:\n")
        for xi, yi in zip(xs, ys):
            print(f"x = {xi:.1f}   y = {yi:.8f}")


x_fine = np.linspace(1, 6, 1200)
plt.plot(x_fine, y_exacta(x_fine), 'k--', lw=2, label='Solución exacta')

plt.title("Método de diferencias finitas vs. solución exacta")
plt.xlabel("x"); plt.ylabel("y(x)")
plt.legend(); plt.grid(True); plt.tight_layout(); plt.show()
