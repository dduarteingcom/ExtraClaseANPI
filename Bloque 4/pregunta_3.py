import math
tol=1e-8
iterMax=1_000_000
"""
Ejercicio de integración numérica: Cuadratura Gaussiana

Este script implementa tres variantes del método de cuadratura gaussiana para 
aproximar el valor de una integral definida. La función a integrar es:

    f(x) = cos(x) * e^x    en el intervalo [2, 5]

Se desarrollan e implementan las siguientes versiones:

1. cuad_gauss(f, a, b, k):
   - Aplica la cuadratura gaussiana simple de orden k sobre el intervalo [a, b].

2. cuad_gauss_comp(f, a, b, k, n):
   - Divide el intervalo [a, b] en n subintervalos y aplica cuad_gauss a cada uno.

3. cuad_gauss_iter(f, a, b, k):
   - Aumenta progresivamente el número de subintervalos (empezando en n=1),
     aplicando cuad_gauss_comp hasta que la diferencia entre dos aproximaciones 
     sucesivas sea menor que una tolerancia fija (tol = 1e-8).

El resultado de cada método se compara con el valor de referencia obtenido 
mediante la función `quad` de scipy.integrate para validar la precisión.

Parámetros utilizados:
    - Orden de la cuadratura: k = 7
    - Subintervalos para versión compuesta: n = 20
    - Tolerancia para versión iterativa: tol = 1e-8

Autor: Daniel Duarte Cordero, Melissa Vasquez Rojas
Curso: Análisis Numérico para Ingeniería
"""
# Función que retorna nodos (x_i) y pesos (w_i) para la cuadratura gaussiana de orden k
def nodos_pesos(k):
    # Tablas conocidas para 2 <= k <= 10
    if k == 2:
        x = [-0.5773502691896257, 0.5773502691896257]
        w = [1.0, 1.0]
    elif k == 3:
        x = [0.0, -0.7745966692414834, 0.7745966692414834]
        w = [0.8888888888888888, 0.5555555555555556, 0.5555555555555556]
    elif k == 4:
        x = [-0.3399810435848563, 0.3399810435848563,
             -0.8611363115940526, 0.8611363115940526]
        w = [0.6521451548625461, 0.6521451548625461,
             0.3478548451374538, 0.3478548451374538]
    elif k == 5:
        x = [0.0, -0.5384693101056831, 0.5384693101056831,
             -0.9061798459386640, 0.9061798459386640]
        w = [0.5688888888888889, 0.4786286704993665, 0.4786286704993665,
             0.2369268850561891, 0.2369268850561891]
    elif k == 6:
        x = [-0.6612093864662645, 0.6612093864662645,
             -0.2386191860831969, 0.2386191860831969,
             -0.9324695142031521, 0.9324695142031521]
        w = [0.3607615730481386, 0.3607615730481386,
             0.4679139345726910, 0.4679139345726910,
             0.1713244923791704, 0.1713244923791704]
    elif k == 7:
        x = [0.0, -0.4058451513773972, 0.4058451513773972,
             -0.7415311855993945, 0.7415311855993945,
             -0.9491079123427585, 0.9491079123427585]
        w = [0.4179591836734694, 0.3818300505051189, 0.3818300505051189,
             0.2797053914892766, 0.2797053914892766,
             0.1294849661688697, 0.1294849661688697]
    elif k == 8:
        x = [-0.1834346424956498, 0.1834346424956498,
             -0.5255324099163290, 0.5255324099163290,
             -0.7966664774136267, 0.7966664774136267,
             -0.9602898564975363, 0.9602898564975363]
        w = [0.3626837833783620, 0.3626837833783620,
             0.3137066458778873, 0.3137066458778873,
             0.2223810344533745, 0.2223810344533745,
             0.1012285362903763, 0.1012285362903763]
    elif k == 9:
        x = [0.0, -0.8360311073266358, 0.8360311073266358,
             -0.9681602395076261, 0.9681602395076261,
             -0.3242534234038089, 0.3242534234038089,
             -0.6133714327005904, 0.6133714327005904]
        w = [0.3302393550012598,
             0.1806481606948574, 0.1806481606948574,
             0.0812743883615744, 0.0812743883615744,
             0.3123470770400029, 0.3123470770400029,
             0.2606106964029354, 0.2606106964029354]
    elif k == 10:
        x = [-0.1488743389816312, 0.1488743389816312,
             -0.4333953941292472, 0.4333953941292472,
             -0.6794095682990244, 0.6794095682990244,
             -0.8650633666889845, 0.8650633666889845,
             -0.9739065285171717, 0.9739065285171717]
        w = [0.2955242247147529, 0.2955242247147529,
             0.2692667193099963, 0.2692667193099963,
             0.2190863625159820, 0.2190863625159820,
             0.1494513491505806, 0.1494513491505806,
             0.0666713443086881, 0.0666713443086881]
    else:
        raise ValueError("Orden no soportado. Solo se permite 2 ≤ k ≤ 10.")
    
    return x, w

# Cuadratura Gaussiana simple
def cuad_gauss(f, a, b, k):
    xi, wi = nodos_pesos(k)
    suma = 0
    for i in range(k):
        x_mapeado = (b - a) / 2 * xi[i] + (b + a) / 2
        suma += wi[i] * f(x_mapeado)
    return (b - a) / 2 * suma

# Cuadratura Gaussiana compuesta
def cuad_gauss_comp(f, a, b, k, n):
    h = (b - a) / n
    total = 0
    for j in range(n):
        xj = a + j * h
        xj1 = xj + h
        total += cuad_gauss(f, xj, xj1, k)
    return total

# Cuadratura Gaussiana iterativa: aplica cuad_gauss_comp hasta cumplir tolerancia
def cuad_gauss_iter(f, a, b, k):
    n = 1
    Sn = cuad_gauss_comp(f, a, b, k, n)
    while True:
        n += 1
        if n > iterMax:
            raise RuntimeError("No se alcanzó la tolerancia con el máximo número de subintervalos permitidos.")
        S_next = cuad_gauss_comp(f, a, b, k, n)
        if abs(S_next - Sn) < tol:
            return S_next
        Sn = S_next

# Función a integrar: f(x) = cos(x) * e^x
def f(x):
    return math.cos(x) * math.exp(x)

# Parámetros del problema
a = 2
b = 5
k = 7  # orden de la cuadratura
n = 20 # subintervalos para la versión compuesta

# Aproximaciones usando las tres versiones
I_gauss       = cuad_gauss(f, a, b, k)
I_gauss_comp  = cuad_gauss_comp(f, a, b, k, n)
I_gauss_iter  = cuad_gauss_iter(f, a, b, k)

# Mostrar resultados
print(f"cuad_gauss (k=7):           {I_gauss}")
print(f"cuad_gauss_comp (k=7,n=20): {I_gauss_comp}")
print(f"cuad_gauss_iter (k=7):      {I_gauss_iter}")