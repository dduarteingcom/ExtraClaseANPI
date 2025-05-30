import numpy as np
import matplotlib.pyplot as plt

def f(x, y):
    return (x + y) / x

def runge_kutta_6(a, b, y0, m):
    h = (b - a) / (m - 1)
    x_points = np.linspace(a, b, m)
    y_points = np.zeros(m)
    y_points[0] = y0

    for n in range(1, m):
        x_old = x_points[n - 1]
        y_old = y_points[n - 1]

        k1 = h * f(x_old, y_old)
        k2 = h * f(x_old + h/3, y_old + k1/3)
        k3 = h * f(x_old + 2*h/5, y_old + (4*k1 + 6*k2)/25)
        k4 = h * f(x_old + h, y_old + (k1 - 12*k2 + 15*k3)/4)
        k5 = h * f(x_old + 2*h/3, y_old + (6*k1 + 90*k2 - 50*k3 + 8*k4)/81)
        k6 = h * f(x_old + 4*h/5, y_old + (6*k1 + 36*k2 + 10*k3 + 8*k4)/75)

        y_points[n] = y_old + (23*k1 + 125*k2 - 81*k5 + 125*k6)/192

    return x_points, y_points

def exact_solution(x):
    return x * np.log(x / 2) + 2 * x

if __name__ == "__main__":
    m_values = [10, 20, 50, 100, 250]
    solutions = [runge_kutta_6(2, 10, 4, m) for m in m_values]
    x_exact = np.linspace(2, 10, 1000)
    y_exact = exact_solution(x_exact)

    for (x, y), m in zip(solutions, m_values):
        plt.plot(x, y, label=f'RK6 m={m}')

    plt.plot(x_exact, y_exact, 'k--', label='Solución exacta')
    plt.title("Runge-Kutta Orden 6 vs Solución Exacta")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
