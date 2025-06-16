%Estudiantes: Ana Melissa Vásquez Rojas, Daniel Duarte Cordero
%Descripción: Cálculo del valor mínimo de n (par) tal que la cota del error
%del método compuesto de Simpson sea menor que la tolerancia indicada.

function pregunta2()
  clc; clear;
  pkg load symbolic
% Definir simbólicamente la función f(x)
  syms x
  f = exp(x)*(26 - 10*x + x^2);  % Función dada
% Intervalo de integración
  a = 5;
  b = 5.5;
% Tolerancia
  tol = 1e-8;
% Calcular el valor mínimo de n
  n = cota_simpson_puntos(f, a, b, tol);
  fprintf('--- Método de Simpson Compuesto ---\n');
  fprintf('Valor de n: %d\n', n);
endfunction
% -------------------------------------------------------------------
% Función: cota_simpson_puntos
% Encuentra el valor mínimo de n (par) tal que la cota del error del
% método compuesto de Simpson sea menor que la tolerancia dada
function n = cota_simpson_puntos(f, a, b, tol)
  syms x
% Cuarta derivada simbólica
  f4 = diff(f, x, 4);
  f4_abs =abs(f4);
  f4_func = matlabFunction(f4_abs); % Convertir a función evaluable

% Encontrar máximo valor absoluto en [a, b]
  xmax = fminbnd(@(x) -f4_func(x), a, b); % Negamos "-f4_func(x)" para encontrar el máximo con fminbnd
  alpha_max = abs(f4_func(xmax));

% Buscar el menor n par que cumpla la condición
  n = 2;
  h = (b - a) / n;
  cota = ((b - a) * h^4 / 180) * alpha_max;

  while cota >= tol
    n += 2;
    h =(b - a) / n;
    cota = ((b - a) * h^4 / 180) * alpha_max;
  endwhile
endfunction

