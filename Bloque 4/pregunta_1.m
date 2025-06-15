%Estudiantes: Ana Melissa Vásquez Rojas, Daniel Duarte Cordero
%Descripción: Cálculo de las cotas de error para la regla del trapecio y Simpson
function pregunta1()
  clc; clear;
  pkg load symbolic

  % Definir simbólicamente la función f(x)
  syms x
  f = log(asin(x)) / log(x);

  % Definir intervalo de integración
  a = 0.2;
  b = 0.8;

  % ------------------- Regla del trapecio -------------------
  E_trapecio = cota_error_trapecio(f, a, b);

  fprintf('--- Regla del Trapecio ---\n');
  fprintf('Cota de error: %.10e\n\n', E_trapecio);

% -------------------------------------------------------------------
% Función: cota_error_trapecio
% Calcula la cota de error del método del trapecio
function E = cota_error_trapecio(f, a, b)
  syms x
  f2 = diff(f, x, 2);             % Segunda derivada simbólica
  f2_abs = abs(f2);
  f2_func = matlabFunction(f2_abs);
  [xmax, ~] = fminbnd(@(x) -f2_func(x), a, b); % Negamos "-f2_func(x)" para encontrar el máximo con fminbnd
  alpha_max = abs(f2_func(xmax));
  E = ((b - a)^3 / 12) * alpha_max;
endfunction

  % ------------------- Regla de Simpson -------------------
  E_simpson = cota_error_simpson(f, a, b);

  fprintf('--- Regla de Simpson ---\n');
  fprintf('Cota de error: %.10e\n', E_simpson);
endfunction

% -------------------------------------------------------------------
% Función: cota_error_simpson
% Calcula la cota de error del método de Simpson
function E = cota_error_simpson(f, a, b)
  syms x
  f4 = diff(f, x, 4);             % Cuarta derivada simbólica
  f4_abs = abs(f4);
  f4_func = matlabFunction(f4_abs);
  [xmax, ~] = fminbnd(@(x) -f4_func(x), a, b);% Negamos "-f4_func(x)" para encontrar el máximo con fminbnd
  alpha_max = abs(f4_func(xmax));
  E = ((b - a)^5 / 2880) * alpha_max;
endfunction


