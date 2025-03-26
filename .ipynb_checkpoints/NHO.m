tol = 10^(-10);
iterMax = 100;
x0 = 0;

function y = f(x)
  y = x^2 - exp(x) -3 * x + 2;
end

function y = df(x)
  y = 2 * x - exp(x) -3;
end

function y = H(x,z)
  y = (df(x) - df(z)) / (3 * df(z) - df(x));
end

function [errores, iters, x_n] = NHO()

end
