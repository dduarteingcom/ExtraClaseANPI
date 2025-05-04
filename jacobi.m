% Archivo: jacobi.m (script principal)
1;
% La línea '1;' evita que Octave confunda el script con una función

function angle = theta(A, i, j)
    a_ij = A(i, j);
    denominator = A(i, i) - A(j, j);
    if abs(denominator) > 1e-16
        angle = 0.5 * atan(2 * a_ij / denominator);
    else
        angle = 0.0;
    end
endfunction

function G = matriz_rotacion(i, j, m, theta)
    I = eye(m);
    Z = zeros(m);
    c = cos(theta);
    s = sin(theta);
    Z(i, i) = c - 1;
    Z(j, j) = c - 1;
    if i != j
        Z(i, j) = -s;
        Z(j, i) = s;
    end
    G = I + Z;
endfunction

function valores_propios = jacobi_valores_propios(A, iterMax, tol)
    A0 = A;
    m = size(A, 1);
    Ak = A0;
    xk = diag(Ak);

    for k = 1:iterMax
        Bk = Ak;

        for i = 1:m
            for j = 1:m
                if i >= j
                    continue;
                end
                theta_ij = theta(Bk, i, j);
                G = matriz_rotacion(i, j, m, theta_ij);
                Bk = G' * Bk * G;
            end
        end

        Ak_next = Bk;
        xk_next = diag(Ak_next);
        ek = norm(xk_next - xk);

        if ek < tol
            break;
        end

        Ak = Ak_next;
        xk = xk_next;
    end

    valores_propios = diag(Ak);
endfunction

% --- Script principal ---
A = zeros(15);
for i = 1:15
    for j = 1:15
        A(i, j) = 0.5 * (i + j);
    end
end

valores_propios = jacobi_valores_propios(A, 1000, 1e-10);
disp("Valores propios calculados:");
disp(sort(valores_propios));
