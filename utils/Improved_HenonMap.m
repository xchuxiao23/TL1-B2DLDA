function [x, y] = Improved_HenonMap(n, x0, y0, a, b)

x = zeros(1, n);
y = zeros(1, n);

x_prev = x0;
y_prev = y0;

for i = 1:n
    x_next = mod(1 - a * x_prev^2 + y_prev, 1);
    y_next = mod(b * x_prev, 1);

    x(i) = x_next;
    y(i) = y_next;

    x_prev = x_next;
    y_prev = y_next;
end

end
