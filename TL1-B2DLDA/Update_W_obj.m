function f = Update_W_obj(x, y, w, a)
% Update_W_obj (GPU Version)
    [m, n, N] = size(x);
    u_labels = unique(y); 
    num_label = length(u_labels);

    x_mean = mean(x, 3);
    x_classmean = zeros(m, n, num_label, 'like', x);
    num_eachclass = zeros(num_label, 1, 'like', x);
    
    for i = 1:num_label
        idx = (y == u_labels(i));
        x_classmean(:, :, i) = mean(x(:, :, idx), 3);
        num_eachclass(i) = sum(idx);
    end

    Sb_data = x_classmean - x_mean;
    Sb_proj = pagefun(@mtimes, w', Sb_data);
    costs_Sb = (a + 1) .* abs(Sb_proj) ./ (a + abs(Sb_proj));
    numerator = sum(sum(costs_Sb, 2) .* reshape(num_eachclass, 1, 1, []));

    X_centers = zeros(m, n, N, 'like', x);
    for i = 1:num_label
        idx = (y == u_labels(i));
        X_centers(:, :, idx) = repmat(x_classmean(:, :, i), 1, 1, sum(idx));
    end
    Sw_data = x - X_centers;
    
    Sw_proj = pagefun(@mtimes, w', Sw_data);
    costs_Sw = (a + 1) .* abs(Sw_proj) ./ (a + abs(Sw_proj));
    denominator = sum(costs_Sw, 'all');

    f = numerator / denominator;
end