function [W] = Update_W(x, y, V, d, a)
% Update_W (GPU Version)
    
    [data_m, data_n, N] = size(x);  
    u_labels = unique(y);                 
    num_label = length(u_labels);

    dataX2 = pagefun(@mtimes, x, V);
    
    T = gpuArray.eye(data_m, 'single');
    W = gpuArray.zeros(data_m, 0, 'single');
    
    UpdateIter = 200;
    tol = 1e-4;
    
    for l = 1:d
        x_curr = pagefun(@mtimes, T', dataX2);
        [m, n, ~] = size(x_curr);
        
        x_mean = mean(x_curr, 3);
        x_classmean = zeros(m, n, num_label, 'like', x);
        num_eachclass = zeros(num_label, 1, 'like', x);
        
        for i = 1:num_label
            idx = (y == u_labels(i));
            x_classmean(:, :, i) = mean(x_curr(:, :, idx), 3);
            num_eachclass(i) = sum(idx);
        end
        
        Sb_data = x_classmean - x_mean;
        
        X_centers = zeros(m, n, N, 'like', x);
        for i = 1:num_label
            idx = (y == u_labels(i));
            X_centers(:, :, idx) = repmat(x_classmean(:, :, i), 1, 1, sum(idx));
        end
        Sw_data = x_curr - X_centers;
        
        w0 = gpuArray.randn(m, 1, 'single');
        w0 = w0 / norm(w0);
        theta0 = pi / 2;
        
        for kk = 1:UpdateIter
            Sb_proj = pagefun(@mtimes, w0', Sb_data);
            Sw_proj = pagefun(@mtimes, w0', Sw_data);
            
            abs_Sb = abs(Sb_proj);
            abs_Sw = abs(Sw_proj);
            
            n_prime_coeff = (a * (a + 1)) .* sign(Sb_proj) ./ ((a + abs_Sb).^2);
            d_prime_coeff = (a * (a + 1)) .* sign(Sw_proj) ./ ((a + abs_Sw).^2);
            
            num4 = sum(sum((a + 1) .* abs_Sb ./ (a + abs_Sb), 2) .* reshape(num_eachclass, 1, 1, []));
            num2 = sum((a + 1) .* abs_Sw ./ (a + abs_Sw), 'all');
            
            grad_n = sum(reshape(num_eachclass, 1, 1, []) .* pagefun(@mtimes, Sb_data, pagefun(@transpose, n_prime_coeff)), 3);
            grad_d = sum(pagefun(@mtimes, Sw_data, pagefun(@transpose, d_prime_coeff)), 3);
            
            numerator = grad_n * num2 - grad_d * num4;
            denominator = num2 ^ 2;
            gradf = numerator / denominator;
            
            g = gradf - (gradf' * w0) * w0;
            g_norm = norm(g);
            if g_norm < 1e-6, break; end
            g0 = g / g_norm;
            
            current_obj = Update_W_obj(x_curr, y, w0, a);
            
            while 1
                w1 = w0 * cos(theta0) + g0 * sin(theta0);
                new_obj = Update_W_obj(x_curr, y, w1, a);
                
                if new_obj >= current_obj
                    break;
                end
                theta0 = theta0 / 2;
                if theta0 < 1e-6, break; end
            end
            
            if abs(new_obj - current_obj) < tol
                w0 = w1;
                break;
            end
            theta0 = min(theta0 * 2, pi / 2);
            w0 = w1;
        end
        
        w_final = T * w0;
        W = [W, w_final];
        
        if l < d
             [Q_qr, ~] = qr(W); 
             T = Q_qr(:, (l+1):end);
        end
    end
    % fprintf('[Update_W] Finished...\n');
end