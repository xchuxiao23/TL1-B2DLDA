function reconstructed_image = func_2DPG(Y, Phi1, Phi2, num_rows, num_cols, num_levels)

max_iterations = 200;
TOL = 0.0001;   % error tolerance 
eta = 6; % bivariate shrinkage constant
D_prev = 0;

pinvPhi1 = pinv(Phi1);
pinvPhi2 = pinv(Phi2);

% Initialization
X = pinvPhi1 * Y * pinvPhi2';

for i = 1:max_iterations
    [X, D] = iteration_bivariate(Y, X, Phi1, Phi2, pinvPhi1, pinvPhi2, num_rows, num_cols, eta, num_levels);
    if ((D_prev ~= 0) && (abs(D - D_prev) < TOL))     
          break;
    end
    D_prev = D;
end

num_levels = 1;
[X, ~] = iteration_bivariate(Y, X, Phi1, Phi2, pinvPhi1, pinvPhi2, num_rows, num_cols, eta, num_levels);
reconstructed_image = X;

end



function [X, D] = iteration_bivariate(Y, X, Phi1, Phi2, pinvPhi1, pinvPhi2, num_rows, num_cols, eta, num_levels)

[af, sf] = farras;

% L = 10;  % 128 * 128的情况
% L = 8;   % 32 * 32的情况
L = 12;
r = 0.8;   % step size

X_prev = X;   % save the previous solution

% Gradient descent
X_hat = X - r * derivation_of_TV(X, num_rows);

% Extend + DWT2D    
ext_size = L * 2^(num_levels - 1);
X_check = symextend(X_hat, ext_size);
X_check = dwt2D(X_check, num_levels, af);

if (nargin == 9)                         
    end_level = 1;
else
    end_level = num_levels - 1;
end

% Bivariate Shrinkage

X_check = bivariate_shrinkage(X_check, end_level, eta);

% 2D iDWT
X_bar = idwt2D(X_check, num_levels, sf);           
Irow = (ext_size + 1):(ext_size + num_rows);
Icol = (ext_size + 1):(ext_size + num_cols);
X_bar = X_bar(Irow, Icol);

X = X_bar + pinvPhi1 * (Y - Phi1 * X_bar * Phi2') * pinvPhi2';

D = RMS(X, X_prev);

end



% Bivariate Shrinkage Function

function x_check = bivariate_shrinkage(x_check, end_level, eta)

windowsize  = 3;
windowfilt = ones(1, windowsize)/windowsize;

tmp = x_check{1}{3};

% Nsig = median(abs(tmp(:)))/0.6745;

for scale = 1:end_level
  for dir = 1:3
    Y_coefficient = x_check{scale}{dir};
    Y_parent = x_check{scale+1}{dir};
    
    Y_parent = expand(Y_parent);
    
    % Wsig = conv2(windowfilt, windowfilt, (Y_coefficient).^2, 'same');
    % 
    % Ssig = sqrt(max(Wsig-Nsig.^2, eps));
    % 
    % T = sqrt(3)*Nsig^2./Ssig;
    
    sigma_i = median(abs(Y_coefficient)) / 0.6745;

    sigma_f = sqrt(conv2(windowfilt, windowfilt, (Y_coefficient).^2, 'same'));

    T = sqrt(3 * sigma_i) ./ sigma_f;
    
    x_check{scale}{dir} = bishrink(Y_coefficient, Y_parent, T * eta);

  end
end

end