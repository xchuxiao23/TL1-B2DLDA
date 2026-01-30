function [W, V] = TL1_B2DLDA(x, y, d, a)
% TL1_B2DLDA: GPU-accelerated Bilateral 2D-LDA with Tl1-norm optimization.
%
% Inputs:
%   x - Input image data (Height x Width x N)
%   y - Class labels
%   d - Projection dimension (Target feature size)
%   a - Parameter alpha for Tl1-norm
%
% Outputs:
%   W, V - Optimized projection matrices (returned as CPU double)

    % --- 1. Data Transfer to GPU (Single Precision for Speed) ---
    x_gpu = gpuArray(single(x)); 
    y_gpu = gpuArray(y);
    
    TotalIter = 30;
    LastObj = -inf; 
    
    [data_m, ~, ~] = size(x_gpu);
    
    % Initialize W as identity matrix on GPU
    W_gpu = gpuArray.eye(data_m, d, 'single');
    V_gpu = []; 
    
    % --- 2. Alternating Iterative Optimization ---
    for t = 1:TotalIter
        fprintf('---------- GPU ITER %d ----------\n', t);
        
        % Step A: Optimize V while fixing W
        V_temp_gpu = Update_V(x_gpu, y_gpu, W_gpu, d, a);
   
        % Step B: Optimize W while fixing V
        W_temp_gpu = Update_W(x_gpu, y_gpu, V_temp_gpu, d, a);
        
        % Step C: Calculate Objective Function
        obj_gpu = TL1_B2DLDA_obj(x_gpu, y_gpu, V_temp_gpu, W_temp_gpu, a);
        obj_cpu = gather(obj_gpu); % Retrieve scalar for comparison
        
        fprintf('Objective Value: %f\n', obj_cpu);
        
        % Step D: Monotonicity Check (Keep the Best)
        if (obj_cpu > LastObj)
            LastObj = obj_cpu;
            W_gpu = W_temp_gpu;
            V_gpu = V_temp_gpu;
        else
            fprintf('Algorithm Converged at iter %d.\n', t);
            break;
        end
    end
    
    % --- 3. Result Retrieval ---
    W = gather(double(W_gpu)); 
    V = gather(double(V_gpu));
end