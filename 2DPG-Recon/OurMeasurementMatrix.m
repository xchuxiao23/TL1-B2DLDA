function [Phi1, Phi2, MaskingParams] = OurMeasurementMatrix(M, N, W_path, V_path)
% OurMeasurementMatrix: Generates Measurement Matrices and Masking Keys
%
% Inputs:
%   M      - Measurement dimension (size of compressed image Y is M x M)
%   N      - Original image dimension (N x N)
%   W_path - Path to projection matrix W
%   V_path - Path to projection matrix V
%
% Outputs:
%   Phi1, Phi2    - Left and Right Measurement Matrices
%   MaskingParams - Struct containing keys for the subsequent encryption stage

    % 1. Determine feature dimension d
    d = round(M / 2);

    % 2. Calculate total length required for Chaotic Sequence
    Num_Elements_Matrix = M * M;           % For P and Q (doubles)
    Num_Bytes_Y = Num_Elements_Matrix * 8; % Y is double type (8 bytes per element)
    
    % Total length of chaotic sequence needed
    Total_Seq_Len = (Num_Elements_Matrix * 2) + (Num_Bytes_Y * 3) + 128; 
    
    % Improved_HenonMap generates 2 values (x, y) per iteration.
    Gen_Iterations = ceil(Total_Seq_Len / 2);

    % 3. Generate Unified Chaotic Sequence (Key1 controlled)
    [Seq_X, Seq_Y] = Improved_HenonMap(Gen_Iterations, 0.3, 0.4, 6, 10);
    
    % Concatenate to form a single continuous stream
    Full_Stream = [Seq_X(:); Seq_Y(:)];
    
    % 4. Extract Parameters Sequentially
    current_idx = 0;
    
    % --- Extract P (M x M) ---
    P_vec = Full_Stream(current_idx + 1 : current_idx + Num_Elements_Matrix);
    current_idx = current_idx + Num_Elements_Matrix;
    P = reshape(P_vec, [M, M]);
    
    % --- Extract Q (M x M) ---
    Q_vec = Full_Stream(current_idx + 1 : current_idx + Num_Elements_Matrix);
    current_idx = current_idx + Num_Elements_Matrix;
    Q = reshape(Q_vec, [M, M]);
    
    % --- Extract Masking Parameters (generated AFTER P and Q) ---
    
    % A. Permutation Key (Use raw doubles for sorting)
    MaskingParams.Key_Perm = Full_Stream(current_idx + 1 : current_idx + Num_Bytes_Y);
    current_idx = current_idx + Num_Bytes_Y;
    
    % B. Diffusion Key Forward (Quantize to uint8)
    raw_diff_fwd = Full_Stream(current_idx + 1 : current_idx + Num_Bytes_Y);
    MaskingParams.Key_Diff_Fwd = uint8(mod(floor(raw_diff_fwd * 1e10), 256));
    current_idx = current_idx + Num_Bytes_Y;
    
    % C. Diffusion Key Backward (Quantize to uint8)
    raw_diff_bwd = Full_Stream(current_idx + 1 : current_idx + Num_Bytes_Y);
    MaskingParams.Key_Diff_Bwd = uint8(mod(floor(raw_diff_bwd * 1e10), 256));
    current_idx = current_idx + Num_Bytes_Y;
    
    % D. Initial Vectors (IV)
    raw_iv = Full_Stream(current_idx + 1 : current_idx + 2);
    MaskingParams.IV_Fwd = uint8(mod(floor(raw_iv(1) * 1e10), 256));
    MaskingParams.IV_Bwd = uint8(mod(floor(raw_iv(2) * 1e10), 256));

    % 5. Construct Measurement Matrices (Phi1, Phi2)
    H = zeros(N, M);
    H(1:M, 1:M) = eye(M);

    % Generate R1, R2 using Key2 (ZeraouliaSprottMap)
    [r1, r2] = Improved_ZeraouliaSprottMap(N * (N - d), 0.5, 0.6, 15, 20);
    r1 = reshape(r1, [N, N - d]);
    r2 = reshape(r2, [N, N - d]);
    
    R1 = zeros(N, N);
    R2 = zeros(N, N);
    R1(:, (d + 1):N) = r1;
    R2(:, (d + 1):N) = r2;

    % Load Projection Matrices
    W = load(W_path).W;
    V = load(V_path).V;

    % Final Computation
    Phi1 = P * H' * (W + R1)';
    Phi2 = ((V + R2) * H * Q)';

end