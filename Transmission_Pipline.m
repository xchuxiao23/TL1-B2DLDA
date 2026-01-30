% ========================================================================
% transmission_pipeline.m
% Simulation of the full pipeline: 
% CS Sampling -> Encryption (Masking) -> Transmission -> Decryption -> Reconstruction
% Target: Level II User (Full Reconstruction Capability)
% ========================================================================

clc;
clear;
close all;

% ========================================================================
% 1. Initialization and CS Setup
% ========================================================================
addpath("./utils/");
addpath("./2DPG-Recon/");
addpath("./WaveletSoftware/");   % WaveletSoftware dependency

dataset_path = "./data/ORL128.mat";
if ~exist(dataset_path, 'file')
    error('Dataset file not found: %s', dataset_path);
end
fprintf('Loading dataset: %s ...\n', dataset_path);
s = load(dataset_path);
X = s.X;

target_img_idx = 3;  
target_subrate = 0.8; 
if target_img_idx > size(X, 3)
    error('Index exceeds the total number of images!');
end

original_image = X(:, :, target_img_idx); 
[num_rows, num_cols] = size(original_image);
fprintf('Testing Image ID: %d, Size: %dx%d, Sampling Rate: %.2f\n', ...
    target_img_idx, num_rows, num_cols, target_subrate);

% ========================================================================
% 2. Sampling (Data Owner)
% ========================================================================
num_levels = 3; 
W_path = "./checkpoint/W_1_iter.mat";
V_path = "./checkpoint/V_1_iter.mat";

% Calculate Measurement Dimension M
M_dim = round(sqrt(target_subrate * num_rows * num_rows)); 
fprintf('Measurement Dim M = %d (Matrix Size: %dx%d)\n', M_dim, M_dim, M_dim);

% --- Step A: Generate Measurement Matrices AND Masking Parameters ---
[Phi1_our, Phi2_our, MaskingParams] = OurMeasurementMatrix(M_dim, num_rows, W_path, V_path);

% --- Step B: Get Compressed Measurements Y ---
val_image = double(original_image) * 255.0; 
Y_our = Phi1_our * val_image * Phi2_our'; % Y_our is double precision matrix
fprintf('CS Sampling complete. Original measurement Y_our generated.\n');

% ========================================================================
% 3. Secure Transmission Layer - Encryption (Masking)
% ========================================================================
fprintf('\n------------------------------------------------\n');
fprintf('  Starting Secure Transmission: Global Scrambling + Diffusion\n');
fprintf('------------------------------------------------\n');

% --- 3.1 Retrieve Deterministic Keys from MaskingParams ---
Key_Perm     = MaskingParams.Key_Perm;      
Key_Diff_Fwd = MaskingParams.Key_Diff_Fwd;  
Key_Diff_Bwd = MaskingParams.Key_Diff_Bwd;  
IV_Fwd       = MaskingParams.IV_Fwd;        
IV_Bwd       = MaskingParams.IV_Bwd;        

% Verify Key Lengths
[M_h, N_w] = size(Y_our);
total_bytes = M_h * N_w * 8; 

if length(Key_Perm) ~= total_bytes
    error('Key length mismatch! Check OurMeasurementMatrix generation logic.');
end

% --- 3.2 Encryption Process ---
Y_uint8 = typecast(Y_our(:), 'uint8');

[~, Perm_Idx] = sort(Key_Perm);
Y_scrambled = Y_uint8(Perm_Idx);

% 3. Bidirectional Modulo Diffusion
C_temp = ModDiffusion.diffusion_fwd_mod(Y_scrambled, Key_Diff_Fwd, IV_Fwd);
C_transmission = ModDiffusion.diffusion_bwd_mod(C_temp, Key_Diff_Bwd, IV_Bwd);

fprintf('Encryption complete. Ciphertext C_transmission ready for transmission.\n');

% ========================================================================
% 4. Receiver Side - Restoration and Decryption
% ========================================================================
fprintf('\n------------------------------------------------\n');
fprintf('  Receiver Decryption in progress...\n');
fprintf('------------------------------------------------\n');

C_rec_temp = ModDiffusion.inv_diffusion_bwd_mod(C_transmission, Key_Diff_Bwd, IV_Bwd);
Y_rec_scrambled = ModDiffusion.inv_diffusion_fwd_mod(C_rec_temp, Key_Diff_Fwd, IV_Fwd);

Inverse_Index = zeros(total_bytes, 1);
Inverse_Index(Perm_Idx) = 1:total_bytes;
Y_rec_uint8 = Y_rec_scrambled(Inverse_Index);

Y_rec_vec = typecast(Y_rec_uint8, 'double');
Y_recovered = reshape(Y_rec_vec, [M_h, N_w]);

is_lossless = isequal(Y_our, Y_recovered);
max_err = max(abs(Y_our(:) - Y_recovered(:)));

if is_lossless
    fprintf('  [Success] Decrypted Y matches original Y_our exactly (Bit-exact).\n');
    fprintf('  [Error] %g\n', max_err);
else
    error('  [Failure] Decryption data mismatch, cannot proceed to reconstruction!');
end

% ========================================================================
% 5. CS Reconstruction (Level II User)
% ========================================================================
fprintf('\n------------------------------------------------\n');
fprintf('  Using Decrypted Data for CS Reconstruction\n');
fprintf('------------------------------------------------\n');

recon_our = func_2DPG(Y_recovered, Phi1_our, Phi2_our, num_rows, num_cols, num_levels);
psnr_our = psnr(uint8(recon_our), uint8(val_image));

fprintf('  Reconstruction complete. PSNR: %.2f dB\n', psnr_our);

% ========================================================================
% 6. Visualization
% ========================================================================
fprintf('\n------------------------------------------------\n');
fprintf('  Visualizing Results...\n');
fprintf('------------------------------------------------\n');

figure('Name', 'Reconstruction Comparison', 'NumberTitle', 'off', 'Color', 'w');

subplot(1, 2, 1);
imshow(uint8(val_image)); 
title(['Original Image (ID: ', num2str(target_img_idx), ')'], 'FontSize', 12);
xlabel(['Size: ', num2str(num_rows), 'x', num2str(num_cols)]);

subplot(1, 2, 2);
imshow(uint8(recon_our));
title(['Reconstructed (CR=', num2str(target_subrate), ')'], 'FontSize', 12);
xlabel(['PSNR: ', num2str(psnr_our, '%.2f'), ' dB']);

fprintf('Visualization displayed.\n');
