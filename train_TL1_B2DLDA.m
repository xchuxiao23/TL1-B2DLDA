%% ===== GPU Test Script for Tl1-B2DLDA Algorithm =====
% This script evaluates the performance of the Tl1-B2DLDA algorithm 
% on high-resolution image datasets.

clear all; clc; close all;

% --- 1. Environment Setup ---
addpath('./TL1-B2DLDA'); 
addpath('./utils');      

% --- 2. GPU Device Initialization ---
g = gpuDevice(1);
fprintf('Detected GPU: %s (Memory: %.2f GB)\n', g.Name, g.TotalMemory/1024^3);
reset(g); 

% --- 3. Data Loading ---
data_path = './data/ORL128.mat';
if exist(data_path, 'file')
    load(data_path);
else
    error('Dataset not found at: %s', data_path);
end

% --- 4. Parameter Configuration ---
train_ratio = 7/10;           % 70% for training, 30% for testing
iterations = 10;              % Number of experimental rounds
imgSize = size(X, 1);         % Image height/width (e.g., 128)
sampling_rates = [0.05, 0.1, 0.3, 0.5, 0.8]; % Compression Ratios (CR)
alpha = 1;                    % Parameter alpha for Tl1-norm

% Calculate feature dimension 'd' for classification.
% The measurement dimension 'm' is derived from CR: m = sqrt(n^2 * CR).
% Here, we set d = m / 2. The classification uses the top-left d x d 
% sub-matrix of the decoded signal as discriminative features.
space = round(round(sqrt(imgSize * imgSize * sampling_rates)) / 2);
num_cr = length(space);

% Result pre-allocation
TL1_B2DLDA_acc = zeros(iterations, num_cr);
Time_Records = zeros(iterations, 1); 

fprintf('------------------------------------------------\n');
fprintf('Evaluating Tl1-B2DLDA on dataset: %s\n', data_path);
fprintf('------------------------------------------------\n');

for i = 1:iterations
    fprintf('\n=== Iteration %d / %d ===\n', i, iterations);
    
    % 1. Data Splitting
    [trainIdx, testIdx] = randomSplit2D(data_path, train_ratio);   
    x_train = X(:,:,trainIdx);                             
    x_test  = X(:,:,testIdx);
    y_train = Y(trainIdx);                                 
    y_test  = Y(testIdx);
    
    % 2. Training
    % Optimizing W and V using the Tl1-B2DLDA algorithm.
    tic; 

    [W, V] = TL1_B2DLDA(x_train, y_train, imgSize, alpha); 

    save(sprintf('./checkpoint/ORL128_W_iter%d.mat', i), 'W');
    save(sprintf('./checkpoint/ORL128_V_iter%d.mat', i), 'V');
    
    t_cost = toc;
    Time_Records(i) = t_cost;
    fprintf('Training Time: %.4f s\n', t_cost);
    
    % 3. Classification (Evaluation at different CR levels)
    idx_cr = 1;
    fprintf('Acc Results: ');
    for d = space
        current_d = min(d, size(W, 2));

        acc = knn_classifier2D_gpu(W(:, 1:current_d), V(:, 1:current_d), ...
                                   x_train, y_train, x_test, y_test);
        TL1_B2DLDA_acc(i, idx_cr) = acc;
        fprintf('CR[d=%d]: %.2f%% | ', current_d, acc * 100);
        idx_cr = idx_cr + 1;
    end
    fprintf('\n');
end

% --- 5. Statistics and Export ---
save('./Result/ORL_128_TL1_B2DLDA_GPU_Result.mat', 'TL1_B2DLDA_acc', 'space', 'Time_Records');
avg_time = mean(Time_Records);
fprintf('\n------------------------------------------------\n');
fprintf('Evaluation Complete.\n');
fprintf('Average Training Time: %.4f s\n', avg_time);
fprintf('Results saved to: ./Result/ORL_128_TL1_B2DLDA_GPU_Result.mat\n');