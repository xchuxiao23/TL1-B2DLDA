function Acc = knn_classifier2D_gpu(W, V, x_train, y_train, x_test, y_test)
    W       = gpuArray(W);
    V       = gpuArray(V);
    x_train = gpuArray(x_train);
    y_train = gpuArray(y_train);
    x_test  = gpuArray(x_test);
    y_test  = gpuArray(y_test);

    [~, ~, n_train] = size(x_train);
    [~, ~, n_test]  = size(x_test);
    d1 = size(W, 2);
    d2 = size(V, 2);

    train_prj = zeros(d1, d2, n_train, 'gpuArray');
    test_prj  = zeros(d1, d2, n_test, 'gpuArray');

    for i = 1:n_train
        train_prj(:,:,i) = W' * x_train(:,:,i) * V;
    end
    
    for i = 1:n_test
        test_prj(:,:,i)  = W' * x_test(:,:,i) * V;
    end

    feat_dim = d1 * d2;
    
    train_feats = reshape(train_prj, feat_dim, n_train)'; 
    test_feats  = reshape(test_prj,  feat_dim, n_test)'; 

    Dists = pdist2(test_feats, train_feats, 'euclidean');

    [~, min_indices] = min(Dists, [], 2);

    pred_labels = y_train(min_indices);

    correct_count = sum(pred_labels == y_test);
    
    Acc = gather(correct_count / n_test);
    
end