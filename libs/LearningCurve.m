% ************************************************************************
%                             LEARNING CURVE
% ************************************************************************

% This script visualizes the Learning Curve of the MLP and SVM Algorithms. 
% It uses the best configuration from the hyper-parameter tuning. For MLP,
% we changed the optimization procedure : instead of the recommended
% 'logsig', we use 'tansig' to accelerate training due to limited
% ressources. In order to stabilize the results, 10-fold cross validation
% is used to output accuracy estimates for both training and validation
% along with their estimated errors as represented by the standard
% deviation.

function LearningCurve(inputs, target, classNames)
    % define experiment variables
    k_folds = 10;
    total = numel(target);
    % create input sizes of 10%, 30%, 50%, 70% and 100% of the datasets
    inputSizes = floor([total*0.1 total*0.3 total*0.5 total*0.7 total*1]);
    % define experiment result variables
    sz = numel(inputSizes);
    MLPTrainAvgScores = zeros(1, sz);
    MLPTrainStdScores = zeros(1, sz);
    MLPValAvgScores = zeros(1, sz);
    MLPValStdScores = zeros(1, sz);

    SVMTrainAvgScores = zeros(1, sz);
    SVMTrainStdScores = zeros(1, sz);
    SVMValAvgScores = zeros(1, sz);
    SVMValStdScores = zeros(1, sz);
    %% run experiment for different input size
    for index = 1:sz
        inputSize = inputSizes(index);
        fprintf("\n- Input Size %d \n", inputSize);
        % create subset of features and target variables
        idx = randperm(numel(target), inputSize);
        X = inputs(idx, :);
        y = target(idx, :);
        
        % Create 10-fold Cross Validation using stratified method to ensure 
        % equal distribution of classes. 
        cv = cvpartition(target(idx), 'KFold', k_folds, 'Stratify', true);
        
        % define cross validation result variables
        MLPTrainScores = zeros(1, k_folds);
        MLPValScores = zeros(1, k_folds);
        SVMTrainScores = zeros(1, k_folds);
        SVMValScores = zeros(1, k_folds);
        %% run cross validation
        for k = 1:k_folds
            fprintf("Cross Validation Fold %d/%d \n", k, k_folds);
            X_train = X(cv.training(k), :);
            y_train = y(cv.training(k), :);
            X_test = X(cv.test(k), :);
            y_test = y(cv.test(k), :);
            % define mlp classifier
            mlp = MLP(X_train, y_train, classNames).fit();
            % define svm classifier
            svm = SVM(X_train, y_train, classNames).fit();
            
            % evaluate MLP model
            MLPTrainScores(k) = mlp.score(X_train, y_train);
            MLPValScores(k) = mlp.score(X_test, y_test);
            
            % evaluate SVM model
            SVMTrainScores(k) = svm.score(X_train, y_train);
            SVMValScores(k) = svm.score(X_test, y_test);
        end
        
        %% Calculate score average and standard deviation and append to results
        % MLP
        MLPTrainAvgScores(index) = mean(MLPTrainScores);
        MLPTrainStdScores(index) = std(MLPTrainScores);
        MLPValAvgScores(index) = mean(MLPValScores);
        MLPValStdScores(index) = std(MLPValScores);
        % SVM
        SVMTrainAvgScores(index) = mean(SVMTrainScores);
        SVMTrainStdScores(index) = std(SVMTrainScores);
        SVMValAvgScores(index) = mean(SVMValScores);
        SVMValStdScores(index) = std(SVMValScores);  
    end
    %% Visualize experiment results
    
    % MLP
    figure('Name', 'Learning Curve', 'pos', [100 100 1200 480]);
    subplot(1,2,1)
    patch([inputSizes fliplr(inputSizes)], [MLPTrainAvgScores+MLPTrainStdScores,...
        fliplr(MLPTrainAvgScores-MLPTrainStdScores)],...
        [205/255 92/255 92/255], 'edgecolor', 'none', 'FaceAlpha', 0.2);
    hold on;
    patch([inputSizes fliplr(inputSizes)], [MLPValAvgScores+MLPValStdScores,...
        fliplr(MLPValAvgScores-MLPValStdScores)],...
        [100/255 149/255 237/255], 'edgecolor', 'none', 'FaceAlpha', 0.2);
    line(inputSizes, MLPTrainAvgScores, 'color', [205/255 92/255 92/255], 'marker', '*', 'lineStyle', '-.');
    line(inputSizes, MLPValAvgScores, 'color', [100/255 149/255 237/255], 'marker', '*', 'lineStyle', '-.');
    title('Learning Curve - MLP');
    xlabel("Training samples");
    ylabel("Accuracy");
    legend('Train Score Error', 'Validation Score Error',...
        'Train Score Estimate', 'Validation Score Estimate',...
        'Location', 'Best');
    % SVM
    subplot(1,2,2);
    patch([inputSizes fliplr(inputSizes)], [SVMTrainAvgScores+SVMTrainStdScores,...
        fliplr(SVMTrainAvgScores-SVMTrainStdScores)],...
        [205/255 92/255 92/255], 'edgecolor', 'none', 'FaceAlpha', 0.2);
    hold on;
    patch([inputSizes fliplr(inputSizes)], [SVMValAvgScores+SVMValStdScores,...
        fliplr(SVMValAvgScores-SVMValStdScores)],...
        [100/255 149/255 237/255], 'edgecolor', 'none', 'FaceAlpha', 0.2);
    
    line(inputSizes, SVMTrainAvgScores, 'color', [205/255 92/255 92/255], 'marker', '*', 'lineStyle', '-.');
    line(inputSizes, SVMValAvgScores, 'color', [100/255 149/255 237/255], 'marker', '*', 'lineStyle', '-.');
    title('Learning Curve - SVM');
    xlabel("Training samples");
    ylabel("Accuracy");
    legend('Train Score Error', 'Validation Score Error',...
        'Train Score Estimate', 'Validation Score Estimate',...
        'Location', 'Best');
end

