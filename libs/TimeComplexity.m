% ************************************************************************
%                   TIME COMPLEXITY ANALYSIS
% ************************************************************************

% This script executes a Multi-Layer Perceptron and Vector Support System
% Complexity Analysis. Time Complexity will be calculated using structured
% models with different number of training examples and attributes. The
% experiment is run N times and the training times for both models are
% determined and combined during each trial.
function TimeComplexity(inputs, target, classNames)
    % define experiment variables
    [rows, columns] = size(inputs);
    % create input sizes of 10%, 30%, 50%, 70% and 100% of the datasets
    inputSizes = floor([rows*0.1 rows*0.3 rows*0.5 rows*0.7 rows*1]);
    % create col sizes
    colSizes = floor([columns*0.1 columns*0.3 columns*0.5 columns*0.7 columns*1]);
    % number of experiments to run for each input size
    N = 10;
    % define experiment result variables
    sz = numel(inputSizes);
    % mlp
    MLPAvgTrainTimes = zeros(1, sz);
    MLPStdTrainTimes = zeros(1, sz);
    % svm
    SVMAvgTrainTimes = zeros(1, sz);
    SVMStdTrainTimes = zeros(1, sz);
    %% Run experiments 1: Sample size
    % Training Time as a Function of Number of Observations
    fprintf('Training Time as a Function of Number of Observations.\n')
    for index = 1:sz
        inputSize = inputSizes(index);
        fprintf("\n- Input Size %d \n", inputSize);
        % create subset of features and target variables
        idx = randperm(rows, inputSize);
        X = inputs(idx, :);
        y = target(idx, :);        
        % define experiments results
        MLPTrainTimes = zeros(1, N);
        SVMTrainTimes = zeros(1, N);
        
        % run experiments for N times
        for k = 1:N
            fprintf("Experiments %d/%d \n", k, N);
            % MLP: train and record MLP training time
            tic;
            MLP(X, y, classNames).fit();
            MLPTrainTimes(k) = toc;
            % SVM: train and record SVM training time
            tic;
            SVM(X, y, classNames).fit();
            SVMTrainTimes(k) = toc;
        end
        %% Calculate training times average and standard deviation and append to results
        % MLP
        MLPAvgTrainTimes(index) = mean(MLPTrainTimes);
        MLPStdTrainTimes(index) = std(MLPTrainTimes);
        % SVM
        SVMAvgTrainTimes(index) = mean(SVMTrainTimes);
        SVMStdTrainTimes(index) = std(SVMTrainTimes);
    end
    
    %% Visualize experiment results
    
    % Experiment 1 - Time Complexity for Sample Size
    figure('Name', 'Time Complexity for Sample Size', 'pos', [100 100 1200 480]);
    subplot(1, 2, 1)
    patch([inputSizes fliplr(inputSizes)], [MLPAvgTrainTimes+MLPStdTrainTimes,...
        fliplr(MLPAvgTrainTimes-MLPStdTrainTimes)],...
        [205/255 92/255 92/255], 'edgecolor', 'none', 'FaceAlpha', 0.2);
    hold on;
    line(inputSizes, MLPAvgTrainTimes, 'color', [205/255 92/255 92/255], 'marker', '*', 'lineStyle', '-.');
    patch([inputSizes fliplr(inputSizes)], [SVMAvgTrainTimes+SVMStdTrainTimes,...
        fliplr(SVMAvgTrainTimes-SVMStdTrainTimes)],...
        [100/255 149/255 237/255], 'edgecolor', 'none', 'FaceAlpha', 0.2);
    line(inputSizes, SVMAvgTrainTimes, 'color', [100/255 149/255 237/255], 'marker', '*', 'lineStyle', '-.');
    xlabel("Training sample size");
    ylabel('Training Times');
    title('Time Complexity for Sample Size');
    legend('Train Time Error - MLP','Training Time Estimate - MLP',...
        'Train Time Error - SVM', 'Training Time Estimate - SVM',...
        'Location', 'Best');
    %% Run experiments 2: Attributes size
    % Training Time as a Function of Number of Attributes
    fprintf('Training Time as a Function of Number of Attributes.\n')
    for index = 1:sz
        colSize = colSizes(index);
        fprintf("\n- Attributes Size %d \n", colSize);
        % randomize attributes selection
        idx = randperm(columns, colSize);
        X = inputs(:, idx);
        y = target(:);        
        % define experiments results
        MLPTrainTimes = zeros(1, N);
        SVMTrainTimes = zeros(1, N);
        
        % run experiments for N times
        for k = 1:N
            fprintf("Experiments %d/%d \n", k, N);
            % MLP: train and record MLP training time
            tic;
            MLP(X, y, classNames).fit();
            MLPTrainTimes(k) = toc;
            % SVM: train and record SVM training time
            tic;
            SVM(X, y, classNames).fit();
            SVMTrainTimes(k) = toc;
        end
        %% Calculate training times average and standard deviation and append to results
        % MLP
        MLPAvgTrainTimes(index) = mean(MLPTrainTimes);
        MLPStdTrainTimes(index) = std(MLPTrainTimes);
        % SVM
        SVMAvgTrainTimes(index) = mean(SVMTrainTimes);
        SVMStdTrainTimes(index) = std(SVMTrainTimes);
    end
    
    %% Visualize experiment results
    % Experiment 2 - Time Complexity for Attributes Size
%     figure('Name', 'Time Complexity for Attributes Size', 'pos', [100 100 1200 480]);
    subplot(1, 2, 2)
    patch([colSizes fliplr(colSizes)], [MLPAvgTrainTimes+MLPStdTrainTimes,...
        fliplr(MLPAvgTrainTimes-MLPStdTrainTimes)],...
        [0.6350 0.0780 0.1840], 'edgecolor', 'none', 'FaceAlpha', 0.2);
    hold on;
    line(colSizes, MLPAvgTrainTimes, 'color', [0.6350 0.0780 0.1840], 'marker', '*', 'lineStyle', '-.');
    patch([colSizes fliplr(colSizes)], [SVMAvgTrainTimes+SVMStdTrainTimes,...
        fliplr(SVMAvgTrainTimes-SVMStdTrainTimes)],...
        [0 0.4470 0.7410], 'edgecolor', 'none', 'FaceAlpha', 0.2);
    line(colSizes, SVMAvgTrainTimes, 'color', [0 0.4470 0.7410], 'marker', '*', 'lineStyle', '-.');
    xlabel("Training Attributes size");
    ylabel('Training Times');
    title('Time Complexity for Attributes Size');
    legend('Train Time Error - MLP','Training Time Estimate - MLP',...
        'Train Time Error - SVM', 'Training Time Estimate - SVM',...
        'Location', 'Best');
end

