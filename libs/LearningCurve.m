function LearningCurve(inputs, target, classNames)
    % define experiment variables
    k_folds = 10;
    inputSizes = [100 500 1000 3000 5000 7195];
    % define experiment result variables
    MLP_trainAvgScores = [];
    MLP_trainStdScores = [];
    MLP_valAvgScores = [];
    MLP_valStdScores = [];

    SVM_trainAvgScores = [];
    SVM_trainStdScores = [];
    SVM_valAvgScores = [];
    SVM_valStdScores = [];
    %% run experiment for different input size
    for inputSize = inputSizes
        fprintf("\n- Input Size %d \n", inputSize);
        % create subset of features and target variables
        idx = randperm(numel(target), inputSize);
        X = inputs(idx, :);
        y = target(idx, :);
        
        % Create 10-fold Cross Validation using stratified method to ensure 
        % equal distribution of classes. 
        cv = cvpartition(target(idx), 'KFold', k_folds, 'Stratify', true);
        
        % define cross validation result variables
        MLP_trainScores = [];
        MLP_valScores = [];
        SVM_trainScores = [];
        SVM_valScores = [];
        %% run cross validation
        for k = 1:k_folds
            fprintf("Cross Validation Fold %d/%d \n", k, k_folds);
            % define mlp classifier
            mlp = MLP(X, y).fit('CrossVal', true, 'cv', cv, 'CVfold', k);
            % define svm classifier
            svm = SVM(X(cv.training(k), :), y(cv.training(k), :), classNames).fit();
            
            % evaluate MLP model
            MLP_trainScore = mlp.score(X(cv.training(k), :), y(cv.training(k), :));
            MLP_valScore = mlp.score(X(cv.test(k), :), y(cv.test(k), :));
            % append MLP results to container
            MLP_trainScores = [MLP_trainScores MLP_trainScore];
            MLP_valScores = [MLP_valScores MLP_valScore];
            % evaluate SVM model
            SVM_trainScore = svm.score(X(cv.training(k), :), y(cv.training(k), :));
            SVM_valScore = svm.score(X(cv.test(k), :), y(cv.test(k), :));
            % append SVM results to container
            SVM_trainScores = [SVM_trainScores SVM_trainScore];
            SVM_valScores = [SVM_valScores SVM_valScore];
        end
        
        %% Calculate score average and standard deviation and append to results
        % MLP
        MLP_trainAvgScores = [MLP_trainAvgScores mean(MLP_trainScores)];
        MLP_trainStdScores = [MLP_trainStdScores std(MLP_trainScores)];
        MLP_valAvgScores = [MLP_valAvgScores mean(MLP_valScores)];
        MLP_valStdScores = [MLP_valStdScores std(MLP_valScores)];
        % SVM
        SVM_trainAvgScores = [SVM_trainAvgScores mean(SVM_trainScores)];
        SVM_trainStdScores = [SVM_trainStdScores std(SVM_trainScores)];
        SVM_valAvgScores = [SVM_valAvgScores mean(SVM_valScores)];
        SVM_valStdScores = [SVM_valStdScores std(SVM_valScores)];  
    end
    %% Visualize experiment results
    
    % MLP
    figure('Name', 'Learning Curve', 'pos', [100 100 1200 480]);
    subplot(1,2,1)
    patch([inputSizes fliplr(inputSizes)], [MLP_trainAvgScores+MLP_trainStdScores,...
        fliplr(MLP_trainAvgScores-MLP_trainStdScores)],...
        [205/255 92/255 92/255], 'edgecolor', 'none', 'FaceAlpha', 0.2);
    hold on;
    patch([inputSizes fliplr(inputSizes)], [MLP_valAvgScores+MLP_valStdScores,...
        fliplr(MLP_valAvgScores-MLP_valStdScores)],...
        [100/255 149/255 237/255], 'edgecolor', 'none', 'FaceAlpha', 0.2);
    line(inputSizes, MLP_trainAvgScores, 'color', [205/255 92/255 92/255], 'marker', '*', 'lineStyle', '-.');
    line(inputSizes, MLP_valAvgScores, 'color', [100/255 149/255 237/255], 'marker', '*', 'lineStyle', '-.');
    title('Learning Curve - MLP');
    legend('Train Score Error', 'Validation Score Error',...
        'Train Score Estimate', 'Validation Score Estimate',...
        'Location', 'Best');
    % SVM
    subplot(1,2,2);
    patch([inputSizes fliplr(inputSizes)], [SVM_trainAvgScores+SVM_trainStdScores,...
        fliplr(SVM_trainAvgScores-SVM_trainStdScores)],...
        [205/255 92/255 92/255], 'edgecolor', 'none', 'FaceAlpha', 0.2);
    hold on;
    patch([inputSizes fliplr(inputSizes)], [SVM_valAvgScores+SVM_valStdScores,...
        fliplr(SVM_valAvgScores-SVM_valStdScores)],...
        [100/255 149/255 237/255], 'edgecolor', 'none', 'FaceAlpha', 0.2);
    
    line(inputSizes, SVM_trainAvgScores, 'color', [205/255 92/255 92/255], 'marker', '*', 'lineStyle', '-.');
    line(inputSizes, SVM_valAvgScores, 'color', [100/255 149/255 237/255], 'marker', '*', 'lineStyle', '-.');
    title('Learning Curve - SVM');
    legend('Train Score Error', 'Validation Score Error',...
        'Train Score Estimate', 'Validation Score Estimate',...
        'Location', 'Best');
end

