% ************************************************************************
%                             ENSEMBLE LEARNING
% ************************************************************************

% This script aims to use Ensembling to achieve a higher performance. The 
% approach we try is called Majority Voting Ensemble where we train 4 
% different models (2 SVMs and 2 MLPs) on all Training Data and make them 
% vote for the class of a given new input. We use a 5-fold Cross Validation
% to obtain more accurate estimates of the generalization performance.

function EnsembleLearning(X,y, classNames)
    %% splits datasets into training and validation
    kfolds = 10;
    cv = cvpartition(y, 'KFold', kfolds, 'Stratify', true);
        
    mlp1Scores = zeros(1, kfolds);
    mlp2Scores = zeros(1, kfolds);
    svm1Scores = zeros(1, kfolds);
    svm2Scores = zeros(1, kfolds);
    ensScores = zeros(1, kfolds);
    
    for k = 1:kfolds
        X_train = X(cv.training(k), :);
        y_train = y(cv.training(k), :);
        X_test = X(cv.test(k), :);
        y_test = y(cv.test(k), :);
        %% train Models
        % MLP
        % define mlp classifier
        mlp1 = MLP(X_train, y_train, classNames).fit('TrainFcn', 'trainscg');
        mlp2 = MLP(X_train, y_train, classNames).fit('TrainFcn', 'traingda');

        % SVM classifier
        svm1 = SVM(X_train, y_train, classNames).fit('KernelFunction', 'rbf');
        svm2 = SVM(X_train, y_train, classNames).fit('KernelFunction', 'polynomial');
        
        
        % make MLP predictions
        mlp1Pred = mlp1.predict(X_test);
        mlp2Pred = mlp2.predict(X_test);
                
        % make SVM predictions
        svm1Pred = svm1.predict(X_test);
        svm2Pred = svm2.predict(X_test);
        % make ensemble of MLP and SVM
        allPred = [mlp1Pred'; mlp2Pred'; svm1Pred'; svm2Pred'];
        ensPred = mode(allPred)';
        
        % compute accuracy
        mlp1Scores(k) = Utils.score(y_test, mlp1Pred);
        mlp2Scores(k) = Utils.score(y_test, mlp2Pred);
        svm1Scores(k) = Utils.score(y_test, svm1Pred);
        svm2Scores(k) = Utils.score(y_test, svm2Pred);
        ensScores(k) = Utils.score(y_test, ensPred);
    end
    % plot performance
    % MLP
    folds = 1:kfolds;
    figure('Name', 'Ensemble Learning', 'pos', [100 100 600 480]);
    line(folds, mlp1Scores, 'color', [0 0.4470 0.7410], 'lineStyle', '--');
    hold on;
    line(folds, mlp2Scores, 'color',[0 0.4470 0.7410], 'lineStyle', '-.');
    line(folds, svm1Scores, 'color',[0.8500 0.3250 0.0980], 'lineStyle', '--');
    line(folds, svm2Scores, 'color',[0.8500 0.3250 0.0980], 'lineStyle', '-.');
    line(folds, ensScores, 'color',[0.9290 0.6940 0.1250], 'lineStyle', '-');
    title('Classification Accuracy');
    xlabel("k-fold");
    ylabel("Accuracy");
    legend('MLP 1', 'MLP 2', 'SVM 1', 'SVM 2', 'Ensemble',...
        'Location', 'Best');
    hold off;
    %% Print Performance
    fprintf('============================================\n')
    fprintf('MODEL      CV Avg. Perf    CV Std. Perf     \n')
    fprintf('============================================\n')
    fprintf('MLP 1        %.2f%%          %.2f%%         \n', mean(mlp1Scores)*100, std(mlp1Scores)*100)
    fprintf('MLP 1        %.2f%%          %.2f%%         \n', mean(mlp2Scores)*100, std(mlp2Scores)*100)
    fprintf('____________________________________________\n')
    fprintf('SVM 1        %.2f%%          %.2f%%         \n', mean(svm1Scores)*100, std(svm1Scores)*100)
    fprintf('SVM 2        %.2f%%          %.2f%%         \n', mean(svm2Scores)*100, std(svm2Scores)*100)
    fprintf('____________________________________________\n')
    fprintf('ENSEMBLE     %.2f%%          %.2f%%         \n', mean(ensScores)*100, std(ensScores)*100)
    fprintf('============================================\n')
end

