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
        
    mlpScores = [];
    svmScores = [];
    ensScores = [];
    
    for k = 1:kfolds
        
        %% train Models
        % MLP
        % define mlp classifier
        mlp1 = MLP(X, y).fit('TransferFcn', 'softmax',...
        'CrossVal', true, 'cv', cv, 'CVfold', k);
        mlp2 = MLP(X, y).fit('TransferFcn', 'tansig',...
        'CrossVal', true, 'cv', cv, 'CVfold', k);

        % SVM classifier
        svm1 = SVM(X, y, classNames).fit('KernelFunction', 'rbf');
        svm2 = SVM(X, y, classNames).fit('KernelFunction', 'polynomial');
        
        X_test = X(cv.test(k), :);
        y_test = y(cv.test(k), :);
        
        % make MLP predictions
        posteriorMLP1 = mlp1.predict(X_test);
        mlp1Pred = MLP.labelsFromScores(posteriorMLP1, classNames);
        posteriorMLP2 = mlp2.predict(X_test);
        mlp2Pred = MLP.labelsFromScores(posteriorMLP2, classNames);
        
        majorityMLP = mode([mlp1Pred'; mlp2Pred'])';
        
        % make SVM predictions
        svm1Pred = predict(svm1.model, X_test);
        svm2Pred = predict(svm2.model, X_test);
        majoritySVM = mode([svm1Pred'; svm2Pred'])';
        % make ensemble of MLP and SVM
        majorityEns = mode([majorityMLP'; majoritySVM'])';
        
        % compute accuracy
        mlpScores = [mlpScores sum(y_test == majorityMLP)/numel(y_test)];
        svmScores = [svmScores sum(y_test == majoritySVM)/numel(y_test)];
        ensScores = [ensScores sum(y_test == majorityEns)/numel(y_test)];
    end
    %% Print Performance
    fprintf('__________________________________________ \n')
    fprintf('MODEL         CV Avg. Perf    CV Std. Perf \n')
    fprintf('__________________________________________ \n')
    fprintf('MLP Ensemble        %.2f%%          %.2f%% \n', mean(mlpScores)*100, std(mlpScores)*100)
    fprintf('SVM Ensemble        %.2f%%          %.2f%% \n', mean(svmScores)*100, std(svmScores)*100)
    fprintf('ENSEMBLE            %.2f%%          %.2f%% \n', mean(ensScores)*100, std(ensScores)*100)
    fprintf('__________________________________________ \n')
end

