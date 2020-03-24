function [mlp, svm] = HyperParameterTuning(X, y, classNames)
    %HYPERPARAMETERTUNING Summary of this function goes here
    %   Detailed explanation goes here
    % define model names
    names = {'MLP', 'SVM'};
    [X_train, y_train, X_test, y_test] = Utils.train_test_split(X, y, 0.1);
    %% Tune model hyperparameters
    % train and optimize SVM classifier
    svm = SVM(X_train, y_train, classNames).optimize('MaxObjectiveEvaluations', inf);
    % train and optimize MLP classifier
    mlp = MLP(X_train, y_train, classNames).optimize('MaxObjectiveEvaluations', inf);
    
    %% train basic models with default parameters
    svmDefault = SVM(X_train, y_train, classNames).fitDefault();
    mlpDefault = MLP(X_train, y_train, classNames).fitDefault();
    %% Plot Minimum Objective Curves
   
    % return the optimization results
    mdls = {mlp, svm};
    
    N = length(mdls);
    for i = 1:N
        % plot Minimum Objective Curve
        plot(mdls{i}.OptimizationResults, @plotMinObjective);
    end
    grid on
    legend(names{1}, names{2},'Location','northeast')
    title('Plot Minimum Objective Curves - Bayesian Optimization')
    xlabel('Number of Iterations')
    ylabel('Minimum Objective Value')
    %% Check Performance with Test Set
    % evaluate models SVM on validation set
    svmDefaultAcc = svmDefault.score(X_test, y_test);
    svmTunedAcc = svm.score(X_test, y_test);
    % evaluate models MLP on validation set
    mlpDefaultAcc = mlpDefault.score(X_test, y_test);
    mlpTunedAcc = mlp.score(X_test, y_test);
    %% Print Performance
    fprintf('========================================== \n')
    fprintf('MODEL              Accuracy                \n')
    fprintf('========================================== \n')
    fprintf('MLP Default          %.2f%%                \n',mlpDefaultAcc*100)
    fprintf('MLP Optimized        %.2f%%                \n',mlpTunedAcc*100)
    fprintf('__________________________________________ \n')
    fprintf('SVM Default          %.2f%%                \n',svmDefaultAcc*100)
    fprintf('SVM Optimized        %.2f%%                \n',svmTunedAcc*100)
    fprintf('========================================== \n')
end

