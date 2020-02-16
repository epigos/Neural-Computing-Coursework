function [mlp, svm] = HyperParameterTuning(X, y, classNames)
    %HYPERPARAMETERTUNING Summary of this function goes here
    %   Detailed explanation goes here
    % define model names
    names = {'MLP', 'SVM'};
    [X_train, y_train, X_test, y_test] = Utils.train_test_split(X, y, 0.3);
    
    %% Tune model hyperparameters
    % train and optimize SVM classifier
    svm = SVM(X_train, y_train, classNames).optimize();
    % train and optimize MLP classifier
    mlp = MLP(X_train, y_train).optimize();
    
    %% Plot Minimum Objective Curves
   
    % return the optimization results
    mdls = {mlp, svm};
    
    N = length(mdls);
    for i = 1:N
        % plot Minimum Objective Curve
        plot(mdls{i}.ObjectiveMinimumTrace, @plotMinObjective);
    end
    grid on
    legend(names{1}, names{2},'Location','northeast')
    title('Plot Minimum Objective Curves - Bayesian Optimization')
    xlabel('Number of Iterations')
    ylabel('Minimum Objective Value')
    %% Check Performance with Test Set
    % evaluate models SVM on validation set
    svmAcc = svm.score(X_test, y_test);
    % evaluate models MLP on validation set
    mlpAcc = mlp.score(X_test, y_test);
    
    fprintf("Classification accuracy for SVM is : %.2f \n", svmAcc);
    fprintf("Classification accuracy for MLP is : %.2f \n", mlpAcc);
end

