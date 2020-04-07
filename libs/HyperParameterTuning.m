% ************************************************************************
%                        HYPER-PARAMETER TUNING
% ************************************************************************

% This script performs hyper-parameter tuning of to find best model
% parameters. It saves the result to csv file in the results folder. Also,
% it trains a basic version of the models and prints the accuracies of both
% optimzed and basic model's performance.
function [mlp, svm] = HyperParameterTuning(X, y, classNames)
    % define model names to be tuned
    names = {'MLP', 'SVM'};
    % split data into training test of 90-10%. The training set is further
    % split into 80-20% for cross-validation during optimizing process.
    [X_train, y_train, X_test, y_test] = Utils.train_test_split(X, y, 0.1);
    
    %% train basic models with default parameters
    fprintf("Running models with default configuration\n");
    svmDefault = SVM(X_train, y_train, classNames).fitDefault();
    mlpDefault = MLP(X_train, y_train, classNames).fitDefault();
    
    %% Tune model hyperparameters
    % train and optimize MLP classifier
    fprintf("Optimizing models hyper-parameters\n");
    mlp = MLP(X_train, y_train, classNames).optimize('MaxObjectiveEvaluations', 200);
    % train and optimize SVM classifier
    svm = SVM(X_train, y_train, classNames).optimize('MaxObjectiveEvaluations', 200);
   
    %% Plot Minimum Objective Curves
   
    % plot Minimum Objective Curve
    figure;
    xvalues = 1:mlp.OptimizationResults.NumObjectiveEvaluations;
    plot(xvalues, mlp.OptimizationResults.ObjectiveMinimumTrace,'r',...
        xvalues, svm.OptimizationResults.ObjectiveMinimumTrace, 'b',...
        'LineWidth',2);
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

