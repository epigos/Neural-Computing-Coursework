function [mlp, svm] = HyperParameterTuning(X, y, classNames)
    %HYPERPARAMETERTUNING Summary of this function goes here
    %   Detailed explanation goes here
    [X_train, y_train, X_test, y_test] = Utils.train_test_split(X, y, 0.3);
    % train and optimize SVM regressor
    svm = SVM(X_train, y_train, classNames).optimize();
    % train and optimize MLP regressor
    mlp = MLP(X_train, y_train).optimize();
    
    % evaluate models SVM on validation set
    svmAcc = svm.score(X_test, y_test);
    % evaluate models MLP on validation set
    mlpAcc = mlp.score(X_test, y_test);
    
    fprintf("Classification accuracy for SVM is : %.2f \n", svmAcc);
    fprintf("Classification accuracy for MLP is : %.2f \n", mlpAcc);
    
    % Plot Minimum Objective Curves
%     figure('Name', 'Hyperparameter Tuning Performance Comparison',...
%         'pos', [10 10 1200 800])
%     subplot(3,2,[1,3])
%     hold on
%     % return the optimization results
%     mdls = {mlpRes, svm.HyperparameterOptimizationResults};
%     N = length(mdls);
%     for i = 1:N
%         % plot Minimum Objective Curve
%         plot(mdls{i}.ObjectiveMinimumTrace,'Marker','o','MarkerSize',5);
%     end
%     grid on
%     names = {'MLP', 'SVM'};
%     legend(names{1}, names{2},'Location','northeast')
%     title('Plot Minimum Objective Curves - Bayesian Optimization')
%     xlabel('Number of Iterations')
%     ylabel('Minimum Objective Value')
end

