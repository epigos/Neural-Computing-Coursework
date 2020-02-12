function [mlp, svm] = HyperParameterTuning(X, y)
    %HYPERPARAMETERTUNING Summary of this function goes here
    %   Detailed explanation goes here
    
    % train and optimize SVM regressor
    svm = SVM(X, y).optimize();
    % train and optimize MLP regressor
    [mlp, mlpRes] = MLP(X, y).optimize();
    
    % Plot Minimum Objective Curves
    figure('Name', 'Hyperparameter Tuning Performance Comparison',...
        'pos', [10 10 1200 800])
    subplot(3,2,[1,3])
    hold on
    % return the optimization results
    mdls = {mlpRes, svm,HyperparameterOptimizationResults};
    N = length(mdls);
    for i = 1:N
        % plot Minimum Objective Curve
        plot(mdls{i}.ObjectiveMinimumTrace,'Marker','o','MarkerSize',5);
    end
    grid on
    names = {'MLP', 'SVM'};
    legend(names{1}, names{2},'Location','northeast')
    title('Plot Minimum Objective Curves - Bayesian Optimization')
    xlabel('Number of Iterations')
    ylabel('Minimum Objective Value')
end

