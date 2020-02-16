% ************************************************************************
%                        FINAL MODEL - TEST COMPARISON
% ************************************************************************

% This script performs a final comparison on optimised models with a simple train-test split.
% Accuracies and confusion matrices are produced to compare each performance.

function PerformanceComparison(X,y, classNames, targetFamily)
     % define model names
    [X_train, y_train, X_test, y_test] = Utils.train_test_split(X, y, 0.3);
    
    %% Train the models with best parameters obtained from tuning
    % train SVM classifier
    svm = SVM(X_train, y_train, classNames).fit();
    % train and optimize MLP classifier
    mlp = MLP(X_train, y_train).fit();
    
    
    %% make predictions
    % define a prediction function that takes a trained model and make
    % predictions
    predictMLPFn = @() mlp.predict(X_test);
    predictSVMFn = @() predict(svm.model, X_test);
    
    % predict targets with SVM on test set
    [predSVM, posteriorSVM] = predictSVMFn();
    % predict targets MLP on validation set
    posteriorMLP = predictMLPFn();
    predMLP = MLP.labelsFromScores(posteriorMLP, classNames);
    %% Compute the standard ROC curve using the scores from the models
    
    [Xsvm,Ysvm,~,AUCsvm, optSVM] = perfcurve(y_test,...
        posteriorSVM(:, 2), targetFamily);
    
    posMLP = posteriorMLP';
    [Xmlp,Ymlp,~,AUCmlp, optMLP] = perfcurve(y_test, posMLP(:, 1), targetFamily);
    
    % Plot the ROC curves on the same graph.
    figure('Name', 'Performance comparision of classifiers',...
        'pos', [100 100 1200 600]);
    subplot(2, 2, [1, 3])
    plot(Xsvm,Ysvm, 'b')
    hold on
    plot(Xmlp,Ymlp, 'r')
    % plot Optimal operating point of the ROC curve
    plot(optSVM(1), optSVM(2),'bo')
    plot(optMLP(1), optMLP(2),'ro')
    grid on
    % write AUC on plot
    text(0.7,0.5,strcat('SVM AUC = ',num2str(AUCsvm)),'EdgeColor','b')
    text(0.7,0.45,strcat('MLP AUC = ',num2str(AUCmlp)),'EdgeColor','r')
    legend('SVM', 'MLP', 'SVM OPTROCPT',...
         'MLP OPTROCPT', 'Location','Best')
    
    xlabel('False positive rate'); ylabel('True positive rate');
    title('ROC Curves for SVM and MLP')
    hold off
    
    %% plot confusion metrics
    % visualize confusion metrics for SVM classifier
    subplot(2, 2, 2)
    Utils.plotConfusionMatrix(y_test, predSVM, 'SVM');
    
    % visualize confusion metrics for MLP classifier
    subplot(2, 2, 4)
    Utils.plotConfusionMatrix(y_test, predMLP, 'MLP');
end

