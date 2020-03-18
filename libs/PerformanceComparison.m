% ************************************************************************
%                        FINAL MODEL - TEST COMPARISON
% ************************************************************************

% This script performs a final comparison on optimised models with a simple train-test split.
% Accuracies and confusion matrices are produced to compare each performance.

function PerformanceComparison(X,y, classNames, targetFamily)
    % define model names
    [X_train, y_train, X_test, y_test] = Utils.train_test_split(X, y, 1/3);
    
    %% Train the models with best parameters obtained from tuning
    % train SVM classifier
    svm = SVM(X_train, y_train, classNames).fit();
    % train and optimize MLP classifier
    mlp = MLP(X_train, y_train, classNames).fit();
    
    
    %% evaluate predictoins
    % predict targets for SVM on training set
    [predTrainSVM, probTrainSVM] = svm.predict(X_train);
    % predict targets for MLP on training set
    [predTrainMLP, probTrainMLP] = mlp.predict(X_train);
    
    % predict targets for SVM on test set
    [predTestSVM, probTestSVM] = svm.predict(X_test);
    % predict targets for MLP on test set
    [predTestMLP, probTestMLP] = mlp.predict(X_test);
    
    % compute scores for MLP
    mlpTrainScore = mlp.score(X_train, y_train);
    mlpTestScore = mlp.score(X_test, y_test);
    
    % compute scores for SVM
    svmTrainScore = svm.score(X_train, y_train);
    svmTestScore = svm.score(X_test, y_test);
    
    %% Compute the standard ROC curve using the scores from the models
    % compute training ROC curve for MLP and SVM
    [Xsvm,Ysvm,~,AUCsvm, optSVM] = perfcurve(y_train,...
        probTrainSVM(:, 2), targetFamily);
    
    [Xmlp,Ymlp,~,AUCmlp, optMLP] = perfcurve(y_train,...
        probTrainMLP(:, 1), targetFamily);
    
    % compute test ROC curve for MLP and SVM
    [Xtsvm,Ytsvm,~,AUCtsvm, opttSVM] = perfcurve(y_test,...
        probTestSVM(:, 2), targetFamily);
    
    [Xtmlp,Ytmlp,~,AUCtmlp, opttMLP] = perfcurve(y_test,...
        probTestMLP(:, 1), targetFamily);
    
    % Plot the ROC curves on the same graph.
    figure('Name', 'ROC for MLP vs SVM',...
        'pos', [20 300 780 420]);
    % train ROC
    subplot(1, 2, 1)

    % MLP
    plot(Xmlp,Ymlp, 'r--')
    hold on;
    % plot Optimal operating point of the ROC curve
    plot(optMLP(1), optMLP(2),'ro', 'MarkerSize', 10)
    % SVM
    plot(Xsvm,Ysvm, 'b-.')
    plot(optSVM(1), optSVM(2),'bo', 'MarkerSize', 10)
    grid on
    % write AUC on plot
    text(0.55,0.3,sprintf('SVM AUC = %.4f%',AUCsvm),'EdgeColor','b')
    text(0.55,0.4,sprintf('MLP AUC = %.4f%',AUCmlp),'EdgeColor','r')
    legend('MLP', 'MLP OPTROCPT', 'SVM',...
         'SVM OPTROCPT', 'Location','Best')
    
    xlabel('False positive rate'); ylabel('True positive rate');
    title('Training ROC')
    hold off
    
    % test ROC
    subplot(1, 2, 2)

    % MLP
    plot(Xtmlp,Ytmlp, 'r--')
    hold on;
    % plot Optimal operating point of the ROC curve
    plot(opttMLP(1), opttMLP(2),'ro', 'MarkerSize', 10)
    % SVM
    plot(Xtsvm,Ytsvm, 'b-.')
    plot(opttSVM(1), opttSVM(2),'bo', 'MarkerSize', 10)
    grid on
    % write AUC on plot
    text(0.55,0.3,sprintf('SVM AUC = %.4f%', AUCtsvm),'EdgeColor','b')
    text(0.55,0.4,sprintf('MLP AUC = %.4f%',AUCtmlp),'EdgeColor','r')
    legend('MLP', 'MLP OPTROCPT', 'SVM',...
         'SVM OPTROCPT', 'Location','Best')
    
    xlabel('False positive rate'); ylabel('True positive rate');
    title('Test ROC')
    hold off
    
    %% plot confusion metrics
    % Plot the ROC curves on the same graph.
    figure('Name', 'Confusion Matrix MLP vs SVM',...
        'pos', [580 150 780 540]);
    % visualize training confusion metrics for SVM classifier
    subplot(2, 2, 1)
    svmTrainCM = Utils.plotConfusionMatrix(y_train, predTrainSVM, 'SVM Training');
    [svmTrainRecall, svmTrainPrecision, svmTrainF1] = Utils.classificationReport(svmTrainCM);
    % visualize test confusion metrics for SVM classifier
    subplot(2, 2, 2)
    svmCM = Utils.plotConfusionMatrix(y_test, predTestSVM, 'SVM Test');
    [svmRecall, svmPrecision, svmF1] = Utils.classificationReport(svmCM);
    % visualize training confusion metrics for MLP classifier
    subplot(2, 2, 3)
    mlpTrainCM = Utils.plotConfusionMatrix(y_train, predTrainMLP, 'MLP Training');
    [mlpTrainRecall, mlpTrainPrecision, mlpTrainF1] = Utils.classificationReport(mlpTrainCM);
    % visualize test confusion metrics for MLP classifier
    subplot(2, 2, 4)
    mlpCM = Utils.plotConfusionMatrix(y_test, predTestMLP, 'MLP Test');
    [mlpRecall, mlpPrecision, mlpF1] = Utils.classificationReport(mlpCM);
    
    %% Print Performance metrics
    fprintf('==========================================================\n')
    fprintf('MODEL        Accuracy    Recall    Precision    F1-score  \n')
    fprintf('==========================================================\n')
    fprintf('MLP Train      %.2f%%      %.2f%%    %.2f%%       %.2f%%  \n', mlpTrainScore*100,...
        mean(mlpTrainRecall)*100, mean(mlpTrainPrecision)*100, mean(mlpTrainF1)*100)
    fprintf('MLP Test       %.2f%%      %.2f%%    %.2f%%       %.2f%%  \n', mlpTestScore*100,...
        mean(mlpRecall)*100, mean(mlpPrecision)*100, mean(mlpF1)*100)
    fprintf('__________________________________________________________\n')
    fprintf('SVM Train      %.2f%%      %.2f%%    %.2f%%       %.2f%%  \n', svmTrainScore*100,...
        mean(svmTrainRecall)*100, mean(svmTrainPrecision)*100, mean(svmTrainF1)*100)
    fprintf('SVM Test       %.2f%%      %.2f%%    %.2f%%       %.2f%%  \n', svmTestScore*100,...
        mean(svmRecall)*100, mean(svmPrecision)*100, mean(svmF1)*100)
    fprintf('==========================================================\n')
end

