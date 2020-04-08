% ************************************************************************
%                        PERFORMANCE COMPARISON
% ************************************************************************

% This script performs a final comparison on optimised models with a simple
% train-test split. Accuracies, AUC, F1-SCORE and confusion matrices are
% produced to compare each performance.
function PerformanceComparison(X,y, classNames)
    % split data into training and test set of 80-20%
    [X_train, y_train, X_test, y_test] = Utils.train_test_split(X, y, 0.2);
    % save test data to file for submission.
%     inputs = X_test;
%     targets = y_test;
%     save('data/test.mat', 'inputs', 'targets');
    %% Train the models with best parameters obtained from tuning
    % train SVM classifier with optimized parameters
    svm = SVM(X_train, y_train, classNames).fit();
    % train and optimize MLP classifier with optimized parameters
    mlp = MLP(X_train, y_train, classNames).fit();
    % save models to file for submission
%     svmModel = svm.model;
%     mlpNet = mlp.net;
%     save('models/final_models.mat', 'mlpNet', 'svmModel');    
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
    
    targetFamily = 'Leptodactylidae';
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
    
    % visualize training confusion metrics for MLP classifier
    subplot(2, 2, 1)
    mlpTrainCM = Utils.plotConfusionMatrix(y_train, predTrainMLP, 'MLP Training');
    mlpTrainF1 = Utils.classificationReport(mlpTrainCM);
    % visualize test confusion metrics for MLP classifier
    subplot(2, 2, 2)
    mlpCM = Utils.plotConfusionMatrix(y_test, predTestMLP, 'MLP Test');
    mlpF1 = Utils.classificationReport(mlpCM);
    
    % visualize training confusion metrics for SVM classifier
    subplot(2, 2, 3)
    svmTrainCM = Utils.plotConfusionMatrix(y_train, predTrainSVM, 'SVM Training');
    svmTrainF1 = Utils.classificationReport(svmTrainCM);
    % visualize test confusion metrics for SVM classifier
    subplot(2, 2, 4)
    svmCM = Utils.plotConfusionMatrix(y_test, predTestSVM, 'SVM Test');
    svmF1 = Utils.classificationReport(svmCM);
    %% Print Performance metrics
    fprintf('==============================================\n')
    fprintf('MODEL        Accuracy      AUC      F1-score  \n')
    fprintf('==============================================\n')
    fprintf('MLP Train      %.2f%%      %.2f%%    %.2f%%   \n', mlpTrainScore*100,...
        AUCmlp*100, mean(mlpTrainF1)*100)
    fprintf('MLP Test       %.2f%%      %.2f%%    %.2f%%   \n', mlpTestScore*100,...
        AUCtmlp*100, mean(mlpF1)*100)
    fprintf('______________________________________________\n')
    fprintf('SVM Train      %.2f%%      %.2f%%    %.2f%%   \n', svmTrainScore*100,...
        AUCsvm*100, mean(svmTrainF1)*100)
    fprintf('SVM Test       %.2f%%      %.2f%%    %.2f%%   \n', svmTestScore*100,...
        AUCtsvm*100, mean(svmF1)*100)
    fprintf('==============================================\n')
end

