% ************************************************************************
%                   FINAL MODELS
% ************************************************************************

% This is to enable the examiner to test the models performance and 
% the submitted test data and models. 
% add libs folder to path
addpath(genpath("libs"));
% load the test data
load('data/test.mat');
% load saved models
load('models/final_models.mat');
% create one-hot-encoding for target values for use in MLP
targetOHE = dummyvar(targets);
targetFamily = 'Leptodactylidae';
%% Make predictions for mlp
mlpPred = mlpNet(inputs');
targetInd = vec2ind(targetOHE');
MLPPredInd = vec2ind(mlpPred);
% compute the accuracy for mlp
mlpTestScore = sum(targetInd == MLPPredInd)/numel(targetInd);
% compute confusion matrix
MLPConfusionMatrix = confusionmat(targetInd, MLPPredInd, 'Order', [1, 2]);
% compute F1-score
mlpF1 = Utils.classificationReport(MLPConfusionMatrix);
% compute ROC and AUC
propMLP = mlpPred';
[Xmlp,Ymlp,~,AUCmlp, optMLP] = perfcurve(targets,...
        propMLP(:, 1), targetFamily);
%% Make predictions for SVM
[svmPred, probSVM] = predict(svmModel, inputs);
% compute the accuracy for mlp
svmTestScore = sum(svmPred==targets)/length(svmPred);
% compute confusion matrix
SVMConfusionMatrix = confusionmat(targets, svmPred);
% compute ROC and AUC
[Xsvm,Ysvm,~,AUCsvm, optSVM] = perfcurve(targets,...
        probSVM(:, 2), targetFamily); 
% compute F1-score
svmF1 = Utils.classificationReport(SVMConfusionMatrix);

%% Print Performance metrics
fprintf("Confusion matrix for MLP\n");
disp(MLPConfusionMatrix);
fprintf("Confusion matrix for SVM\n");
disp(SVMConfusionMatrix);
fprintf("Classification Report\n");
fprintf('===========================================\n')
fprintf('MODEL    Accuracy      AUC       F1-score  \n')
fprintf('===========================================\n')
fprintf('MLP       %.2f%%      %.3f       %.3f  \n', mlpTestScore*100,...
    AUCmlp, mean(mlpF1))
fprintf('___________________________________________\n')
fprintf('SVM       %.2f%%      %.3f       %.3f  \n', svmTestScore*100,...
    AUCsvm, mean(svmF1))
fprintf('===========================================\n')

