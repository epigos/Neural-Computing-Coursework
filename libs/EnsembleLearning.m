% ************************************************************************
%                             ENSEMBLE LEARNING
% ************************************************************************

%The aim of this script is to use Ensembling to achieve higher efficiency.
%The method we seek is called Majority Voting Ensemble, where we train 6
%different models (3 SVMs and 3 MLPs) on all training data and make them
%vote for the new input class. We use a 10-fold Cross Validation to achieve
%more precise estimations of the results of the generalisation.
function EnsembleLearning(X,y, classNames)
    %% create the 10-fold partition
    kfolds = 10;
    cv = cvpartition(y, 'KFold', kfolds, 'Stratify', true);
    % define arrays to hold model scores
    mlp1Scores = zeros(1, kfolds);
    mlpEnScores = zeros(1, kfolds);
    svm1Scores = zeros(1, kfolds);
    svmEnsScores = zeros(1, kfolds);
    ensScores = zeros(1, kfolds);
    % start k-fold training process
    for k = 1:kfolds
        fprintf("Cross Validation Fold %d/%d \n", k, kfolds);
        % split data into training and validation set using the kth value
        X_train = X(cv.training(k), :);
        y_train = y(cv.training(k), :);
        X_test = X(cv.test(k), :);
        y_test = y(cv.test(k), :);
        %% train Models
        % MLP1 default inbuilt configuration
        mlp1 = MLP(X_train, y_train, classNames).fitDefault();
        % MLP2 classifier with model best parameters
        mlp2 = MLP(X_train, y_train, classNames).fit('TransferFcn', 'tansig');
        % MLP2 classifier with linear activation function 
        mlp3 = MLP(X_train, y_train, classNames).fit('TransferFcn', 'poslin');

        % SVM1 default inbuilt configuration
        svm1 = SVM(X_train, y_train, classNames).fitDefault();
        % SVM1 classifier with model best parameters
        svm2 = SVM(X_train, y_train, classNames).fit();
        % SVM2 classifier polynomial kernel 
        svm3 = SVM(X_train, y_train, classNames).fit('KernelFunction', 'polynomial');
        
        % make MLP predictions
        mlp1Pred = mlp1.predict(X_test);
        mlp2Pred = mlp2.predict(X_test);
        mlp3Pred = mlp3.predict(X_test);
        ensMLPPred = mode([mlp1Pred'; mlp2Pred'; mlp3Pred'])';
                
        % make SVM predictions
        svm1Pred = svm1.predict(X_test);
        svm2Pred = svm2.predict(X_test);
        svm3Pred = svm3.predict(X_test);
        ensSVMPred = mode([svm1Pred'; svm2Pred'; svm3Pred'])';
        % make ensemble of MLP and SVM
        allPred = [mlp1Pred'; mlp2Pred'; mlp3Pred'; svm1Pred'; svm2Pred'; svm3Pred'];
        ensPred = mode(allPred)';
        
        % compute accuracy
        mlp1Scores(k) = Utils.score(y_test, mlp1Pred);
        mlpEnScores(k) = Utils.score(y_test, ensMLPPred);
        svm1Scores(k) = Utils.score(y_test, svm1Pred);
        svmEnsScores(k) = Utils.score(y_test, ensSVMPred);
        ensScores(k) = Utils.score(y_test, ensPred);
    end
    % plot performance
    models = {'Single MLP', 'Ensemble MLP', 'Single SVM', 'Ensemble SVM', 'Multi-model Ensemble'};
    xlabels = categorical(models);
    xvalues = reordercats(xlabels, models);
    yvalues = [mean(mlp1Scores), mean(mlpEnScores), mean(svm1Scores),...
        mean(svmEnsScores), mean(ensScores)];
    
    figure('Name', 'Ensemble methods', 'pos', [100 100 600 480]);
    bar(xvalues, yvalues);
    title('Average Classification Accuracy');
    xlabel("Model");
    ylabel("Accuracy");
   
    %% Print Performance
    fprintf('====================================================\n')
    fprintf('MODEL                  CV Avg. Perf    CV Std. Perf \n')
    fprintf('====================================================\n')
    fprintf('Single MLP               %.2f%%            %.2f%%   \n', mean(mlp1Scores)*100, std(mlp1Scores)*100)
    fprintf('Ensemble MLP             %.2f%%            %.2f%%   \n', mean(mlpEnScores)*100, std(mlpEnScores)*100)
    fprintf('____________________________________________________\n')
    fprintf('Single SVM               %.2f%%            %.2f%%   \n', mean(svm1Scores)*100, std(svm1Scores)*100)
    fprintf('Ensemble SVM             %.2f%%            %.2f%%   \n', mean(svmEnsScores)*100, std(svmEnsScores)*100)
    fprintf('____________________________________________________\n')
    fprintf('Multi-model Ensemble     %.2f%%            %.2f%%   \n', mean(ensScores)*100, std(ensScores)*100)
    fprintf('====================================================\n')
end

