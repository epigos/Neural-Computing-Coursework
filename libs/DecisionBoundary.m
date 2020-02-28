% Comparison of the shape of Decision Boundaries between MLP and SVM
% classifier
% We fit the classifiers using two chosen continuous attributes to be able
% to visualize in 2D and 3D

function DecisionBoundary(X, y, classNames)
    fprintf("Computing decision boundaries...\n");
    %% create feature grid
    % select 2 numerical columns
    cols = {5, 19};
    colNames = {'MFCCs_5', 'MFCCs_19'};
    xLabel = Utils.cleanLabel(colNames{1});
    yLabel = Utils.cleanLabel(colNames{2});
    % Define a grid of values in the observed predictor space. Predict the
    % posterior probabilities for each instance in the grid.
    colOne = X(:, cols{1});
    colTwo = X(:, cols{2});
    x1Max = max(colOne)+1; x1Min = min(colOne)-1; step1 = (x1Max-x1Min)/500;
    x2Max = max(colTwo)+1; x2Min = min(colTwo)-1; step2 = (x2Max-x2Min)/500;
    
    [x1Grid,x2Grid] = meshgrid(x1Min:step1:x1Max, x2Min:step2:x2Max);
    XGrid = [x1Grid(:),x2Grid(:)];
    %% Train models
    names = {'MLP','SVM'};
    X_train = [colOne, colTwo];
    % Train MLP classifier using the parameters obtained from
    % hyper-parameter tuning.
    mlp = MLP(X_train,y).fit();
    % Train SVM classifier using the parameters obtained from
    % hyper-parameter tuning.
    svm = SVM(X_train,y, classNames).fit();
    
    %% predict targe class
    % predict the labels and posterior probabilities for each observation
    % using all classifiers.
    % make predictions for MLP
    posteriorMLP = mlp.predict(XGrid);
    predMLP = MLP.labelsFromScores(posteriorMLP, classNames);
    % make predictions for SVM
    [predSVM, posteriorSVM] = predict(svm.model, XGrid);
    %% DECISION SURFACE
    % Visualize the Decision Surface for each classifier 
    % MLP
    t1 = 'Other';
    t2 = 'Leptodactylidae';
    figure('pos', [50 400 1200 400])
    % MLP
    subplot(1,2,1)
    gscatter(x1Grid(:), x2Grid(:), predMLP, 'rb');
    
    title("Decision Surface - "+names{1});
%     legend('Prediction Region 1', 'Prediction Region 2', t1, t2);
    xlabel(xLabel);
    ylabel(yLabel);
    axis tight
    % SVM
    subplot(1,2,2)
    hold on;
    gscatter(x1Grid(:), x2Grid(:), predSVM, 'rb');
    title("Decision Surface - "+names{2});
%     legend('Prediction Region 1', 'Prediction Region 2', 'Class 1', 'Class 2');
    xlabel(xLabel);
    ylabel(yLabel);
    axis tight
    hold off;
    %% Visualize the Probability Decision Boundary of MLP
    sz = size(x1Grid);
    % plot 2D projection of MLP
    figure('pos', [100 200 1200 400])
    subplot(1,2,1)
    hold on
    surf(x1Grid, x2Grid, reshape(posteriorMLP(1,:), sz),...
        'EdgeColor', 'none')
    surf(x1Grid, x2Grid, reshape(posteriorMLP(2,:), sz),...
        'EdgeColor', 'none')
    colorbar
    view(2)
    title("2D Classification Probability - " + names{1})
    xlabel(xLabel);
    ylabel(yLabel);
    axis tight
    hold off
    % plot 2d projection for SVM
    subplot(1, 2, 2)
    hold on
    surf(x1Grid, x2Grid, reshape(posteriorSVM(:,1), sz),...
        'EdgeColor', 'none')
    surf(x1Grid, x2Grid, reshape(posteriorSVM(:,2), sz),...
        'EdgeColor', 'none')
    colorbar
    view(2)
    title("2D Classification Probability - " + names{2})
    xlabel(xLabel);
    ylabel(yLabel);
    axis tight
    hold off
    %% Visualize the Probability Decision Boundary 3D

    % plot 3D projection for MLP
    figure('pos', [200 100 1200 400])
    subplot(1,2,1)
    hold on
    surf(x1Grid, x2Grid, reshape(posteriorMLP(1,:), sz),...
        'FaceColor', 'red', 'EdgeColor', 'none')
    surf(x1Grid, x2Grid, reshape(posteriorMLP(2,:), sz),...
        'FaceColor', 'blue', 'EdgeColor', 'none')
    alpha(0.4)
    view(3)
    title("3D Classification Probability - " + names{1})
    xlabel(xLabel);
    ylabel(yLabel);
    axis tight
    legend(classNames)
    % plot 3D for SVM.
    subplot(1, 2, 2)
    hold on
    surf(x1Grid, x2Grid, reshape(posteriorSVM(:,1), sz),...
        'FaceColor', 'red', 'EdgeColor', 'none')
    surf(x1Grid, x2Grid, reshape(posteriorSVM(:,2), sz),...
        'FaceColor', 'blue', 'EdgeColor', 'none')
    alpha(0.4)
    view(3)
    title("3D Classification Probability - " + names{2})
    xlabel(xLabel);
    ylabel(yLabel);
    axis tight
    legend(classNames)
    hold off
end

