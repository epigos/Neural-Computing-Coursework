% Comparison of the shape of Decision Boundaries between MLP and SVM
% classifier
% We fit the classifiers using two chosen continuous attributes to be able
% to visualize in 2D and 3D

function DecisionBoundary(X, y, classNames)
    fprintf("Computing decision boundaries...\n");
    %% create feature grid
    % select 2 numerical columns
    cols = {5, 9};
    colNames = {'MFCCs_5', 'MFCCs_9'};
    xLabel = Utils.cleanLabel(colNames{1});
    yLabel = Utils.cleanLabel(colNames{2});
    % Define a grid of values in the observed predictor space. Predict the
    % posterior probabilities for each instance in the grid.
    x1Max = max(X(:, cols{1})); x1Min = min(X(:, cols{1}));
    x2Max = max(X(:, cols{2})); x2Min = min(X(:, cols{2}));
    step = 0.01;
    [x1Grid,x2Grid] = meshgrid(x1Min:step:x1Max, x2Min:step:x2Max);
    XGrid = [x1Grid(:),x2Grid(:)];
    %% Train models
    names = {'MLP','SVM'};
    % Train MLP classifier using the parameters obtained from
    % hyper-parameter tuning.
    mlp = MLP(X,y).fit('trainNet', false);
    net = train(mlp.net, XGrid', y');
    % Train SVM classifier using the parameters obtained from
    % hyper-parameter tuning.
    svm = SVM(X,y).fit();
    %% Visualize training feature space
    % Visualize the scatterplot of the two selected features 
    % colored by class labels
    figure('Name', 'Decision Surface for each classifier',...
        'pos', [300 200 1200 600])
    subplot(2, 2, [1, 3])
    gscatter(X(:, cols{1}), X(:, cols{2}), y,...
        [0 0.4470 0.7410; 0.8500 0.3250 0.0980], 'xo')
    title(sprintf('Plot of %s vs %s with target class', yLabel, xLabel))
    % label axis.
    xlabel(xLabel);
    ylabel(yLabel);
    hold on
    
    %% predict targe class
    % predict the labels and posterior probabilities for each observation
    % using all classifiers.
    % make predictions for MLP
    [predMLP, posteriorMLP] = net(XGrid');
    % make predictions for SVM
    [predSVM, posteriorSVM] = predict(svm, XGrid);
    
    %% DECISION SURFACE
    % Visualize the Decision Surface for each classifier 
    % MLP
    subplot(2, 2, 2)
    gscatter(x1Grid(:), x2Grid(:), predMLP, 'rb');
    title(names{1})
    xlabel(xLabel);
    ylabel(yLabel);
    % SVM
    subplot(2, 2, 4)
    gscatter(x1Grid(:), x2Grid(:), predSVM, 'rb');
    title(names{2})
    xlabel(xLabel);
    ylabel(yLabel);
    
    %% Visualize the Probability Decision Boundary of MLP
    sz = size(x1Grid);
    % plot 2D projection of MLP
    figure('Name', 'PREDICTED PROBABILITIES OF CLASSIFICATION MODELS',...
        'pos', [10 10 1200 600])
    subplot(2, 2, 1)
    hold on
    surf(x1Grid, x2Grid, reshape(posteriorMLP(:,1), sz),...
        'EdgeColor', 'none')
    surf(x1Grid, x2Grid, reshape(posteriorMLP(:,2), sz),...
        'EdgeColor', 'none')
    colorbar
    view(2)
    title(sprintf('Probability Decision Boundary - %s', names{1}))
    xlabel(xLabel);
    ylabel(yLabel);
    hold off
    % plot 3D projection for MLP
    subplot(2, 2, 2)
    hold on
    surf(x1Grid, x2Grid, reshape(posteriorMLP(:,1), sz),...
        'FaceColor', 'red', 'EdgeColor', 'none')
    surf(x1Grid, x2Grid, reshape(posteriorMLP(:,2), sz),...
        'FaceColor', 'blue', 'EdgeColor', 'none')
    alpha(0.2)
    view(3)
    title(sprintf('Posterior Probability Distribution of each Class - %s', names{1}))
    xlabel(xLabel);
    ylabel(yLabel);
    legend(classNames)
    hold off
    
    %% Visualize the Probability Decision Boundary for SVM
    % plot 2d projection for SVM
    subplot(2, 2, 3)
    hold on
    surf(x1Grid, x2Grid, reshape(posteriorSVM(:,1), sz),...
        'EdgeColor', 'none')
    surf(x1Grid, x2Grid, reshape(posteriorSVM(:,2), sz),...
        'EdgeColor', 'none')
    colorbar
    view(2)
    title(sprintf('Probability Decision Boundary - %s', names{1}))
    xlabel(xLabel);
    ylabel(yLabel);
    hold off
    % plot 3D for SVM.
    subplot(2, 2, 4)
    hold on
    surf(x1Grid, x2Grid, reshape(posteriorSVM(:,1), sz),...
        'FaceColor', 'red', 'EdgeColor', 'none')
    surf(x1Grid, x2Grid, reshape(posteriorSVM(:,2), sz),...
        'FaceColor', 'blue', 'EdgeColor', 'none')
    alpha(0.2)
    view(3)
    title(sprintf('Posterior Probability Distribution of each Class - %s', names{1}))
    xlabel(xLabel);
    ylabel(yLabel);
    legend(classNames)
    hold off
end

