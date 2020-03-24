% Comparison of the shape of Decision Boundaries between MLP and SVM
% classifier
% We fit the classifiers using two chosen continuous attributes to be able
% to visualize in 2D and 3D

function DecisionBoundary(data, predictorNames, classNames)
    fprintf("Computing decision boundaries...\n");
    % define predictors and response variables
    X = table2array(data(:, predictorNames));
    y = categorical(data.FamilyGroup);
    %% create feature grid
    % select 2 numerical columns
    cols = {9, 11};
    xLabel = 'Feature 1';
    yLabel = 'Feature 2';
    % Define a grid of values in the observed predictor space. Predict the
    % posterior probabilities for each instance in the grid.
    colOne = X(:, cols{1});
    colTwo = X(:, cols{2});
    x1Max = max(colOne); x1Min = min(colOne); step1 = (x1Max-x1Min)/500;
    x2Max = max(colTwo); x2Min = min(colTwo); step2 = (x2Max-x2Min)/500;
    
    [x1Grid,x2Grid] = meshgrid(x1Min:step1:x1Max, x2Min:step2:x2Max);
    XGrid = [x1Grid(:),x2Grid(:)];
    %% Train models
    names = {'MLP','SVM'};
    X_train = [colOne, colTwo];
    % Train MLP classifier using the parameters obtained from
    % hyper-parameter tuning.
    mlp = MLP(X_train,y, classNames).fit();
    % Train SVM classifier using the parameters obtained from
    % hyper-parameter tuning.
    svm = SVM(X_train,y, classNames).fit();
    
    %% predict targe class
    % predict the labels and posterior probabilities for each observation
    % using all classifiers.
    % make predictions for MLP
    [predMLP, posteriorMLP] = mlp.predict(XGrid);
    % make predictions for SVM
    [predSVM, posteriorSVM] = svm.predict(XGrid);
    %% DECISION SURFACE
    % Visualize the Decision Surface for each classifier 
    % MLP
    t1 = string(classNames(1));
    t2 = string(classNames(2));
    % define columns rows for scatter plots
    x1_t1 = table2array(data(y==t1, cols{1})); x2_t1 = table2array(data(y==t1, cols{2}));
    x1_t2 = table2array(data(y==t2, cols{1})); x2_t2 = table2array(data(y==t2, cols{2}));
    % create plot window
    figure('pos', [50 400 1200 400])
    % MLP
    subplot(1,2,1)
    scatter(x1Grid(predMLP==t1), x2Grid(predMLP==t1), 1, [220, 20, 60]/255, 'filled', 'MarkerEdgeAlpha', .1)
    hold on;
    scatter(x1Grid(predMLP==t2), x2Grid(predMLP==t2), 1, [63, 0, 255]/255, 'filled', 'MarkerEdgeAlpha', .1)
    scatter(x1_t1, x2_t1, 20, [0.4660 0.6740 0.1880], 'filled');
    scatter(x1_t2, x2_t2, 20, [0.9290 0.6940 0.1250], 'filled');
    title("Decision Surface - "+names{1});
    legend('Prediction Region 1', 'Prediction Region 2', t1, t2, 'Location', 'Best');
    xlabel(xLabel);
    ylabel(yLabel);
    axis tight
    % SVM
    subplot(1,2,2)
    hold on;
    scatter(x1Grid(predSVM==t1), x2Grid(predSVM==t1), 1, [220, 20, 60]/255, 'filled', 'MarkerEdgeAlpha', .1)
    hold on;
    scatter(x1Grid(predSVM==t2), x2Grid(predSVM==t2), 1, [63, 0, 255]/255, 'filled', 'MarkerEdgeAlpha', .1)
    scatter(x1_t1, x2_t1, 20, [0.4660 0.6740 0.1880], 'filled');
    scatter(x1_t2, x2_t2, 20, [0.9290 0.6940 0.1250], 'filled');
    title("Decision Surface - "+names{2});
    legend('Prediction Region 1', 'Prediction Region 2', t1, t2, 'Location', 'Best');
    xlabel(xLabel);
    ylabel(yLabel);
    axis tight
    hold off;
    %% Visualize the Probability Decision Boundary of MLP
    sz = size(x1Grid);
    % plot 2D projection of MLP
    figure('pos', [150 150 1200 400])
    subplot(1,2,1)
    hold on
    surf(x1Grid, x2Grid, reshape(posteriorMLP(:,1), sz),...
        'EdgeColor', 'none')
    surf(x1Grid, x2Grid, reshape(posteriorMLP(:,2), sz),...
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
    figure('pos', [200 50 1200 400])
    subplot(1,2,1)
    hold on
    surf(x1Grid, x2Grid, reshape(posteriorMLP(:,1), sz),...
        'FaceColor', 'red', 'EdgeColor', 'none')
    surf(x1Grid, x2Grid, reshape(posteriorMLP(:, 2), sz),...
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

