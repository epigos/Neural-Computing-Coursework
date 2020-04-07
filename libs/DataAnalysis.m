% ************************************************************************
%                        DataAnalysis
% ************************************************************************

% This script performs Exploratory Data Analysis on the dataset. It
% visualizes the distribution of target families, Parrallel Coordinates and
% Scatter pairplot of target variable with Features.
function DataAnalysis(data, predictorNames)
    fprintf("Performing exploratory data analysis...\n");
    % Data Shape
    [rows, columns] = size(data);
    % define categorical column parameters
    catColumns = {'Family', 'Genus', 'Species'};
    % print out the data size and feature types
    fprintf("The data contains %d observations with %d columns\n\n", rows, columns);
    fprintf("The data set contains %d categorical and %d numerical features.\n",...
        length(catColumns), columns - length(catColumns));
    % show first 5 rows
    disp("Display the first 5 rows");
    head(data, 5)
    % summarize the data
    disp("Print summary of data table:");
    summary(data);
    %% Univariate visualizations
    
    % Distribution of Anuran Families - Original
    figure('Name', "Distribution of Anuran Families", 'pos', [10 400 800 400])
    subplot(1,2,1)
    histogram(categorical(data.Family));
    title('Distribution of Anuran Families - Original');
    ylabel('Number of Observations');
    % Distribution of Anuran Families - Regrouped
    subplot(1,2,2)
    histogram(categorical(data.FamilyGroup));
    title('Distribution of Anuran Families - Regrouped');
    ylabel('Number of Observations');
    
    %% Multivariate plots
    % Parralel coordinates of Target and Feature variables
    data = removevars(data, 'Family');
    figure('Name', "Parralel coordinates", 'pos', [10 400 1200 640]);
    parallelplot(data, 'GroupVariable', 'FamilyGroup');
    title('Parralel coordinates of Features and Target Variable');
    
    % Scatter Pairplot - Relationship with targets variable
    features = data(:, predictorNames);
    features = table2array(features);
    xnames = string(1:size(features, 2));

    figure('Name', "Scatter Pairplot", 'pos', [100 100 1200 800])
    gplotmatrix(features, [], data.FamilyGroup, lines(2),[],[],[],[],xnames);
    xtickangle(45);
    title('Scatter matrix of MFCCs colored by Anuran Families');
end

