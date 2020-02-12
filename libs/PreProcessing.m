function [cleanData, X, y] =  PreProcessing (rawData, targetCol) 
    
    fprintf("Preprocessing data...\n");
    % define categorical column parameters
    catColumns = {'Sex'};
    % convert columns to categorical data type
    cleanData = convertvars(rawData, catColumns, 'categorical');
    
    % check if there are missing values
    missingValues = any(ismissing(cleanData));
    if missingValues
        fprintf("There are %d missing values in the data set.", length(missingValues));
    else
        fprintf("There are no missing values in the data set.");
    end
    fprintf("\n\n");
    % create age (target) column by adding 1.5 to Rings value
    cleanData.(targetCol) = cleanData.Rings + 1.5;
    % drop unused Rings column
    cleanData = removevars(cleanData, {'Rings'});
    % get column names
    columnNames = cleanData.Properties.VariableNames;
    % seperate features and target variables
    targetFilter = ismember(columnNames, targetCol);
    predictorNames = cleanData.Properties.VariableNames(~targetFilter);
    % create one hot encoding of categorical variables
    X = Utils.getDummies(cleanData(:, predictorNames), catColumns);
    X = table2array(X);
    % assign target variables
    y = cleanData.(targetCol);
end