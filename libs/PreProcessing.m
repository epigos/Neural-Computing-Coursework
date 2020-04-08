% ************************************************************************
%                        Preprocessing
% ************************************************************************

% This script performs a initial preprocessing of dataset such as removing 
% unused columns,checking for missing values, spliting data into features
% and response variables.
function [cleanData, X, y, predictorNames] = PreProcessing(rawData) 
    
    fprintf("Preprocessing data...\n");
    % define categorical columns
    % remove unused columns
    catColumns = {'Family', 'Genus', 'Species'};
    cleanData = convertvars(rawData, catColumns, 'categorical');
    
    % check if there are missing values
    missingValues = any(ismissing(cleanData));
    if missingValues
        fprintf("There are %d missing values in the data set.", length(missingValues));
    else
        fprintf("There are no missing values in the data set.");
    end
    fprintf("\n\n");
    % remove unused columns
    colsToRemove = {'RecordID', 'Genus', 'Species'};
    cleanData = removevars(rawData, colsToRemove);
    % get column names
    columnNames = cleanData.Properties.VariableNames;
    % seperate features and target variables
    familyCol = 'Family';
    targetFilter = ismember(columnNames, familyCol);
    predictorNames = cleanData.Properties.VariableNames(~targetFilter);
    X = table2array(cleanData(:, predictorNames));
    % Regroup family class to binary classification tasks
    cleanData.FamilyGroup = cleanData.(familyCol);
    mask = ~ismember(cleanData.FamilyGroup, 'Leptodactylidae');
    cleanData.FamilyGroup(mask) = {'Other'};
    y = categorical(cleanData.FamilyGroup);
    % add noise to the results by randomly switch 30% of the
    % classifications.
    sz = numel(y);
    % create random sample of indices of dataset
    idx = randsample(sz, floor(sz*0.30));
    % exchange classes
    for k = 1:numel(idx)
       if y(k) == "Leptodactylidae"
           y(k) = "Other";
       else
           y(k) = "Leptodactylidae";
       end
    end
    
    cleanData.FamilyGroup = y;
    y = categorical(y);
end