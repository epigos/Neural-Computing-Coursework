function [cleanData, X, y, predictorNames] = PreProcessing(rawData, targetFamily) 
    
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
    % assign target variables by converting it to numeric first.
    %cleanData.FamilyGroup = ismember(cleanData.(familyCol), targetFamily);
    cleanData.FamilyGroup = cleanData.(familyCol);
    mask = ~ismember(cleanData.FamilyGroup, targetFamily);
    cleanData.FamilyGroup(mask) = {'Other'};
    y = categorical(cleanData.FamilyGroup);
end