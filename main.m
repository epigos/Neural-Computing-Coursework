% Main entry point.

% Clear workspace and Command window
close all; clc; clear;
% add models folder to path
addpath(genpath("models"));
% add libs folder to path
addpath(genpath("libs"));
% data source https://fsdkenya.org/publication/finaccess2019/
dataPath = sprintf('%s/data/abalone.csv', pwd);
rawData = readtable(dataPath);

% define the target column
targetCol = 'Age';
%Pre process data by converting all categorical columns to categorical data
%type and reposition the target column to the end of the table.
[cleanData, X, y] = PreProcessing(rawData, targetCol);
%% Main entry points to run scripts

section = 1;
while section ~= 0
    %close all; clear all;
    fprintf('\nPlease, type a number between 1 and 6 to run the related script, or 0 to exit the program\n\n')
    fprintf('>> 1 : Exploratory Data Analysis\n')
    fprintf('>> 2 : Hyperparameter Tuning (warning: It may take a long time to complete.) \n')
    fprintf('>> 3 : Performance Comparison: MLP vs SVM\n')
    fprintf('>> 4 : Decision Boundaries: MLP vs SVM\n')
    fprintf('>> 5 : Learning Curve: MLP vs SVM\n')
    fprintf('Type 0 to exit the program ...\n\n')

    section = input('Enter a number: ');
    
    switch section
        case 1
            % Exploratory Data Analysis
            DataAnalysis(cleanData);
            pause(3)
        case 2
            % Hyperparameter Tuning of Models
            [mlp, svm] = HyperParameterTuning(X, y);
            pause(3)
        case 3
            % Performance of Models: MLP vs SVM
            PerformanceComparison(X, y);
            pause(3)
        case 4
            % Decision Boundaries: MLP vs SVM
            DecisionBoundary(X, y);
            pause(3)
        case 5
            % Learning curve of Models: MLP vs SVM
            LearningCurve(X, y)
            pause(3)
        case 0
            continue
        otherwise
            % if number is invalid
            fprintf('\nPlease pick a viable number between 1 and 5.\n')
            fprintf('Type 0 to exit the program ...\n\n')
            pause(1)
    end
end

% Clear workspace, console
close all; clc;
fprintf('Program exited.\n\n')