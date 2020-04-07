% ************************************************************************
%                        UTILS - Helper Functions
% ************************************************************************

% This script contains class definition with static functions which are
% reusable throughout the code.
classdef Utils
    
    methods(Static)
        function [X_train, y_train, X_test, y_test] = train_test_split(X, y, holdout)
            % Split training data into train and test using holdout
            % cross-validation method.
            rng("default") % For reproducibility
            cv = cvpartition(size(y, 1), "Holdout", holdout);
            idx = cv.test;
            
            X_train = X(~idx, :);
            y_train = y(~idx, :);
            X_test = X(idx, :);
            y_test = y(idx, :);
        end
        
        function [X] = standardScalar(X)
            % standardized numerical data using zscore
            X.age_of_respondent = zscore(X.age_of_respondent);
            X.household_size = zscore(X.household_size);
        end
        
        function [cmNorm] = plotConfusionMatrix(y_test, labels, model_name)
            % Plot conusion matrix for model predictions
            cm = confusionchart(y_test, labels,...
                'ColumnSummary','column-normalized',...
                'RowSummary','row-normalized');
            title(model_name);
            fprintf("Confusion matrix for %s\n", model_name);
            cmNorm = cm.NormalizedValues;
            disp(cmNorm);
        end
        
        function [f1Score, recall, precision] = classificationReport(confusionMatrix)
            % Compute classification report from confusion matrix. This
            % returns the F1-score, recall and precision values.
            sz = size(confusionMatrix, 1);
            recall = zeros(1, sz);
            precision = zeros(1, sz);
            f1Score = zeros(1, sz);
            
            for i = 1:sz
                % calculate recall = TP / (TP + FP)
                recall(i)=confusionMatrix(i,i)/sum(confusionMatrix(i,:));
                % calculate precision = TP / (TP + FP)
                precision(i)=confusionMatrix(i,i)/sum(confusionMatrix(:,i));
                % calculate f1-score = 
                % 2 * ((Precision * Recall) / (Precision + Recall))
                f1Score(i) = 2 * ((precision(i) * recall(i))/(precision(i) + recall(i)));
            end
        end
        
        function data = getDummies(data, columns)
            % Creates One-Hot-Encoding for specified columns in data table.
            for i=1:numel(columns)
                colName = columns{i};
                rowData = data(:, colName);
                unique_values = unique(rowData);
                dv = dummyvar(table2array(rowData));
                
                varNames = strcat(colName, {'_'}, cellstr(unique_values.(1)));
                T = array2table(dv, 'VariableNames', varNames);
                
                data = [data T];
                data = removevars(data, colName);
            end
        end
        
        function [accuracy, loss] = score(targets, predictions)
            % Function to compute classification accuracy and loss
            accuracy = sum(targets == predictions)/numel(targets);
            loss = 1 - accuracy;
        end
        
        function bayesoptResultsToCSV(results, model)
            % Function to write bayesian optimization results to csv
            tbl = results.XTrace;
            tbl.('Validation Accuracy') = 1 - results.ObjectiveTrace;
            % write results to table
            writetable(tbl, sprintf('results/Bayesopts_%s.csv', model));
        end
    end
end

