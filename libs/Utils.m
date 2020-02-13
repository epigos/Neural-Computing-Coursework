classdef Utils
    %UTILS Summary of this class goes here
    %   Detailed explanation goes here
    
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
        
        function [cm] = plotConfusionMatrix(y_test, labels, model_name)
            % Plot conusion matrix for model predictions
            cm = confusionchart(y_test, labels);
            txt = sprintf("Confusion matrix for %s", model_name);
            title(txt);
            disp(txt);
            cm.NormalizedValues
        end
        
        function [label] = cleanLabel(colName)
            % Function to replace underscore with spaces in column names
            % for labeling axis in charts.
           label = strrep(colName, '_', ' ');
        end
        
        function data = getDummies(data, columns)
            
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
    end
end

