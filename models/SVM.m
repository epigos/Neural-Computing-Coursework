% ************************************************************************
%                        SVM class
% ************************************************************************

% This script contains class definition of functions to train, optimize and
% make predictions for SVM model. 
% E.g usage
%   svm = SVM(inputs, targets, classLabels)
% To train the network, call:
%   svm.fit()  
% To run hyper-parameter tuning:
%   svm.optimize()
% To make predictions:
%   svm.predict(testInput)
% To prediction scores:
%   svm.score(testInput, testTargets)
classdef SVM
    % Creates a class which wraps the inbuilt fitcsvm classifier and
    % creates resuable functions for use in all the experiments.
    % Class inspiration: https://uk.mathworks.com/help/matlab/matlab_oop/create-a-simple-class.html
    properties
        % Defines the properties of the class
        % X: input features
        X
        % y: input targets
        y
        % classNames: class labels
        classNames
        % model: trained fitcsvm model - Instance of fitcsvm
        model
        % OptimizationResults: instance of BayesianOptimization - Bayesian
        % optimization results.
        OptimizationResults
    end
    
    methods
        function obj = SVM(X,y, classNames)
            %SVM Construct an instance of this class
            %   X:          training set features
            %   y:          training set targets
            %   classNames: class labels
            obj.X = X;
            % convert categorical array to cell array
            obj.y = cellstr(y);
            % assign classnames
            obj.classNames = classNames;
        end
        
        function obj = fit(obj, varargin)
            % Train SVM classifier using the guassian kernel
            % function and other parameters obtained during hyper-parameter
            % tuning process.
            p = inputParser;
            p.addParameter('BoxConstraint', 9);
            p.addParameter('KernelScale', 1.0001);
            p.addParameter('KernelFunction', 'rbf');
            parse(p, varargin{:});
            % For reproducibility
            rng default;
            % train svm model
            obj.model = fitcsvm(obj.X, obj.y,...
                'ClassNames', obj.classNames,...
                'KernelFunction', p.Results.KernelFunction,...
                'BoxConstraint', p.Results.BoxConstraint,...
                'KernelScale', p.Results.KernelScale);
        end
        
        function obj = optimize(obj, varargin)
            % Run hyper-paremeter tuning for SVM classifier and returns a
            % retrained model with the best model parameters.
            p = inputParser;
            p.addParameter('MaxObjectiveEvaluations', 30);
            parse(p, varargin{:});
            % Set up a partition for cross-validation. This step fixes the
            % train and test sets that the optimization uses at each step.
            % create holdout cross validation of 80-20% splits
            cv = cvpartition(size(obj.y, 1), 'Holdout', 0.2);
        
            try
                % Start a parallel pool
                poolobj = gcp;
                % 'UseParallel' as true to run Bayesian optimization in
                % parallel to speed up the process. Requires Parallel
                % Computing Toolbox.
                useParallel = true;
            catch
                % Disable parralel pool if functionality is not available
                useParallel = false;
            end
            % For reproducibility
            rng default;
            % Train and optimize SVM classifier using the guassian kernel
            % function and the hyper-parameter options.
            % Define hyperparameters to optimize
            vars = [optimizableVariable('C',[1, 10],'Type','integer');
                optimizableVariable('sigma', [1, 10], 'Transform', 'log')];
            % initialize objective function for the network
            minfn = @(v)SVM.wrapFitSVM(obj.X,obj.y,obj.classNames,cv,...
                v.C, v.sigma);
            
            % Optimize hyperparameters
            % set options to use Bayesian optimization. Use the same
            % cross-validation partition cv in all optimizations. For
            % reproducibility, we use the 'expected-improvement-plus'
            % acquisition function.
            results = bayesopt(minfn, vars, 'UseParallel', useParallel,...
                'IsObjectiveDeterministic', false,...
                'MaxObjectiveEvaluations',p.Results.MaxObjectiveEvaluations,...
                'AcquisitionFunctionName', 'expected-improvement-plus');
            T = bestPoint(results);
            
            % set hyper-parameter search results
            obj.OptimizationResults = results;
            % save results to CSV
            Utils.bayesoptResultsToCSV(results, 'SVM');
            % retrain the model with best parameters. 
            obj = obj.fit('BoxConstraint', T.C,...
                'KernelScale', T.sigma);
        end
        
        function obj = fitDefault(obj)
            % Train and SVM classifier with default parameters
            % For reproducibility
            rng default;
            
            % train svm model
            obj.model = fitcsvm(obj.X, obj.y,...
                'ClassNames', obj.classNames);
        end
        
        function [outputs, scores] = predict(obj, inputs)
            % Make prediction and returns the labels and posterior
            % probabilities of the predictions.
            %  inputs: features for validation set
            [outputs, scores] = predict(obj.model, inputs); 
        end
        
        function [acc, predicted]  = score(obj, inputs, targets)
            % Evaluate model performance on validation set by computing
            % classification accuracy.
            %   inputs: features for validation set
            %   targets: targets for validation set
            predicted = predict(obj.model, inputs);
            % compute classification accuracy
            acc = sum(targets == predicted)/numel(targets);
        end
    end
    methods(Static)
        function loss = wrapFitSVM(inputs, targets, classNames, cv,...
                C, sigma)
            % Objective Function that will be used in the bayesian
            % Optimization procedure for Hyper-Parameter tuning. It builds
            % an SVM and evaluates its performance on a Holdout
            % set and returns the classification loss.
            X_train = inputs(cv.training(), :);
            y_train = targets(cv.training(), :);
            X_val = inputs(cv.test(), :);
            y_val = targets(cv.test(), :);
            % Build onevsone SVM
           
            % train svm model
            svm = fitcsvm(X_train, y_train,...
                'ClassNames', classNames,...
                'KernelFunction', 'rbf',...
                'BoxConstraint', C,...
                'KernelScale', sigma);
            % make predictions
            outputs = predict(svm, X_val);
            % compute classification loss
            loss = sum(y_val ~= outputs)/numel(y_val);
        end
    end
end

