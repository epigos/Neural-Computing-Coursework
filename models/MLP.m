% ************************************************************************
%                        MLP class
% ************************************************************************

% This script contains class definition of functions to train, optimize and
% make predictions for MLP network. 
% E.g usage
%   mlp = MLP(inputs, targets, classLabels)
% To train the network, call:
%   mlp.fit()  
% To run hyper-parameter tuning:
%   mlp.optimize()
% To make predictions:
%   mlp.predict(testInput)
% To prediction scores:
%   mlp.score(testInput, testTargets)
classdef MLP
    % Creates a class which wraps the inbuilt patternnet classifier and
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
        % net: trained fitcsvm model - Instance of fitcsvm
        net
        % OptimizationResults: instance of BayesianOptimization - Bayesian
        % optimization results.
        OptimizationResults
    end
    
    methods
        function obj = MLP(X,y, classNames)
            %SVM Construct an instance of this class
            %   X:          training set features
            %   y:          training set targets
            %   classNames: class labels
            obj.X = X;
            % convert to one-hot targets
            obj.y = dummyvar(y);
            % assign classnames
            obj.classNames = classNames;
        end
        
        function obj = fit(obj, varargin)
            % Trains Pattern recognition network using the trainscg as the
            % default training function other parameters obtained during
            % hyper-parameter tuning process.
            p = inputParser;
            p.addParameter('HiddenNeurons', 20);
            p.addParameter('NetworkDepth', 2);
            p.addParameter('TransferFcn', 'tansig');
            p.addParameter('epochs', 200);
            p.addParameter('CrossVal', false);
            p.addParameter('cv', cvpartition(size(obj.y, 1), 'Holdout', 1/3));
            p.addParameter('CVfold', 0);
            p.addParameter('divideMode', false);
            parse(p, varargin{:});
            
            % Build Network
            networkDepth = p.Results.NetworkDepth;
            hiddenLayerSize = ones(1, networkDepth) * p.Results.HiddenNeurons;
            obj.net = patternnet(hiddenLayerSize); 
            % Early stopping after 6 consecutive increases of Validation Performance
            obj.net.trainParam.max_fail = 6;
            % Specify number of epochs
            obj.net.trainParam.epochs = p.Results.epochs;
            for i = 1:networkDepth
                obj.net.layers{i}.transferFcn = char(p.Results.TransferFcn); % Update Activation Function of Layers
            end
           
            % set cross validation parameters
            if p.Results.CrossVal
                % Divide Training Data into Train-Validation sets
                cv = p.Results.cv;
                k = p.Results.CVfold;
                rng = 1:cv.NumObservations;         
                if k 
                    obj.net.divideFcn = 'divideind';
                    obj.net.divideParam.trainInd = rng(cv.training(k));
                    obj.net.divideParam.valInd = rng(cv.test(k));
                else
                    obj.net.divideParam.trainRatio = rng(cv.training);
                    obj.net.divideParam.valRatio = rng(cv.test);
                    obj.net.divideParam.testRatio = 0;
                end
            elseif p.Results.divideMode
                obj.net.divideParam.trainRatio = .85; 
                obj.net.divideParam.valRatio = .15; 
                obj.net.divideParam.testRatio = 0;
            else
                obj.net.divideMode = 'none'; % Use all data for Training
            end
            
            % train network
            obj.net = train(obj.net, obj.X', obj.y');
        end
        
        function obj = optimize(obj, varargin)
            % Run hyper-paremeter tuning for MLP classifier and returns a
            % retrained model with the best model parameters.
            p = inputParser;
            p.addParameter('MaxObjectiveEvaluations', 30);
            parse(p, varargin{:});
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
            % Set up a partition for cross-validation. This step fixes the
            % train and test sets that the optimization uses at each step.
            % create holdout cross validation of 80-20% splits
            cv = cvpartition(size(obj.y, 1), 'Holdout', 0.2);
            % Define hyperparameters to optimize
            vars = [optimizableVariable('networkDepth', [1, 2], 'Type', 'integer');...
                optimizableVariable('hiddenNeurons', [5, 30], 'Type', 'integer');
                optimizableVariable('transferFcn', {'logsig', 'tansig'}, 'Type', 'categorical')];

            % initialize objective function for the network
            minfn = @(n)MLP.wrapFitNet(obj.X', obj.y', cv,...
                n.hiddenNeurons, n.networkDepth, n.transferFcn);
            % Optimize hyperparameters
            results = bayesopt(minfn, vars, 'UseParallel', useParallel,...
                'IsObjectiveDeterministic', false,...
                'MaxObjectiveEvaluations',p.Results.MaxObjectiveEvaluations,...
                'AcquisitionFunctionName', 'expected-improvement-plus');
            T = bestPoint(results);
            
            % set hyper-parameter search results
            obj.OptimizationResults = results;
            % save results to CSV
            Utils.bayesoptResultsToCSV(results, 'MLP');
            
            % Train final model on full training set using the best hyperparameters
            obj = obj.fit('HiddenNeurons', T.hiddenNeurons,...
                'NetworkDepth',T.networkDepth, 'TransferFcn', T.transferFcn);
            
        end
        
        function obj = fitDefault(obj)
            % Train a basic MLP classifier with default parameters            
            
            % Build Network
            obj.net = patternnet();
            obj.net = train(obj.net, obj.X', obj.y');
        end
        
        function [labels, scores] = predict(obj, inputs)
            % Make prediction and returns the labels and posterior
            % probabilities of the predictions.
            %  inputs: features for validation set
            scores = obj.net(inputs');
            ind = vec2ind(scores)';
           
            scores = scores';
            % convert output to it's categorical class names
            labels = categorical(ind, [2, 1], cellstr(obj.classNames));
        end
        function [acc, predicted] = score(obj, inputs, targets)
            % Evaluate model performance on validation set by computing
            % classification accuracy.
            %   inputs: features for validation set
            %   targets: targets for validation set
            
            % make predictions
            predicted = obj.predict(inputs);
            % compute classification accuracy
            acc = sum(targets == predicted)/numel(targets);
        end
    end
    
    methods(Static)
        function loss = wrapFitNet(inputs, targets, cv,...
            hiddenNeurons, networkDepth, transferFcn)
            % Objective Function that will be used in the bayesian
            % Optimization procedure for Hyper-Parameter tuning. It builds
            % a Neural Network and evaluates its performance on a Holdout
            % set and returns the classification loss.
            
            % Build Network Architecture
            hiddenLayerSize = ones(1, networkDepth) * hiddenNeurons;
            net = patternnet(hiddenLayerSize); 
            % Specify number of epochs
            net.trainParam.epochs = 500;
            % Early stopping after 6 consecutive increases of Validation Performance
            net.trainParam.max_fail = 6;
            for i = 1:networkDepth
                net.layers{i}.transferFcn = char(transferFcn); % Update Activation Function of Layers
            end
           
            % Divide Training Data into Train-Validation sets
            rng = 1:cv.NumObservations;
            net.divideFcn = 'divideind';
            net.divideParam.trainInd = rng(cv.training);
            net.divideParam.valInd = rng(cv.test);
            % Train Network
            net = train(net, inputs, targets);
            % Evaluate on test set and compute classification error
            % Evaluate on validation set and compute Classification Error
            outputs = net(inputs);
            tind = vec2ind(targets);
            yind = vec2ind(outputs);
            % compute classification loss
            loss = sum(tind(cv.test) ~= yind(cv.test))/numel(tind(cv.test));
        end
        
    end
end

