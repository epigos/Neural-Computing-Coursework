classdef MLP
    %MLP Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        X
        y
    end
    
    methods
        function obj = MLP(X,y)
            %MLP Construct an instance of this class
            %   Detailed explanation goes here
            obj.X = X;
            obj.y = y;
        end
        
        function net = train(obj, varargin)
            %METHOD1 Summary of this method goes here
            %   Detailed explanation goes here
            net = patternnet(10, 'traingd');
            net.trainParam.lr = 0.1;
            net = train(net, obj.X', obj.y');
        end
        
        function [net, results] = optimize(obj)
            % Improve the speed of a Bayesian optimization by using
            % parallel objective function evaluation.
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
            
            % Define a train/validation split to use inside the objective function
            cv = cvpartition(numel(obj.y), 'Holdout', 1/3);
            % Define hyperparameters to optimize
            vars = [optimizableVariable('networkDepth', [1, 3], 'Type', 'integer');
                    optimizableVariable('hiddenLayerSize', [1, 20], 'Type', 'integer');
                    optimizableVariable('lr', [1e-3 1], 'Transform', 'log');
                    optimizableVariable('momentum', [0.8 0.95]);
                    optimizableVariable('trainFcn', {'traingda', 'traingdm', 'traingdx', 'trainscg', 'trainbr', 'trainlm',}, 'Type', 'categorical');
                    optimizableVariable('transferFcn', {'logsig', 'poslin', 'tansig', 'purelin'}, 'Type', 'categorical')];

            % initialize objective function for the network
            minfn = @(n)MLP.wrapFitNet(obj.X', obj.y', cv,...
                n.networkDepth, n.hiddenLayerSize,...
                n.lr, n.momentum, n.trainFcn, n.transferFcn);
            % Optimize hyperparameters
            results = bayesopt(minfn, vars, 'UseParallel', useParallel,...
                'IsObjectiveDeterministic', false,...
                'AcquisitionFunctionName', 'expected-improvement-plus');
            T = bestPoint(results);
            
            % Train final model on full training set using the best hyperparameters
            hiddenLayerSize = ones(1, T.networkDepth) * T.hiddenLayerSize;
            net = fitnet(hiddenLayerSize, char(T.trainFcn));
            net.trainParam.lr = T.lr; % Update Learning Rate (if any)
            net.trainParam.mc = T.momentum; % Update Momentum Constant (if any)
            net.divideMode = 'none'; % Use all data for Training
            for i = 1:T.networkDepth
                % Update Activation Function of Layers
                net.layers{i}.transferFcn = char(T.transferFcn); 
            end
            % train network
            net = train(net, obj.X', obj.y');
            % use mean squared error as the re-train performance metrics
            net.performFn = 'mse';
            % compute network performance
            rmse = sqrt(perform(net, obj.X', obj.y'));
            
            fprintf("Root mean squared for MLP is : %.2f%\n", rmse);
        end
    end
    
    methods(Static)
        function loss = wrapFitNet(X, y, cv,...
                hiddenLayerSize, networkDepth, lr, momentum,...
                trainFcn, transferFcn)
            % Objective Function that will be used in the bayesian
            % Optimization procedure for Hyper-Parameter tuning. It builds
            % a Neural Network and evaluates its performance on a Holdout
            % set and returns the classification loss.
            
            % Build Network Architecture
            
            % Define vector of Hidden Layer Size (Network Architecture)
            hiddenLayerSize = hiddenLayerSize * ones(1, networkDepth);
            % Build Network
            net = fitnet(hiddenLayerSize, char(trainFcn)); 
            % Specify number of epochs
            net.trainParam.epochs = 1000;
            % Early stopping after 6 consecutive increases of Validation Performance
            net.trainParam.max_fail = 6;
            net.trainParam.lr = lr; % Update Learning Rate
            net.trainParam.mc = momentum; % Update Learning Rate
            % Update Activation Function of Layers
            for i = 1:networkDepth
                net.layers{i}.transferFcn = char(transferFcn); 
            end
            
            % Divide Training Data into Train-Validation sets
            rng = 1:cv.NumObservations;
            net.divideFcn = 'divideind';
            net.divideParam.trainInd = rng(cv.training);
            net.divideParam.valInd = rng(cv.test);
            % Train Network
            net = train(net, X, y);
            % Evaluate on test set and compute classification error
            % Evaluate on validation set and compute Classification Error
            loss = MLP.netLoss(net, X(:, cv.test), y(cv.test));
        end
        
        function loss = netLoss(net, inputs, targets)
            % Evaluate network performance on validation set by computing
            % rmse.
            predicted = net(inputs);
            loss = sqrt(mean((predicted - targets).^2));
        end
    end
end

