classdef MLP
    %MLP Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        X
        y
        net
    end
    
    methods
        function obj = MLP(X,y)
            %MLP Construct an instance of this class
            %   Detailed explanation goes here
            obj.X = X;
            % convert to one-hot targets
            obj.y = dummyvar(y);
        end
        
        function obj = train(obj, varargin)
            %METHOD1 Summary of this method goes here
            %   Detailed explanation goes here
            obj.net = patternnet(10, 'traingd');
            obj.net.trainParam.lr = 0.1;
            obj.net = train(obj.net, obj.X', obj.y');
        end
        
        function obj = optimize(obj)
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
            cv = cvpartition(size(obj.y, 1), 'Holdout', 1/3);
            % Define hyperparameters to optimize
            vars = [optimizableVariable('networkDepth', [1, 3], 'Type', 'integer');
                optimizableVariable('hiddenLayerSize', [1, 50], 'Type', 'integer');
                optimizableVariable('lr', [1e-3 1], 'Transform', 'log');
                optimizableVariable('momentum', [0.8 0.95]);
                optimizableVariable('trainFcn', {'traingda', 'traingdm', 'traingdx', 'trainscg', 'trainoss'}, 'Type', 'categorical');
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
            obj.net = patternnet(hiddenLayerSize, char(T.trainFcn));
            obj.net.trainParam.lr = T.lr; % Update Learning Rate (if any)
            obj.net.trainParam.mc = T.momentum; % Update Momentum Constant (if any)
            obj.net.divideMode = 'none'; % Use all data for Training
            for i = 1:T.networkDepth
                % Update Activation Function of Layers
                obj.net.layers{i}.transferFcn = char(T.transferFcn); 
            end
        end
        function acc = score(obj, inputs, targets)
            % Evaluate network performance on validation set by computing
            % rmse.
            % train network with evaluation set
            targets = dummyvar(targets);
            obj.net = train(obj.net, inputs', targets');
            % make predictions
            predicted = obj.net(inputs');
            targetInd = vec2ind(targets');
            predInd = vec2ind(predicted);
            acc = sum(targetInd == predInd)/numel(targetInd);
        end
    end
    
    methods(Static)
        function loss = wrapFitNet(inputs, targets, cv,...
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
            net = patternnet(hiddenLayerSize, char(trainFcn)); 
            % Specify number of epochs
            net.trainParam.epochs = 500;
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
            net = train(net, inputs, targets);
            % Evaluate on test set and compute classification error
            % Evaluate on validation set and compute Classification Error
            outputs = net(inputs);
            tind = vec2ind(targets);
            yind = vec2ind(outputs);
            loss = sum(tind(cv.test) ~= yind(cv.test))/numel(tind(cv.test));
        end
    end
end

