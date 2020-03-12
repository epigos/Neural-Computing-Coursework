classdef MLP
    %MLP Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        X
        y
        net
        ObjectiveMinimumTrace
    end
    
    methods
        function obj = MLP(X,y)
            %MLP Construct an instance of this class
            %   Detailed explanation goes here
            obj.X = X;
            % convert to one-hot targets
            obj.y = dummyvar(y);
        end
        
        function obj = fit(obj, varargin)
            %METHOD1 Summary of this method goes here
            %   Detailed explanation goes here
            % handle input variables
            p = inputParser;
            p.addParameter('NetworkDepth', 1);
            p.addParameter('HiddenLayerSize', 13);
            p.addParameter('Lr', 0.062977);
            p.addParameter('Momentum', 0.062977);
            p.addParameter('TrainFcn', 'traingdm');
            p.addParameter('TransferFcn', 'tansig');
            p.addParameter('epochs', 500);
            p.addParameter('trainNet', true);
            p.addParameter('CrossVal', false);
            p.addParameter('cv', cvpartition(size(obj.y, 1), 'Holdout', 1/3));
            p.addParameter('CVfold', 0);
            parse(p, varargin{:});
            
            % Train final model on full training set using the best hyperparameters
            hiddenLayerSize = ones(1, p.Results.NetworkDepth) * p.Results.HiddenLayerSize;
            % Build Network
            obj.net = patternnet(hiddenLayerSize, char(p.Results.TrainFcn)); 
            % Specify number of epochs
            obj.net.trainParam.epochs = p.Results.epochs;
            obj.net.trainParam.lr = p.Results.Lr; % Update Learning Rate (if any)
            obj.net.trainParam.mc = p.Results.Momentum; % Update Momentum Constant (if any)
            
            % set cross validation parameters
            if p.Results.CrossVal
                % Divide Training Data into Train-Validation sets
                cv = p.Results.cv;
                k = p.Results.CVfold;
                if k
                    rng = 1:cv.NumObservations;
                    obj.net.divideFcn = 'divideind';
                    obj.net.divideParam.trainInd = rng(cv.training(k));
                    obj.net.divideParam.testInd = rng(cv.test(k));
                else
                    obj.net.divideParam.trainRatio = 0.85;
                    obj.net.divideParam.valRatio = 0.15;
                    obj.net.divideParam.testRatio = 0;
                end
            else
                obj.net.divideMode = 'none'; % Use all data for Training
            end
            % update activation functions
            for i = 1:p.Results.NetworkDepth
                % Update Activation Function of Layers
                obj.net.layers{i}.transferFcn = char(p.Results.TransferFcn); 
            end
            
            % train network
            if p.Results.trainNet == true
                obj.net = train(obj.net, obj.X', obj.y');
            end
        end
        
        function obj = optimize(obj, varargin)
            % Improve the speed of a Bayesian optimization by using
            % parallel objective function evaluation.
            p = inputParser;
            p.addParameter('MaxObjectiveEvaluations', 200);
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
            % Define a train/validation split to use inside the objective function
            cv = cvpartition(size(obj.y, 1), 'Holdout', 1/3);
            % Define hyperparameters to optimize
            vars = [optimizableVariable('networkDepth', [1, 4], 'Type', 'integer');
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
                'MaxObjectiveEvaluations',p.Results.MaxObjectiveEvaluations,...
                'AcquisitionFunctionName', 'expected-improvement-plus');
            T = bestPoint(results);
            
            % set hyper-parameter search results
            obj.ObjectiveMinimumTrace = results;
            
            % Train final model on full training set using the best hyperparameters
            obj = obj.fit('NetworkDepth', T.networkDepth,...
                'HiddenLayerSize', T.hiddenLayerSize,...
                'Lr',T.lr, 'Momentum', T.momentum,...
                'TrainFcn', T.trainFcn,'TransferFcn', T.transferFcn);
            
        end
        function outputs = predict(obj, inputs)
           outputs = obj.net(inputs'); 
        end
        function acc = score(obj, inputs, targets)
            % Evaluate network performance on validation set by computing
            % classification accuracy.
            
            % make predictions
            predicted = obj.predict(inputs);
            
            targets = dummyvar(targets);
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
            % Early stopping after 5 consecutive increases of Validation Performance
            net.trainParam.max_fail = 5;
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
        
        function labels = labelsFromScores(scores, classNames)
            ind = vec2ind(scores)';
            labels = categorical(ind, [2, 1], cellstr(classNames));
        end
    end
end

