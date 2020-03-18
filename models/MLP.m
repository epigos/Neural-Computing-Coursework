classdef MLP
    %MLP Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        X
        y
        classNames
        net
        ObjectiveMinimumTrace
    end
    
    methods
        function obj = MLP(X,y, classNames)
            %MLP Construct an instance of this class
            %   Detailed explanation goes here
            obj.X = X;
            % convert to one-hot targets
            obj.y = dummyvar(y);
            % assign classnames
            obj.classNames = classNames;
        end
        
        function obj = fit(obj, varargin)
            %METHOD1 Summary of this method goes here
            %   Detailed explanation goes here
            % handle input variables
            p = inputParser;
            p.addParameter('HiddenLayerSize', 17);
            p.addParameter('Lr', 0.003952);
            p.addParameter('Momentum', 0.80065);
            p.addParameter('TrainFcn', 'trainscg');
            p.addParameter('epochs', 500);
            p.addParameter('trainNet', true);
            p.addParameter('CrossVal', false);
            p.addParameter('cv', cvpartition(size(obj.y, 1), 'Holdout', 1/3));
            p.addParameter('CVfold', 0);
            p.addParameter('divideMode', false);
            parse(p, varargin{:});
            
            % Build Network
            obj.net = patternnet(p.Results.HiddenLayerSize, char(p.Results.TrainFcn)); 
            % Specify number of epochs
            obj.net.trainParam.epochs = p.Results.epochs;
            obj.net.trainParam.lr = p.Results.Lr; % Update Learning Rate (if any)
            obj.net.trainParam.mc = p.Results.Momentum; % Update Momentum Constant (if any)
            
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
            cv = cvpartition(size(obj.y, 1), 'Holdout', 0.2);
            % Define hyperparameters to optimize
            vars = [optimizableVariable('hiddenLayerSize', [1, 20], 'Type', 'integer');
                optimizableVariable('lr', [1e-3 1], 'Transform', 'log');
                optimizableVariable('momentum', [0.8 0.95]);
                optimizableVariable('trainFcn', {'trainscg', 'traingda', 'traingdm', 'traingdx'}, 'Type', 'categorical')];

            % initialize objective function for the network
            minfn = @(n)MLP.wrapFitNet(obj.X', obj.y', cv,...
                n.hiddenLayerSize,...
                n.lr, n.momentum, n.trainFcn);
            % Optimize hyperparameters
            results = bayesopt(minfn, vars, 'UseParallel', useParallel,...
                'IsObjectiveDeterministic', false,...
                'MaxObjectiveEvaluations',p.Results.MaxObjectiveEvaluations,...
                'AcquisitionFunctionName', 'expected-improvement-plus',...
                'OutputFcn', {@saveToFile},...
                'SaveFileName', 'results/Bayesopts/mlp.mat');
            T = bestPoint(results);
            
            % set hyper-parameter search results
            obj.ObjectiveMinimumTrace = results;
            
            % Train final model on full training set using the best hyperparameters
            obj = obj.fit('HiddenLayerSize', T.hiddenLayerSize,...
                'Lr',T.lr, 'Momentum', T.momentum,...
                'TrainFcn', T.trainFcn);
            
        end
        function [labels, scores] = predict(obj, inputs)
           scores = obj.net(inputs');
           ind = vec2ind(scores)';
           
           scores = scores';
           labels = categorical(ind, [2, 1], cellstr(obj.classNames));
        end
        function [acc, predicted] = score(obj, inputs, targets)
            % Evaluate network performance on validation set by computing
            % classification accuracy.
            
            % make predictions
            predicted = obj.predict(inputs);
            acc = sum(targets == predicted)/numel(targets);
        end
    end
    
    methods(Static)
        function loss = wrapFitNet(inputs, targets, cv,...
                hiddenLayerSize, lr, momentum, trainFcn)
            % Objective Function that will be used in the bayesian
            % Optimization procedure for Hyper-Parameter tuning. It builds
            % a Neural Network and evaluates its performance on a Holdout
            % set and returns the classification loss.
            
            % Build Network Architecture
            
            
            % Build Network
            net = patternnet(hiddenLayerSize, char(trainFcn)); 
            % Specify number of epochs
            net.trainParam.epochs = 50;
            % Early stopping after 6 consecutive increases of Validation Performance
            net.trainParam.max_fail = 6;
            net.trainParam.lr = lr; % Update Learning Rate
            net.trainParam.mc = momentum; % Update Learning Rate
           
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

