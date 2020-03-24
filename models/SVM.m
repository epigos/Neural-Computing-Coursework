classdef SVM
    %SVM Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        X
        y
        classNames
        model
        OptimizationResults
    end
    
    methods
        function obj = SVM(X,y, classNames)
            %SVM Construct an instance of this class
            %   Detailed explanation goes here
            obj.X = X;
            obj.y = y;
            obj.classNames = classNames;
        end
        
        function obj = fit(obj, varargin)
            % Train and optimize SVM classifier using the guassian kernel
            % function. It is good practice to standardize the data.
            % handle input variables
            p = inputParser;
            p.addParameter('BoxConstraint', 8);
            p.addParameter('KernelScale', 0.64935);
            p.addParameter('KernelFunction', 'rbf');
            parse(p, varargin{:});
            % For reproducibility
            rng default;
            % create svm template with optimized values
            t = templateSVM('BoxConstraint', p.Results.BoxConstraint,...
                'KernelFunction', char(p.Results.KernelFunction),...
                'KernelScale', p.Results.KernelScale);
            % train svm model
            obj.model = fitcecoc(obj.X, obj.y,...
                'ClassNames', obj.classNames,...
                'Learners', t);
        end
        
        function obj = optimize(obj, varargin)
            p = inputParser;
            p.addParameter('MaxObjectiveEvaluations', 200);
            parse(p, varargin{:});
            % Set up a partition for cross-validation. This step fixes the
            % train and test sets that the optimization uses at each step.
            % create holdout cross validation
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
            % function and the hyper-parameter options. Optimize all
            % eligible parameters (BoxConstraint, KernelScale,
            % KernelFunction, PolynomialOrder, Standardize).
            % Define hyperparameters to optimize
            vars = [optimizableVariable('box',[1, 20],'Type','integer');
                optimizableVariable('sigma',[1e-1,1e1],'Transform','log');
                optimizableVariable('kernel', {'rbf', 'linear', 'polynomial'}, 'Type', 'categorical')];
            % initialize objective function for the network
            minfn = @(v)SVM.wrapFitSVM(obj.X,obj.y,obj.classNames,cv,...
                v.box, v.sigma, v.kernel);
            
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
            % save results
            Utils.bayesoptResultsToCSV(results, 'SVM');
            
            obj = obj.fit('BoxConstraint', T.box,...
                'KernelFunction', T.kernel,...
                'KernelScale',T.sigma);
        end
        
        function obj = fitDefault(obj)
            % Train and SVM classifier with default parameters
            % For reproducibility
            rng default;
            % create svm template with optimized values
            t = templateSVM();
            % train svm model
            obj.model = fitcecoc(obj.X, obj.y,...
                'ClassNames', obj.classNames,...
                'Learners', t);
        end
        
        function [outputs, scores] = predict(obj, inputs)
           [outputs, scores] = predict(obj.model, inputs); 
        end
        
        function [acc, predicted]  = score(obj, inputs, targets)
            % Evaluate network performance on validation set by computing
            % rmse.
            predicted = predict(obj.model, inputs);
            acc = sum(targets == predicted)/numel(targets);
        end
    end
    methods(Static)
        function loss = wrapFitSVM(inputs, targets, classNames, cv,...
                box, sigma, kernel)
            % Objective Function that will be used in the bayesian
            % Optimization procedure for Hyper-Parameter tuning. It builds
            % an SVM and evaluates its performance on a Holdout
            % set and returns the classification loss.
            X_train = inputs(cv.training(), :);
            y_train = targets(cv.training(), :);
            X_val = inputs(cv.test(), :);
            y_val = targets(cv.test(), :);
            % Build onevsone SVM
            % create svm template with optimized values
            t = templateSVM('BoxConstraint', box,...
                'KernelFunction', char(kernel),...
                'KernelScale', sigma);
            % train svm model
            svm = fitcecoc(X_train, y_train,...
                'ClassNames', classNames,...
                'Learners', t);
            
            outputs = predict(svm, X_val);
            loss = sum(y_val ~= outputs)/numel(y_val);
        end
    end
end

