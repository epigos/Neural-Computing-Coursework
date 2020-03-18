classdef SVM
    %SVM Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        X
        y
        classNames
        model
        ObjectiveMinimumTrace
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
            p.addParameter('BoxConstraint', 23);
            p.addParameter('KernelFunction', 'gaussian');
            parse(p, varargin{:});
            % For reproducibility
            rng default;
            % create svm template with optimized values
            t = templateSVM('BoxConstraint', p.Results.BoxConstraint,...
                'KernelFunction', p.Results.KernelFunction);
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
            % create 5-fold cross validation
            cv = cvpartition(size(obj.y, 1), 'Holdout', 0.2);
            % set options to use Bayesian optimization. Use the same
            % cross-validation partition cv in all optimizations. For
            % reproducibility, we use the 'expected-improvement-plus'
            % acquisition function.
            opts = struct('Optimizer','bayesopt','ShowPlots',true,...
                'CVPartition',cv, ...
                'MaxObjectiveEvaluations',p.Results.MaxObjectiveEvaluations,...
                'AcquisitionFunctionName', 'expected-improvement-plus');
            try
                % Start a parallel pool
                poolobj = gcp;
                % 'UseParallel' as true to run Bayesian optimization in
                % parallel to speed up the process. Requires Parallel
                % Computing Toolbox.
                opts.UseParallel = true;
            catch
                % Disable parralel pool if functionality is not available
                opts.UseParallel = false;
            end
            % For reproducibility
            rng default;
            % Train and optimize SVM classifier using the guassian kernel
            % function and the hyper-parameter options. Optimize all
            % eligible parameters (BoxConstraint, KernelScale,
            % KernelFunction, PolynomialOrder, Standardize).
            vars = {'BoxConstraint', 'KernelFunction'};
            obj.model = fitcecoc(obj.X, obj.y,...
                'ClassNames', obj.classNames,...
                'OptimizeHyperparameters', vars,...
                'HyperparameterOptimizationOptions',opts);
            % retrieve hyper-parameter search results
            results = obj.model.HyperparameterOptimizationResults; 
            obj.ObjectiveMinimumTrace = results;
            % save results
            save('results/Bayesopts/svm.mat', 'results');
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
end

