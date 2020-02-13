classdef SVM
    %SVM Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        X
        y
        classNames
        model
    end
    
    methods
        function obj = SVM(X,y, classNames)
            %SVM Construct an instance of this class
            %   Detailed explanation goes here
            obj.X = X;
            obj.y = y;
            obj.classNames = classNames;
        end
        
        function obj = train(obj)
            % Train and optimize SVM classifier using the guassian kernel
            % function. It is good practice to standardize the data.
            % For reproducibility
            rng default;
            obj.model = fitcecoc(obj.X, obj.y, 'ClassNames', obj.classNames);
        end
        
        function obj = optimize(obj)
            % Set up a partition for cross-validation. This step fixes the
            % train and test sets that the optimization uses at each step.
            % create 5-fold cross validation
            cv = cvpartition(size(obj.y, 1), 'Holdout', 1/3);
            % set options to use Bayesian optimization. Use the same
            % cross-validation partition cv in all optimizations. For
            % reproducibility, we use the 'expected-improvement-plus'
            % acquisition function.
            opts = struct('Optimizer','bayesopt','ShowPlots',true,...
                'CVPartition',cv, ...
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
            obj.model = fitcecoc(obj.X, obj.y,...
                'ClassNames', obj.classNames,...
                'OptimizeHyperparameters', {'BoxConstraint','KernelScale'},...
                'HyperparameterOptimizationOptions',opts);     
        end
        
        function acc = score(obj, inputs, targets)
            % Evaluate network performance on validation set by computing
            % rmse.
            predicted = predict(obj.model, inputs);
            acc = sum(targets == predicted)/numel(targets);
        end
    end
end

