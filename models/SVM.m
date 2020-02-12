classdef SVM
    %SVM Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        X
        y
        holdoutSize = 5
    end
    
    methods
        function obj = SVM(X,y)
            %SVM Construct an instance of this class
            %   Detailed explanation goes here
            obj.X = X;
            obj.y = y;
        end
        
        function model = train(obj)
            % Train and optimize SVM classifier using the guassian kernel
            % function. It is good practice to standardize the data.
            % For reproducibility
            rng default;
            model = fitcsvm(obj.X, obj.y);
        end
        
        function model = optimize(obj)
            % Set up a partition for cross-validation. This step fixes the
            % train and test sets that the optimization uses at each step.
            [rows, ~] = size(obj.y);
            
            cv = cvpartition(rows,'KFold', obj.holdoutSize);
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
            model = fitrsvm(obj.X, obj.y,...
                'OptimizeHyperparameters','auto',...
                'HyperparameterOptimizationOptions',opts);
            % Compare the generalization error of the models. In this case,
            % the generalization error is the out-of-sample mean-squared
            % error.
%             rmse = sqrt(kfoldLoss(model));
%             fprintf("Root mean squared for SVM is: %.2f%\n", rmse);
        end
    end
end

