# Neural Computing Coursework - Comparing MLP and SVM

This repo contains Matlab scripts to compare the performance MLP and SVM machine learning models.

# Matlab Version

Matlab 9.7.0.1296695 (R2019b) Update 4 was used to develop all the work.

# Project Structure

.
├── FinalModels.m
├── README.txt
├── charts
│   ├── bayesopts.png
│   ├── cm.png
│   ├── decision_boundary.png
│   ├── decision_boundary3d.png
│   ├── dist.png
│   ├── ensemble.png
│   ├── learning_curve.png
│   ├── roc.png
│   └── time_complexity.png
├── data
│   ├── Frogs_MFCCs.csv
│   ├── Readme.txt
│   └── test.mat
├── libs
│   ├── DataAnalysis.m
│   ├── DecisionBoundary.m
│   ├── EnsembleLearning.m
│   ├── HyperParameterTuning.m
│   ├── LearningCurve.m
│   ├── PerformanceComparison.m
│   ├── PreProcessing.m
│   ├── TimeComplexity.m
│   └── Utils.m
├── main.m
├── models
│   ├── MLP.m
│   ├── SVM.m
│   └── final_models.mat
└── results
    ├── Bayesopts_MLP.csv
    └── Bayesopts_SVM.csv

# Scripts

In this section, I will explain how the scripts can be run.

To run the final models on the test data, run the FinalModels.m. This loads both the saved model and test data and make predictions.

To run all the experiments, run the main.m file. It serves as an entry point to run all scripts. 

Running the main.m file asks the user to select a number that matches the script to run. Type 0 to leave the programme. 

The main entry point loads the data from the data directory and passes it as an argument to the selected function.


Definition of each scripts:

	- main.m : Main interface for running all scripts from one command.

    - FinalModels.m : This scripts contains a function to run the both MLP and SVM with the hyper-parameters and prints out the output

	- DataAnalysis.m : This script output summary and other descriptive statistics about the data, in the form of figures and console prints.

    - HyperParameterTuning.m : This script performs auto optimization of the hyper-parameters using Bayesian optimization to obtain the optimal values for each models.

    - PerformanceComparison.m : This script shows the difference in predictive performance and time performance between the two algorithms. It uses a simple train-test cross-validation.

	- DecisionBoundary.m : This script visualizes the decision boundaries of Naive Bayes and Random Forest in 2D by training the models on the numerical attributes. The goal here is to show the difference between the shapes.

	- EnsembleLearning.m : This script executes Ensembles of MLP and SVM Algorithm and compare the accuracies with the base classifier. It uses the optimal setup of the hyper-parameter tuning.

    - LearningCurve.m : This script visualises the MLP and SVM Algorithm Learning Curve. It uses the optimal setup of the hyper-parameter tuning.

    - TimeComplexity.m : This script executes a Multi-Layer Perceptron and Vector Support System Complexity Analysis.

    - Utils.m : This contains a helper class with static methods which shared among all the scripts.

    - MLP.m : A class definition of MLP used throughout the experiments which contains function to train, optimize and predict for MLP model

    - SVM.m : A class definition of MLP used throughout the experiments which contains function to train, optimize and predict for SVM model
