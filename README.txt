
################################################################
			     README
################################################################



# Introduction
##############

The Readme file contains instructions and comments on the deliverables of the course work developed by Philip Adzanoukpe. Before moving on, please take a moment to read it.

# Matlab Version
################

Matlab 9.7.0.1296695 (R2019b) Update 4 was used to develop all the work.

# Project Structure
########################

.
|-- data                                        # Data directory
|       |-- Frogs_MFCCs.csv                     # Data file in CSV format
|       |-- test.mat                            # contains test data to test the models
|       |-- Readme.txt                          # Data source description
|-- libs                                        # Contains all functions used in the project
|       |-- DataAnalysis.m                      # Function to run exploratory data analysis
|       |-- DecisionBoundary.m                  # Function to run decision boundaries for both models
|       |-- EnsembleLearning.m                  # Function to run Ensemble experiment
|       |-- HyperParameterTuning.m              # Function to run hyper-parameter tuning
|       |-- LearningCurve.m                     # Function to run learning curve experiments
|       |-- PerformanceComparison.m             # Function to compare model performance
|       |-- PreProcessing.m                     # Function to preprocess the data
|       |-- TimeComplexity.m                    # Function to run time complexity experiments
|       |-- Utils.m                             # Contains resuable functions
|-- models                                      
|       |-- MLP.m                               # A class definition of MLP used throughout the experiments
|       |-- SVM.m                               # A class definition of SVM used throughout the experiments
|       |-- final_models.mat                    # Matlab file containing final models which can be loaded and used for testing
|-- results
|       |-- Bayesopts_MLP.csv                   # CSV file containing full Hyper-parameter results of MLP
|       |-- Bayesopts_SVM.csv                   # CSV file containing full Hyper-parameter results of SVM
|-- charts
|       |...                                    # Contains generated charts from the experiments
|-- README.txt
|-- main.m                                      # Main entry point to run selected scripts.
|-- FinalModels.m                               # Scripts to execute test run on models and test data.

# Scripts
#########

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
