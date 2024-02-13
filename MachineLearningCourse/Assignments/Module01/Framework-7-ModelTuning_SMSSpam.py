#This preamble tells the Python interpreter to look in the folder containing
#the MachineLearningCourse dir for the relevant Python files.
import sys,os
curDir = os.path.dirname(os.path.abspath(__file__))
projDir = os.path.join(curDir,"..","..","..")
sys.path.append(projDir) #look in the directory containing MachineLearningCourse/
sys.path.append(curDir)  #look in the directory of this file too, i.e., Module01/

#specify the directory to store your visualization files
kOutputDirectory = "/Users/dhanushlalitha/Documents/GIT/IS/IS5126_Guided_Project/MachineLearningCourse/Plots//"  #use this for Mac or Linux
#kOutputDirectory = "C:\\temp\\visualize" #use this for Windows

import MachineLearningCourse.MLProjectSupport.SMSSpam.SMSSpamDataset as SMSSpamDataset

(xRaw, yRaw) = SMSSpamDataset.LoadRawData()

import MachineLearningCourse.MLUtilities.Data.Sample as Sample
(xTrainRaw, yTrain, xValidateRaw, yValidate, xTestRaw, yTest) = Sample.TrainValidateTestSplit(xRaw, yRaw, percentValidate=.1, percentTest=.1)

import MachineLearningCourse.MLUtilities.Learners.LogisticRegression as LogisticRegression
import MachineLearningCourse.MLUtilities.Evaluations.EvaluateBinaryClassification as EvaluateBinaryClassification
import MachineLearningCourse.MLUtilities.Evaluations.ErrorBounds as ErrorBounds
import MachineLearningCourse.Assignments.Module01.SMSSpamFeaturize as SMSSpamFeaturize
import MachineLearningCourse.MLUtilities.Data.CrossValidation as CrossValidation

import time

# A helper function for calculating FN rate and FP rate across a range of thresholds
def TabulateModelPerformanceForROC(model, xValidate, yValidate):
   pointsToEvaluate = 100
   thresholds = [ x / float(pointsToEvaluate) for x in range(pointsToEvaluate + 1)]
   FPRs = []
   FNRs = []
   yPredicted = model.predictProbabilities(xValidate)

   try:
      for threshold in thresholds:
        yHats = [ 1 if pred > threshold else 0 for pred in yPredicted ]
        FPRs.append(EvaluateBinaryClassification.FalsePositiveRate(yValidate, yHats))
        FNRs.append(EvaluateBinaryClassification.FalseNegativeRate(yValidate, yHats))
   except NotImplementedError:
      raise UserWarning("The 'model' parameter must have a 'predict' method that supports using a 'classificationThreshold' parameter with range [ 0 - 1.0 ] to create classifications.")

   return (FPRs, FNRs, thresholds)

import MachineLearningCourse.MLUtilities.Visualizations.Charting as Charting
## This function will help you plot with error bars. Use it just like PlotSeries, but with parallel arrays of error bar sizes in the second variable
#     note that the error bar size is drawn above and below the series value. So if the series value is .8 and the confidence interval is .78 - .82, then the value to use for the error bar is .02

# Charting.PlotSeriesWithErrorBars([series1, series2], [errorBarsForSeries1, errorBarsForSeries2], ["Series1", "Series2"], xValues, chartTitle="<>", xAxisTitle="<>", yAxisTitle="<>", yBotLimit=0.8, outputDirectory=kOutputDirectory, fileName="<name>")


## This helper function should execute a single run and save the results on 'runSpecification' (which could be a dictionary for convienience)
#    for later tabulation and charting...
def ExecuteEvaluationRun(runSpecification, xTrainRaw, yTrain, numberOfFolds = 5):
    startTime = time.time()

    featurizer = SMSSpamFeaturize.SMSSpamFeaturize()
    featurizer.CreateVocabulary(xTrainRaw, yTrain, numFrequentWords=runSpecification['numFrequentWords'],
                                numMutualInformationWords=runSpecification['numMutualInformationWords'])

    xTrain = featurizer.Featurize(xTrainRaw)
    xValidate = featurizer.Featurize(xValidateRaw)

    # K-fold cross validation
    if numberOfFolds > 1:
        accuracy_values = []
        for foldId in range(numberOfFolds):
            xTrain_fold, yTrain_fold, xEvaluate_fold, yEvaluate_fold = CrossValidation.CrossValidation(xTrain,
                                                                                                       yTrain,
                                                                                                       numberOfFolds,
                                                                                                       foldId)
            # # HERE upgrade this to use cross validation
            # featurizer = SMSSpamFeaturize.SMSSpamFeaturize()
            # featurizer.CreateVocabulary(xTrain_fold, yTrain_fold, numFrequentWords=runSpecification['numFrequentWords'],
            #                             numMutualInformationWords=runSpecification['numMutualInformationWords'])
            #
            # xTrain_fold = featurizer.Featurize(xTrain_fold)
            # xEvaluate_fold = featurizer.Featurize(xEvaluate_fold)

            model = LogisticRegression.LogisticRegression()
            model.fit(xTrain_fold, yTrain_fold, convergence=runSpecification['convergence'], stepSize=runSpecification['stepSize'],
                      verbose=True)

            validationSetAccuracy = EvaluateBinaryClassification.Accuracy(yEvaluate_fold, model.predict(xEvaluate_fold))
            accuracy_values.append(validationSetAccuracy)

        mean = sum(accuracy_values) / numberOfFolds
        runSpecification['accuracy'] = mean
        lower, upper = ErrorBounds.GetAccuracyBounds(runSpecification['accuracy'], len(yEvaluate_fold), 0.5)
        runSpecification['accuracyErrorBound'] = upper - mean


    # Train on the entire training set and evaluate on the validation set
    elif numberOfFolds == 1:
        featurizer = SMSSpamFeaturize.SMSSpamFeaturize()
        featurizer.CreateVocabulary(xTrainRaw, yTrain, numFrequentWords=runSpecification['numFrequentWords'],
                                    numMutualInformationWords=runSpecification['numMutualInformationWords'])
        xTrain = featurizer.Featurize(xTrainRaw)
        xValidate = featurizer.Featurize(xValidateRaw)
        model = LogisticRegression.LogisticRegression()
        model.fit(xTrain, yTrain,
                  convergence=runSpecification['convergence'],
                  stepSize=runSpecification['stepSize'], verbose=True)
        validationSetAccuracy = EvaluateBinaryClassification.Accuracy(yValidate, model.predict(xValidate))
        runSpecification['accuracy'] = validationSetAccuracy
        confidence = 0.5
        lower, upper = ErrorBounds.GetAccuracyBounds(runSpecification['accuracy'], len(yValidate), confidence)
        runSpecification['accuracyErrorBound'] = upper - validationSetAccuracy
    endTime = time.time()
    runSpecification['runtime'] = endTime - startTime
    return runSpecification

# Part 1: Defining the hyperparameter values to sweep  #larger ones then shift to smaller ones
step_values = [1.75, 1.5, 1.25, 1.0, 0.75, 0.5]
convergence_values = [0.5, 0.1, 0.05, 0.01, 0.005, 0.001, 0.0001]
numMutualInformationWords_values = [300, 150, 100, 50, 20, 0]
numFrequentWords_values = [300, 150, 100, 50, 20, 0]


numberOfFolds = 2
hyperparams = {'numFrequentWords': numFrequentWords_values, 'numMutualInformationWords': numMutualInformationWords_values, 'stepSize': step_values, 'convergence': convergence_values}

def parameterSweepRunSpecifications(hyper_param, initialRunSpecification):
    hyper_param_key, hyper_param_values = hyper_param
    print(hyper_param_key, hyper_param_values)
    evaluationRunSpecifications = []
    if hyper_param_key == 'stepSize':
        for hyper_param_value in hyper_param_values:
            runSpecification = {}
            runSpecification['optimizing'] = hyper_param_key
            runSpecification['stepSize'] = hyper_param_value
            runSpecification['convergence'] = initialRunSpecification['convergence']
            runSpecification['numFrequentWords'] = initialRunSpecification['numFrequentWords']
            runSpecification['numMutualInformationWords'] = initialRunSpecification['numMutualInformationWords']
            evaluationRunSpecifications.append(runSpecification)
    elif hyper_param_key == 'convergence':
        for hyper_param_value in hyper_param_values:
            runSpecification = {}
            runSpecification['optimizing'] = hyper_param_key
            runSpecification['convergence'] = hyper_param_value
            runSpecification['numFrequentWords'] = initialRunSpecification['numFrequentWords']
            runSpecification['numMutualInformationWords'] = initialRunSpecification['numMutualInformationWords']
            runSpecification['stepSize'] = initialRunSpecification['stepSize']
            evaluationRunSpecifications.append(runSpecification)
    elif hyper_param_key == 'numFrequentWords':
        for hyper_param_value in hyper_param_values:
            runSpecification = {}
            runSpecification['optimizing'] = hyper_param_key
            runSpecification['numFrequentWords'] = hyper_param_value
            runSpecification['stepSize'] = initialRunSpecification['stepSize']
            runSpecification['convergence'] = initialRunSpecification['convergence']
            runSpecification['numMutualInformationWords'] = initialRunSpecification['numMutualInformationWords']
            evaluationRunSpecifications.append(runSpecification)
    elif hyper_param_key == 'numMutualInformationWords':
        for hyper_param_value in hyper_param_values:
            runSpecification = {}
            runSpecification['optimizing'] = hyper_param_key
            runSpecification['numMutualInformationWords'] = hyper_param_value
            runSpecification['stepSize'] = initialRunSpecification['stepSize']
            runSpecification['convergence'] = initialRunSpecification['convergence']
            runSpecification['numFrequentWords'] = initialRunSpecification['numFrequentWords']
            evaluationRunSpecifications.append(runSpecification)
    return evaluationRunSpecifications

def findOptimalSpecs(specs):
    # bestUpperBoundValue = max((spec['accuracy'] + spec['accuracyErrorBound']) for spec in specs)  #highest upper bound value
    # tiedUpperBounds = [spec for spec in specs if (spec['accuracy'] + spec['accuracyErrorBound']) == bestUpperBoundValue] # tied values
    # best_runtime = float('inf')
    # best_spec ={}
    # for spec in tiedUpperBounds:
    #     runtime = spec['runtime']  # Unpack accuracy and runtime
    #     if runtime < best_runtime:
    #         best_runtime = runtime
    #         best_spec = spec  # Track the best model
    #
    #     # Return the best model
    # return {
    #     'stepSize': best_spec['stepSize'],
    #     'convergence': best_spec['convergence'],
    #     'numFrequentWords': best_spec['numFrequentWords'],
    #     'numMutualInformationWords': best_spec['numMutualInformationWords']
    # }

    specs = sorted(specs, key=lambda x: x['accuracy'])

    best_specs = specs[0]
    bestAccUpperBound = best_specs['accuracy'] + best_specs['accuracyErrorBound']
    for i in range(1, len(specs)):
        newAccLowerBound = specs[i]['accuracy'] - specs[i]['accuracyErrorBound']
        # Part 3a: the value you pick is tied with the highest accuracy value according to a 75% 1-sided bound
        if newAccLowerBound > bestAccUpperBound:
            best_specs = specs[i]

        # Part 3b: the value you pick has the lowest runtime among these ‘tied’ values.
        elif specs[i]['accuracy'] > best_specs['accuracy'] and specs[i]['runtime'] < best_specs['runtime']:
            best_specs = specs[i]

    return {
        'stepSize': best_specs['stepSize'],
        'convergence': best_specs['convergence'],
        'numFrequentWords':  best_specs['numFrequentWords'],
        'numMutualInformationWords': best_specs['numMutualInformationWords']
    }






initialHyperParams = optimalHyperParams = {'numFrequentWords': 0, 'numMutualInformationWords': 20, 'stepSize': 1.0, 'convergence': 0.005}

#######################################################################################################################
# Part 2: Optimal hyperparameters ( Independent of other hyperparameters not updated after each sweep)
for sweep_number, hyper_param in enumerate(hyperparams.items(), start=1): # 1 sweep for each hyperparameter
    hyper_param_key, hyper_param_values = hyper_param
    print("Optimising hyperparameter: --->", hyper_param_key)
    evaluationRunSpecifications = parameterSweepRunSpecifications(hyper_param, optimalHyperParams)
    evaluations = [ExecuteEvaluationRun(runSpec, xTrainRaw, yTrain, 2) for runSpec in evaluationRunSpecifications]

    for evaluation in evaluations:
        print(evaluation)

    # Extracting data for plotting
    accuracy_values = [evaluation['accuracy'] for evaluation in evaluations]
    accuracy_errors = [evaluation['accuracyErrorBound'] for evaluation in evaluations]
    runtime_values = [evaluation['runtime'] for evaluation in evaluations]

    #Plotting the accuracy
    Charting.PlotSeriesWithErrorBars([accuracy_values], [accuracy_errors], [hyper_param_key],
                                     hyperparams[hyper_param_key], chartTitle="Accuracy vs " + hyper_param_key,
                                     xAxisTitle=hyper_param_key, yAxisTitle="Accuracy", yBotLimit=0.8,
                                     outputDirectory=kOutputDirectory, fileName="Part_2_Accuracy_vs_" + hyper_param_key)

    # Plotting the runtime
    Charting.PlotSeries([runtime_values], [hyper_param_key], hyperparams[hyper_param_key],
                        chartTitle="Runtime vs " + hyper_param_key, xAxisTitle=hyper_param_key,
                        yAxisTitle="Runtime (seconds)", outputDirectory=kOutputDirectory,
                        fileName="Part_2_Runtime_vs_" + hyper_param_key)

#######################################################################################################################
# Part 3: Optimal hyperparameters (hyperparameters are updated after each sweep) # Sweep 1: init spec

init_specs = ExecuteEvaluationRun(optimalHyperParams, xTrainRaw, yTrain, 1)
validation_accuracy_across_sweeps = [init_specs['accuracy']] # initial accuracy
validation_accuracy_errors_across_sweeps = [init_specs['accuracyErrorBound']] # initial accuracy error
runtime_across_sweeps = [init_specs['runtime']] # initial spec runtime
for sweep_number, hyper_param in enumerate(hyperparams.items(), start=2):
    hyper_param_key, hyper_param_values = hyper_param
    print("Optimising hyperparameter: --->", hyper_param_key)
    evaluationRunSpecifications = parameterSweepRunSpecifications(hyper_param, optimalHyperParams)
    evaluations = [ExecuteEvaluationRun(runSpec, xTrainRaw, yTrain, 2) for runSpec in evaluationRunSpecifications] #hyper paramter search with 2 fold cross validation

    for evaluation in evaluations:
        print(evaluation)

    # Extracting data for plotting
    accuracy_values = [evaluation['accuracy'] for evaluation in evaluations]
    accuracy_errors = [evaluation['accuracyErrorBound'] for evaluation in evaluations]
    runtime_values = [evaluation['runtime'] for evaluation in evaluations]

    # Plotting the accuracy
    Charting.PlotSeriesWithErrorBars([accuracy_values], [accuracy_errors], [hyper_param_key],
                                     hyperparams[hyper_param_key], chartTitle="Accuracy vs " + hyper_param_key,
                                     xAxisTitle=hyper_param_key, yAxisTitle="Accuracy", yBotLimit=0.8,
                                     outputDirectory=kOutputDirectory,
                                     fileName="Update_Part_2_Accuracy_vs_" + hyper_param_key)

    # Plotting the runtime
    Charting.PlotSeries([runtime_values], [hyper_param_key], hyperparams[hyper_param_key],
                        chartTitle="Runtime vs " + hyper_param_key, xAxisTitle=hyper_param_key,
                        yAxisTitle="Runtime (seconds)", outputDirectory=kOutputDirectory,
                        fileName="Update_Part_2_Runtime_vs_" + hyper_param_key)

    optimalHyperParams = findOptimalSpecs(evaluations)

    # Print the optimal specs for current sweep
    print("#"*50)
    print(f"Optimal hyper parameters after sweep {sweep_number}: ", optimalHyperParams)

    # Part 4a: Evaluating the accuracy on the validation set after optimising each hyperparameter
    result = ExecuteEvaluationRun(optimalHyperParams, xTrainRaw, yTrain, 1)
    validation_accuracy_across_sweeps.append(result['accuracy'])
    validation_accuracy_errors_across_sweeps.append(result['accuracyErrorBound'])
    sweep_list = [i for i in range(1, sweep_number+1)] # 1 to sweep_number+1(initial accuracy)
    # Part 4b: Plotting the validation accuracy across the sweeps
    Charting.PlotSeriesWithErrorBars([validation_accuracy_across_sweeps], [validation_accuracy_errors_across_sweeps],
                                     ["Accuracy"],
                                     sweep_list, chartTitle=f"Accuracy across Sweep: {sweep_number}",
                                     xAxisTitle="Sweep", yAxisTitle="Accuracy", yBotLimit=0.8,
                                     outputDirectory=kOutputDirectory, fileName=f"Sweep_{sweep_number}_Accuracy")


# Best hyperarameters after all sweeps
print(f"Best hyper parameters after all sweeps: ", optimalHyperParams)


# Part 6: ROC of initial VS Best model
seriesFPRs = []
seriesFNRs = []
seriesLabels = []


featurizer = SMSSpamFeaturize.SMSSpamFeaturize()
featurizer.CreateVocabulary(xTrainRaw, yTrain, numFrequentWords = initialHyperParams['numFrequentWords'], numMutualInformationWords = initialHyperParams['numMutualInformationWords'])

xTrain = featurizer.Featurize(xTrainRaw)
xTest = featurizer.Featurize(xTestRaw)

init_model = LogisticRegression.LogisticRegression()
init_model.fit(xTrain, yTrain, convergence=initialHyperParams['convergence'], stepSize=initialHyperParams['stepSize'], verbose=False)
EvaluateBinaryClassification.ExecuteAll(yTest, init_model.predict(xTest))
(init_modelFPRs, init_modelFNRs, init_thresholds) = TabulateModelPerformanceForROC(init_model, xTest, yTest)
seriesFPRs.append(init_modelFPRs)
seriesFNRs.append(init_modelFNRs)
seriesLabels.append('initial hyper parameters')

featurizer = SMSSpamFeaturize.SMSSpamFeaturize()
featurizer.CreateVocabulary(xTrainRaw, yTrain, numFrequentWords = optimalHyperParams['numFrequentWords'], numMutualInformationWords = optimalHyperParams['numMutualInformationWords'])

xTrain = featurizer.Featurize(xTrainRaw)
xTest = featurizer.Featurize(xTestRaw)

best_model = LogisticRegression.LogisticRegression()
best_model.fit(xTrain, yTrain, convergence=optimalHyperParams['convergence'], stepSize=optimalHyperParams['stepSize'], verbose=False)
EvaluateBinaryClassification.ExecuteAll(yTest, best_model.predict(xTest))

(best_modelFPRs, best_modelFNRs, best_thresholds) = TabulateModelPerformanceForROC(best_model, xTest, yTest)
seriesFPRs.append(best_modelFPRs)
seriesFNRs.append(best_modelFNRs)
seriesLabels.append('optimal hyper parameters')

Charting.PlotROCs(seriesFPRs, seriesFNRs, seriesLabels, useLines=True, chartTitle="ROC Comparison", xAxisTitle="False Negative Rate", yAxisTitle="False Positive Rate", outputDirectory=kOutputDirectory, fileName="Best_VS_Initial_ROC")

