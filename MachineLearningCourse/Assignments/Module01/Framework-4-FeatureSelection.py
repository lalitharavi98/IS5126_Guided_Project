# region START
#This preamble tells the Python interpreter to look in the folder containing
#the MachineLearningCourse dir for the relevant Python files.
import sys,os
curDir = os.path.dirname(os.path.abspath(__file__))
projDir = os.path.join(curDir,"..","..","..")
sys.path.append(projDir) #look in the directory containing MachineLearningCourse/
sys.path.append(curDir)  #look in the directory of this file too, i.e., Module01/

#specify the directory to store your visualization files
# use this for Mac or Linux
kOutputDirectory = os.path.join(curDir, "Visualizations")
#kOutputDirectory = "C:\\temp\\visualize" #use this for Windows
# endregion START

import MachineLearningCourse.MLProjectSupport.SMSSpam.SMSSpamDataset as SMSSpamDataset

(xRaw, yRaw) = SMSSpamDataset.LoadRawData()

import MachineLearningCourse.MLUtilities.Data.Sample as Sample
(xTrainRaw, yTrain, xValidateRaw, yValidate, xTestRaw, yTest) = Sample.TrainValidateTestSplit(xRaw, yRaw, percentValidate=.1, percentTest=.1)

import MachineLearningCourse.Assignments.Module01.SMSSpamFeaturize as SMSSpamFeaturize

findTop10Words = True
if findTop10Words:
    featurizer = SMSSpamFeaturize.SMSSpamFeaturize(useHandCraftedFeatures=False)

    print("Top 10 words by frequency: ", featurizer.FindMostFrequentWords(xTrainRaw, 10))
    print("Top 10 words by mutual information: ", featurizer.FindTopWordsByMutualInformation(xTrainRaw, yTrain, 10))

# set to true when your implementation of the 'FindWords' part of the assignment is working
doModeling = True
if doModeling:
    # Now get into model training
    import MachineLearningCourse.MLUtilities.Learners.LogisticRegression as LogisticRegression
    import MachineLearningCourse.MLUtilities.Evaluations.EvaluateBinaryClassification as EvaluateBinaryClassification

    # The hyperparameters to use with logistic regression for this assignment
    stepSize = 1.0
    convergence = 0.001

    # Remeber to create a new featurizer object/vocabulary for each part of the assignment
    featurizer = SMSSpamFeaturize.SMSSpamFeaturize(useHandCraftedFeatures=False)
    featurizer.CreateVocabulary(xTrainRaw, yTrain, numFrequentWords = 10)

    # Remember to reprocess the raw data whenever you change the featurizer
    xTrain      = featurizer.Featurize(xTrainRaw)
    xValidate   = featurizer.Featurize(xValidateRaw)
    xTest       = featurizer.Featurize(xTestRaw)

    ## Good luck!

    """
    Question 4b: Training the logistic regression model

    """
    # Question 4b: Traiining the logistic regression model
    print("Training the logistic regression model:")
    logisticRegressionModel = LogisticRegression.LogisticRegression()

    logisticRegressionModel.fit(
        xTrain, yTrain, stepSize=stepSize, convergence=convergence)

    print("\nLogistic regression model:")
    logisticRegressionModel.visualize()

    EvaluateBinaryClassification.ExecuteAll(
        yValidate, logisticRegressionModel.predict(xValidate, classificationThreshold=0.5))

"""
Question 4c & 4d : Parameter Sweep on Number of features(number of frequent words & number of mutual information words)

"""
# Determine whether to perform a parameter sweep on the number of features =>number of frequent words & number of mutual information words)
doParameterTuning = True
if doParameterTuning:

    """
    For Parameter sweep on number of frequent words

    """
    # Store model performance scores
    trainLosses = []
    validationLosses = []

    # Numeric values to evaluate for feature selection(Number of frequent words)
    lossXLabels = [1, 10, 20, 30, 40, 50]
    for n_feature in lossXLabels:
       
       # Import the logistic regression model
       import MachineLearningCourse.MLUtilities.Learners.LogisticRegression as LogisticRegression

       # Create a new featurizer for each feature count 
       featurizer = SMSSpamFeaturize.SMSSpamFeaturize(useHandCraftedFeatures=False)
       featurizer.CreateVocabulary(xTrainRaw, yTrain, numFrequentWords=n_feature)

        # Convert raw text data into numerical features  
       xTrain = featurizer.Featurize(xTrainRaw)
       xValidate = featurizer.Featurize(xValidateRaw)
       xTest = featurizer.Featurize(xTestRaw)
        
        
       logisticRegressionModel = LogisticRegression.LogisticRegression()
       
        # Hyper parameter values
       stepSize = 1.0
       convergence = 0.001

        # Train the model
       logisticRegressionModel.fit(
           xTrain, yTrain, stepSize = stepSize, convergence = convergence)

       # Evaluate model performance on training and validation data
       trainLosses.append(logisticRegressionModel.loss(xTrain, yTrain))
       validationLosses.append(
            logisticRegressionModel.loss(xValidate, yValidate))
       
    # Plotting the results
    import MachineLearningCourse.MLUtilities.Visualizations.Charting as Charting
    # trainLosses, validationLosses, and lossXLabels are parallel arrays with the losses you want to plot at the specified x coordinates
    Charting.PlotSeries([trainLosses, validationLosses], ['Train', 'Validate'], lossXLabels, chartTitle="No of Frequent Words vs. Logistic Regression Loss ",
                        xAxisTitle="Number of Frequent words", yAxisTitle="Avg. Loss", outputDirectory=kOutputDirectory, fileName="Parameter Sweep_Num Frequent Words Train vs Validate loss")

    """
    For Parameter sweep on the number of mutual information words(MI)

    """
    # Store model performance scores
    trainLosses_MI = []
    validationLosses_MI = []

    # Numeric values to evaluate for feature selection(Number of mutual information words(MI))
    lossXLabels_MI = [1, 10, 20, 30, 40, 50]

    for n_feature in lossXLabels_MI:

       # Import the logistic regression model
       import MachineLearningCourse.MLUtilities.Learners.LogisticRegression as LogisticRegression

       # Create a new featurizer for each feature count
       featurizer = SMSSpamFeaturize.SMSSpamFeaturize(
           useHandCraftedFeatures=False)
       featurizer.CreateVocabulary(
           xTrainRaw, yTrain, numMutualInformationWords = n_feature)

       # Convert raw text data into numerical features
       xTrain = featurizer.Featurize(xTrainRaw)
       xValidate = featurizer.Featurize(xValidateRaw)
       xTest = featurizer.Featurize(xTestRaw)

       logisticRegressionModel = LogisticRegression.LogisticRegression()

       # Hyper parameter values
       stepSize = 1.0
       convergence = 0.001

       # Train the model
       logisticRegressionModel.fit(
           xTrain, yTrain, stepSize = stepSize, convergence = convergence)

       # Evaluate model performance on training and validation data
       trainLosses_MI.append(logisticRegressionModel.loss(xTrain, yTrain))
       validationLosses_MI.append(
           logisticRegressionModel.loss(xValidate, yValidate))

    # Plotting the results
    import MachineLearningCourse.MLUtilities.Visualizations.Charting as Charting
    # trainLosses, validationLosses, and lossXLabels are parallel arrays with the losses you want to plot at the specified x coordinates
    Charting.PlotSeries([trainLosses_MI, validationLosses_MI], ['Train', 'Validate'], lossXLabels_MI, chartTitle="No of Mutual Information Words vs. Logistic Regression Loss  ",
                        xAxisTitle="Number of Mutual Information words", yAxisTitle="Avg. Loss", outputDirectory=kOutputDirectory, fileName="Parameter Sweep_Mutual Information words_ Train vs Validate loss")
