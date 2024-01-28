#This preamble tells the Python interpreter to look in the folder containing
#the MachineLearningCourse dir for the relevant Python files.
import sys,os
curDir = os.path.dirname(os.path.abspath(__file__))
projDir = os.path.join(curDir,"..","..","..")
sys.path.append(projDir) #look in the directory containing MachineLearningCourse/
sys.path.append(curDir)  #look in the directory of this file too, i.e., Module01/

import MachineLearningCourse.MLProjectSupport.SMSSpam.SMSSpamDataset as SMSSpamDataset

(xRaw, yRaw) = SMSSpamDataset.LoadRawData()

import MachineLearningCourse.MLUtilities.Data.Sample as Sample
import MachineLearningCourse.Assignments.Module01.SMSSpamFeaturize as SMSSpamFeaturize
import MachineLearningCourse.MLUtilities.Learners.LogisticRegression as LogisticRegression
import MachineLearningCourse.MLUtilities.Learners.MostCommonClassModel as MostCommonClassModel
import MachineLearningCourse.MLUtilities.Evaluations.EvaluateBinaryClassification as EvaluateBinaryClassification

(xTrainRaw, yTrain, xValidateRaw, yValidate, xTestRaw, yTest) = Sample.TrainValidateTestSplit(xRaw, yRaw, percentValidate=.1, percentTest=.1)

doModelEvaluation = True
if doModelEvaluation:
    ######
    ### Build a model and evaluate on validation data
    stepSize = 1.0
    convergence = 0.001

    featurizer = SMSSpamFeaturize.SMSSpamFeaturize()
    featurizer.CreateVocabulary(xTrainRaw, yTrain, numMutualInformationWords = 25)

    xTrain      = featurizer.Featurize(xTrainRaw)
    xValidate   = featurizer.Featurize(xValidateRaw)
    xTest       = featurizer.Featurize(xTestRaw)

    frequentModel = LogisticRegression.LogisticRegression()
    frequentModel.fit(xTrain, yTrain, convergence=convergence, stepSize=stepSize, verbose=True)

    ######
    ### Use equation 5.1 from Mitchell to bound the validation set error and the true error
    import MachineLearningCourse.MLUtilities.Evaluations.ErrorBounds as ErrorBounds

    print("Logistic regression with 25 features by mutual information:")
    validationSetAccuracy = EvaluateBinaryClassification.Accuracy(yValidate, frequentModel.predict(xValidate))
    print("Validation set accuracy: %.4f." % (validationSetAccuracy))
    for confidence in [.5, .8, .9, .95, .99]:
        (lowerBound, upperBound) = ErrorBounds.GetAccuracyBounds(validationSetAccuracy, len(xValidate), confidence)    
        print(" %.2f%% accuracy bound: %.4f - %.4f" % (confidence, lowerBound, upperBound))

    print()
    
    ######
    ### Evaluate the most common model
    print("Most common model:")
    mostCommonModel = MostCommonClassModel.MostCommonClassModel()
    mostCommonModel.fit(xTrainRaw, yTrain)
    
    validationSetAccuracy = EvaluateBinaryClassification.Accuracy(yValidate, mostCommonModel.predict(xValidate))
    print("Validation set accuracy: %.4f." % (validationSetAccuracy))
    for confidence in [.5, .8, .9, .95, .99]:

        (lowerBound, upperBound) = ErrorBounds.GetAccuracyBounds(validationSetAccuracy, len(xValidate), confidence)    
        print(" %.2f%% accuracy bound: %.4f - %.4f" % (confidence, lowerBound, upperBound))


doCrossValidation = True
if doCrossValidation:
    import MachineLearningCourse.MLUtilities.Data.CrossValidation as CrossValidation
    
    numCorrectLogisticRegression = 0
    numCorrectMostCommonClass = 0
    
    numberOfFolds = 5
    for foldID in range(numberOfFolds):
        print("Fold %d" % foldID)
        
        (xFoldTrainRaw, yFoldTrain, xFoldValidateRaw, yFoldValidate) = CrossValidation.CrossValidation(xTrainRaw, yTrain, numberOfFolds, foldID)

        # Learn and evalutate the logistic regression model for this fold        
        featurizer = SMSSpamFeaturize.SMSSpamFeaturize()
        featurizer.CreateVocabulary(xFoldTrainRaw, yFoldTrain, numMutualInformationWords = 25)

        xFoldTrain      = featurizer.Featurize(xFoldTrainRaw)
        xFoldValidate   = featurizer.Featurize(xFoldValidateRaw)

        stepSize = 1.0
        convergence = 0.001

        frequentModel = LogisticRegression.LogisticRegression()
        frequentModel.fit(xFoldTrain, yFoldTrain, convergence=convergence, stepSize=stepSize, verbose=True)
        
        foldValidationSetAccuracy = EvaluateBinaryClassification.Accuracy(yFoldValidate, frequentModel.predict(xFoldValidate))
        numCorrectLogisticRegression = numCorrectLogisticRegression + (foldValidationSetAccuracy * len(xFoldValidate))

        # learn and evaluate the most common class model for this fold
        mostCommonModel = MostCommonClassModel.MostCommonClassModel()
        mostCommonModel.fit(xFoldTrain, yFoldTrain)
        
        foldValidationSetAccuracy = EvaluateBinaryClassification.Accuracy(yFoldValidate, mostCommonModel.predict(xFoldValidate))
        numCorrectMostCommonClass = numCorrectMostCommonClass + (foldValidationSetAccuracy * len(xFoldValidate))

    # Now tabulate and output intervals for all the required confidence levels 
    import MachineLearningCourseInstructor.MLUtilities.Evaluations.ErrorBounds as ErrorBounds
    
    logisticRegressionAccuracy = numCorrectLogisticRegression / len(xTrainRaw)
    for confidence in [.5, .8, .9, .95, .99]:
        (lowerBound, upperBound) = ErrorBounds.GetAccuracyBounds(logisticRegressionAccuracy, len(xTrainRaw), confidence)    
        print("Logistic Regression %.2f bound: %.4f - %.4f" % (confidence, lowerBound, upperBound))

    mostCommonModelAccuracy = numCorrectMostCommonClass / len(xTrainRaw)
    for confidence in [.5, .8, .9, .95, .99]:
        (lowerBound, upperBound) = ErrorBounds.GetAccuracyBounds(mostCommonModelAccuracy, len(xTrainRaw), confidence)    
        print("Most Common Class %.2f bound: %.4f - %.4f" % (confidence, lowerBound, upperBound))

        
