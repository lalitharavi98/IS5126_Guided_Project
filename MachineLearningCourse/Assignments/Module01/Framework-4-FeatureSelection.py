#This preamble tells the Python interpreter to look in the folder containing
#the MachineLearningCourse dir for the relevant Python files.
import sys,os
curDir = os.path.dirname(os.path.abspath(__file__))
projDir = os.path.join(curDir,"..","..","..")
sys.path.append(projDir) #look in the directory containing MachineLearningCourse/
sys.path.append(curDir)  #look in the directory of this file too, i.e., Module01/

#specify the directory to store your visualization files
kOutputDirectory = "/users/stanleykok"  #use this for Mac or Linux
#kOutputDirectory = "C:\\temp\\visualize" #use this for Windows


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
doModeling = False #True
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
