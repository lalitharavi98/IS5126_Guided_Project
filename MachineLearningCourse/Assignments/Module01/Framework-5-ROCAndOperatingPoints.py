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

import MachineLearningCourse.MLUtilities.Learners.LogisticRegression as LogisticRegression
import MachineLearningCourse.MLUtilities.Evaluations.EvaluateBinaryClassification as EvaluateBinaryClassification
import MachineLearningCourse.Assignments.Module01.SMSSpamFeaturize as SMSSpamFeaturize
import MachineLearningCourse.MLUtilities.Visualizations.Charting as Charting

# A helper function for calculating FN rate and FP rate across a range of thresholds
def TabulateModelPerformanceForROC(model, xValidate, yValidate):
   pointsToEvaluate = 100
   thresholds = [ x / float(pointsToEvaluate) for x in range(pointsToEvaluate + 1)]
   FPRs = []
   FNRs = []

   try:
      for threshold in thresholds:
         FPRs.append(EvaluateBinaryClassification.FalsePositiveRate(yValidate, model.predict(xValidate, classificationThreshold=threshold)))
         FNRs.append(EvaluateBinaryClassification.FalseNegativeRate(yValidate, model.predict(xValidate, classificationThreshold=threshold)))
   except NotImplementedError:
      raise UserWarning("The 'model' parameter must have a 'predict' method that supports using a 'classificationThreshold' parameter with range [ 0 - 1.0 ] to create classifications.")

   return (FPRs, FNRs, thresholds)

def FindIndexOfClosestPointTo(target, values):
    (closestIndex, closestValue) = min(enumerate(values), key=lambda x: abs(target - x[1]))
    return closestIndex

# Hyperparameters to use for the run
stepSize = 1.0
convergence = 0.001

# Set up to hold information for creating ROC curves
seriesFPRs = []
seriesFNRs = []
seriesLabels = []

#### Learn a model with 25 frequent features
featurizer = SMSSpamFeaturize.SMSSpamFeaturize(useHandCraftedFeatures=False)
featurizer.CreateVocabulary(xTrainRaw, yTrain, numFrequentWords = 25)

xTrain      = featurizer.Featurize(xTrainRaw)
xValidate   = featurizer.Featurize(xValidateRaw)
xTest       = featurizer.Featurize(xTestRaw)

model = LogisticRegression.LogisticRegression()
model.fit(xTrain,yTrain,convergence=convergence, stepSize=stepSize)

(modelFPRs, modelFNRs, thresholds) = TabulateModelPerformanceForROC(model, xValidate, yValidate)
seriesFPRs.append(modelFPRs)
seriesFNRs.append(modelFNRs)
seriesLabels.append('25 Frequent')

# gather info for the answers
frequentFNAt50FP = modelFNRs[FindIndexOfClosestPointTo(.5, modelFPRs)]
frequentFNAt10FP = modelFNRs[FindIndexOfClosestPointTo(.1, modelFPRs)]
frequentFPAt40FN = modelFPRs[FindIndexOfClosestPointTo(.4, modelFNRs)]
frequentThresholdFor10FP = thresholds[FindIndexOfClosestPointTo(.1, modelFPRs)]

#### Learn a model with 25 features by mutual information
featurizer = SMSSpamFeaturize.SMSSpamFeaturize(useHandCraftedFeatures=False)
featurizer.CreateVocabulary(xTrainRaw, yTrain, numMutualInformationWords = 25)

xTrain      = featurizer.Featurize(xTrainRaw)
xValidate   = featurizer.Featurize(xValidateRaw)
xTest       = featurizer.Featurize(xTestRaw)

model = LogisticRegression.LogisticRegression()
model.fit(xTrain,yTrain,convergence=convergence, stepSize=stepSize)

(modelFPRs, modelFNRs, thresholds) = TabulateModelPerformanceForROC(model, xValidate, yValidate)
seriesFPRs.append(modelFPRs)
seriesFNRs.append(modelFNRs)
seriesLabels.append('25 Mutual Information')

# gather info for the answers
mutualFNAt50FP = modelFNRs[FindIndexOfClosestPointTo(.5, modelFPRs)]
mutualFNAt10FP = modelFNRs[FindIndexOfClosestPointTo(.1, modelFPRs)]
mutualFPAt40FN = modelFPRs[FindIndexOfClosestPointTo(.4, modelFNRs)]
mutualThresholdFor10FP = thresholds[FindIndexOfClosestPointTo(.1, modelFPRs)]


#### Learn a model with 25 features by mutual information
featurizer = SMSSpamFeaturize.SMSSpamFeaturize(useHandCraftedFeatures=False)
featurizer.CreateVocabulary(xTrainRaw, yTrain, numMutualInformationWords = 50)

xTrain      = featurizer.Featurize(xTrainRaw)
xValidate   = featurizer.Featurize(xValidateRaw)
xTest       = featurizer.Featurize(xTestRaw)

model = LogisticRegression.LogisticRegression()
model.fit(xTrain,yTrain,convergence=convergence, stepSize=stepSize)

(modelFPRs, modelFNRs, thresholds) = TabulateModelPerformanceForROC(model, xValidate, yValidate)
seriesFPRs.append(modelFPRs)
seriesFNRs.append(modelFNRs)
seriesLabels.append('50 Mutual Information')

print()
print("Answers:")
print()

print("%s is better at a 50%% FP rate (f%.2f vs m%.2f)" % ("25 Frequent" if frequentFNAt50FP < mutualFNAt50FP else "25 Mutual", frequentFNAt50FP, mutualFNAt50FP))
print("%s is better at a 10%% FP rate (f%.2f vs m%.2f)" % ("25 Frequent" if frequentFNAt10FP < mutualFNAt10FP else "25 Mutual", frequentFNAt10FP, mutualFNAt10FP))
print("%s is better at a 40%% FN rate (f%.2f vs m%.2f)" % ("25 Frequent" if frequentFPAt40FN < mutualFPAt40FN else "25 Mutual", frequentFPAt40FN, mutualFPAt40FN))
print("Threshold to achieve 10%% FPR with 25 Frequent model is %.2f" % frequentThresholdFor10FP)

print("Hyperparameters used to achieve a model better at both thresholds was: numFrequentWords = 0 numMutualInformationWords = 50")


Charting.PlotROCs(seriesFPRs, seriesFNRs, seriesLabels, useLines=True, chartTitle="ROC Comparison", xAxisTitle="False Negative Rate", yAxisTitle="False Positive Rate", outputDirectory=kOutputDirectory, fileName="Plot-SMSSpamROCs")

