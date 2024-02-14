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
(xTrainRaw, yTrain, xValidateRaw, yValidate, xTestRaw, yTest) = Sample.TrainValidateTestSplit(xRaw, yRaw, percentValidate=.1, percentTest=.1)

# This time we aren't going to use the xRaw values - we are going to convert our xRaws into feature vectors
import MachineLearningCourse.Assignments.Module01.SMSSpamFeaturize as SMSSpamFeaturize

# Create an instance of the featurizer, and tell it to use some hand-crafted code we created to produce features.
featurizer = SMSSpamFeaturize.SMSSpamFeaturize(useHandCraftedFeatures=True)

# We'll also create a vocabulary and use the presence or absence of specific words in the feature vector.

#  In a later assignment, you'll update 'CreateVocabulary' to select the vocabulary automatically. For 
#   now, just add in a few 'spammy' (?) words by hand.
featurizer.CreateVocabulary(xTrainRaw, yTrain, supplementalVocabularyWords=['call', 'to', 'your'])

# Apply the featurerizer to the raw data sets to produce feature vectors. In this case, each message will be converted to an array
#  with one bit per feature that is 1 if the message has the feature, and 0 if the message does not have the feature.
xTrain      = featurizer.Featurize(xTrainRaw)
xValidate   = featurizer.Featurize(xValidateRaw)
xTest       = featurizer.Featurize(xTestRaw)

print("\n - Inspect the features -")
for i in range(len(xTrain[0])):
    print(featurizer.GetFeatureInfo(i))

print("\n - Inspect feature values for a few training samples -")
for i in range(5):
    print(yTrain[i], "-", xTrain[i], xTrainRaw[i])
    
# Now let's up our modeling game (as compared to predicting the most common class)
#  we'll use a heuristic (hand-tuned) linear model.

# import MachineLearningCourse.MLUtilities.Learners.LinearHeuristicModel as LinearHeuristicModel
# model = LinearHeuristicModel.LinearHeuristicModel()
# model.fit(xTrain, yTrain, -1.0, [ .75, .75, .75, .25, .25 ])

import MachineLearningCourse.MLUtilities.Learners.MostCommonClassModel as MostCommonClassModel
model = MostCommonClassModel.MostCommonClassModel()
model.fit(xTrain, yTrain)

print("\n - Inspect the weights on the heuristically-tuned model -")
model.visualize()

yValidatePredicted = model.predict(xValidate)
    
import MachineLearningCourse.MLUtilities.Evaluations.EvaluateBinaryClassification as EvaluateBinaryClassification
EvaluateBinaryClassification.ExecuteAll(yValidate, yValidatePredicted)
