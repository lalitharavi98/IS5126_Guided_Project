#This preamble tells the Python interpreter to look in the folder containing
#the MachineLearningCourse dir for the relevant Python files.
import sys,os
curDir = os.path.dirname(os.path.abspath(__file__))
projDir = os.path.join(curDir,"..","..","..")
sys.path.append(projDir) #look in the directory containing MachineLearningCourse/
sys.path.append(curDir)  #look in the directory of this file too, i.e., Module01/

#specify the directory to store your visualization files
# kOutputDirectory = "/users/stanleykok"  #use this for Mac or Linux
kOutputDirectory = "C:\\Users\\winyan\\Git Projects\\IS5126_Guided_Project\\Visual" #use this for Windows

runUnitTest = False
if runUnitTest:
    # Little synthetic dataset to help with implementation. 2 features, 8 samples.
    xTrain = [[.1, .1], [.2, .2], [.2, .1], [.1, .2], [.95, .95], [.9, .8], [.8, .9], [.7, .6]]
    yTrain = [0, 0, 0, 0, 1, 1, 1, 1]

    # create a linear model with the right number of weights initialized
    import MachineLearningCourse.MLUtilities.Learners.LogisticRegression as LogisticRegression
    model = LogisticRegression.LogisticRegression(featureCount = len(xTrain[0]))

    # To use this visualizer you need to install the PIL imaging library. Instructions are in the lecture notes.
    import MachineLearningCourse.MLUtilities.Visualizations.Visualize2D as Visualize2D

    while not model.converged:
        # do 10 iterations of training
        model.incrementalFit(xTrain, yTrain, maxSteps=10, stepSize=1.0, convergence=0.005)
        
        # then look at the models weights
        model.visualize()
        
        # then look at how training set loss is converging
        print(" fit for %d iterations, train set loss is %.4f" % (model.totalGradientDescentSteps, model.loss(xTrain, yTrain)))

        # and visualize the model's decision boundary
        visualization = Visualize2D.Visualize2D(kOutputDirectory, "{0:04}.test".format(model.totalGradientDescentSteps))
        visualization.Plot2DDataAndBinaryConcept(xTrain, yTrain, model)
        visualization.Save()
        
# Once your LogisticRegression learner seems to be working, set this flag to True and try it on the spam data
runSMSSpam = True
if runSMSSpam:
    import MachineLearningCourse.MLProjectSupport.SMSSpam.SMSSpamDataset as SMSSpamDataset

    ############################
    # Set up the data

    (xRaw, yRaw) = SMSSpamDataset.LoadRawData()
    
    import MachineLearningCourse.MLUtilities.Data.Sample as Sample
    (xTrainRaw, yTrain, xValidateRaw, yValidate, xTestRaw, yTest) = Sample.TrainValidateTestSplit(xRaw, yRaw, percentValidate=.1, percentTest=.1)

    import MachineLearningCourse.Assignments.Module01.SMSSpamFeaturize as SMSSpamFeaturize
    featurizer = SMSSpamFeaturize.SMSSpamFeaturize(useHandCraftedFeatures=True)
    featurizer.CreateVocabulary(xTrainRaw, yTrain, supplementalVocabularyWords=['call','to','your'])

    xTrain      = featurizer.Featurize(xTrainRaw)
    xValidate   = featurizer.Featurize(xValidateRaw)
    xTest       = featurizer.Featurize(xTestRaw)

    #############################
    # Learn the logistic regression model
    
    print("Learning the logistic regression model:")
    import MachineLearningCourse.MLUtilities.Learners.LogisticRegression as LogisticRegression
    import MachineLearningCourse.MLUtilities.Evaluations.EvaluateBinaryClassification as EvaluateBinaryClassification

    logisticRegressionModel = LogisticRegression.LogisticRegression()

    # logisticRegressionModel.fit(xTrain, yTrain, stepSize=1.0, convergence=0.005)
    # FOr question 3c, convergence parameter tuning
    convergence_values = [0.05, 0.01, 0.001, 0.0001, 0.00001]

    for convergence in convergence_values:
        print("\nConvergence value used:", convergence)

        # Fit the model with the current convergence value
        logisticRegressionModel.fit(xTrain, yTrain, stepSize=1.0, convergence=convergence)

        # Evaluate the model
        print("\nLogistic regression model at convergence value of:", convergence)
        logisticRegressionModel.visualize()
        EvaluateBinaryClassification.ExecuteAll(yValidate, logisticRegressionModel.predict(xValidate, classificationThreshold=0.5))

    #############################
    # You may find the following module helpful for making charts. You'll have to install matplotlib (see the lecture notes).
    #
    import MachineLearningCourse.MLUtilities.Visualizations.Charting as Charting

    # trainLosses, validationLosses, and lossXLabels are parallel arrays with the losses you want to plot at the specified x coordinates
    trainLosses = []
    validationLosses = []
    lossXLabels = []

    for step in range(0, 1000, 100):
        logisticRegressionModel.incrementalFit(xTrain, yTrain, maxSteps=100, stepSize=1.0, convergence=0.00001)

        train_loss = logisticRegressionModel.loss(xTrain, yTrain)
        trainLosses.append(train_loss)

        validation_loss = logisticRegressionModel.loss(xValidate, yValidate)
        validationLosses.append(validation_loss)

        lossXLabels.append(step)

    Charting.PlotSeries([trainLosses, validationLosses], ['Train', 'Validate'], lossXLabels, chartTitle="Logistic Regression", xAxisTitle="Gradient Descent Steps", yAxisTitle="Avg. Loss", outputDirectory=kOutputDirectory, fileName="3-Logistic Regression Train vs Validate loss")
