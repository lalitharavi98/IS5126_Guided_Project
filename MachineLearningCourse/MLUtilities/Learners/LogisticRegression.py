import time
import math
import MachineLearningCourse.MLUtilities.Evaluations.EvaluateBinaryProbabilityEstimate as EvaluateBinaryProbabilityEstimate


class LogisticRegression(object):
    """Stub class for a Logistic Regression Model"""

    def __init__(self, featureCount=None):
        self.isInitialized = False

        if featureCount != None:
            self.__initialize(featureCount)

    def __testInput(self, x, y):
        if len(x) == 0:
            raise UserWarning("Trying to fit but can't fit on 0 training samples.")

        if len(x) != len(y):
            raise UserWarning("Trying to fit but length of x != length of y.")

    def __initialize(self, featureCount):
        self.weights = [0.0 for i in range(featureCount)]
        self.weight0 = 0.0

        self.converged = False
        self.totalGradientDescentSteps = 0

        self.isInitialized = True

    def loss(self, x, y):
        return EvaluateBinaryProbabilityEstimate.LogLoss(y, self.predictProbabilities(x))

    def sigmoid(self, z):
        return 1 / (1 + math.exp(-z))

    def predictProbabilities(self, x):
        # For each sample do the dot product between features and weights (remember the bias weight, weight0)
        #  pass the results through the sigmoid function to convert to probabilities.
        probabilities = []

        for xi in x:
            z = self.weight0
            for feature, weight in zip(xi, self.weights):
                z += feature * weight

            probability = self.sigmoid(z)
            probabilities.append(probability)
        return probabilities

    def predict(self, x, classificationThreshold=0.5):
        predictions = []
        probabilities = self.predictProbabilities(x)
        for probability in probabilities:

            if probability >= classificationThreshold:
                predictions.append(1)
            else:
                predictions.append(0)
        return predictions

    def __gradientDescentStep(self, x, y, stepSize):
        n = len(x)
        predictions = self.predictProbabilities(x)

        # Computing gradient for the bias term
        gradient0 = 0
        for i in range(n):
            gradient0 += predictions[i] - y[i]
        gradient0 = gradient0 / n

        # Computing gradients for each weight
        gradients = [0.0] * len(self.weights)
        for j in range(len(self.weights)):
            gradient = 0
            for i in range(n):
                gradient += (predictions[i] - y[i]) * x[i][j]
            gradients[j] = gradient / n

        # To update bias and weights
        self.weight0 -= stepSize * gradient0
        for j in range(len(self.weights)):
            self.weights[j] = self.weights[j] - stepSize * gradients[j]

        self.totalGradientDescentSteps = self.totalGradientDescentSteps + 1

    # Allows you to partially fit, then pause to gather statistics / output intermediate information, then continue fitting
    def incrementalFit(self, x, y, maxSteps=1, stepSize=1.0, convergence=0.005):
        self.__testInput(x, y)
        if self.isInitialized == False:
            self.__initialize(len(x[0]))

        previousLoss = float('inf')
        # do a maximum of 'maxSteps' of gradient descent with the indicated stepSize (use the __gradientDescentStep stub function for code clarity).
        for _ in range(maxSteps):
            self.__gradientDescentStep(x, y, stepSize)
            currentLoss = self.loss(x, y)
            #  stop and set self.converged to true if the mean log loss on the training set decreases by less than 'convergence' on a gradient descent step.
            if abs(previousLoss - currentLoss) < convergence:
                self.converged = True
                break

            previousLoss = currentLoss

        # do a maximum of 'maxSteps' of gradient descent with the indicated stepSize (use the __gradientDescentStep stub function for code clarity).
        #  stop and set self.converged to true if the mean log loss on the training set decreases by less than 'convergence' on a gradient descent step.

        # print("Stub incrementalFit in ", __file__)

    def fit(self, x, y, maxSteps=50000, stepSize=1.0, convergence=0.005, verbose=True):

        startTime = time.time()

        self.incrementalFit(x, y, maxSteps=maxSteps, stepSize=stepSize, convergence=convergence)

        endTime = time.time()
        runtime = endTime - startTime

        if not self.converged:
            print("Warning: did not converge after taking the maximum allowed number of steps.")
        elif verbose:
            print(
                "LogisticRegression converged in %d steps (%.2f seconds) -- %d features. Hyperparameters: stepSize=%f and convergence=%f." % (
                    self.totalGradientDescentSteps, runtime, len(self.weights), stepSize, convergence))

    def visualize(self):
        print("w0: %f " % self.weight0, end='')

        for i in range(len(self.weights)):
            print("w%d: %f " % (i + 1, self.weights[i]), end='')

        print("\n")
