import math

def __CheckEvaluationInput(y, yPredicted):
    # Check sizes
    if (len(y) != len(yPredicted)):
        raise UserWarning("Attempting to evaluate between the true labels and predictions.\n   Arrays contained different numbers of samples. Check your work and try again.")

    # Check values
    valueError = False
    for value in y:
        if value < 0 or value > 1:
            valueError = True
    for value in yPredicted:
        if value < 0 or value > 1:
            valueError = True

    if valueError:
        raise UserWarning("Attempting to evaluate between the true labels and predictions.\n   Arrays contained unexpected values. Must be between 0 and 1.")

def MeanSquaredErrorLoss(y, yPredicted):
    __CheckEvaluationInput(y, yPredicted)

    n = len(y)
    sumOfSquaredErrors = 0
    for actual, predicted in zip(y, yPredicted):
        error = actual - predicted
        squaredError = error ** 2
        sumOfSquaredErrors += squaredError
    mse = sumOfSquaredErrors / n

    return mse

    # print("Stub MeanSquaredErrorLoss in ", __file__)
    # return 0.0


def LogLoss(y, yPredicted):
    __CheckEvaluationInput(y, yPredicted)

    n = len(y)
    epsilon = 1e-15  # to prevent possibility of  log(0)
    logLossSum = 0
    for actual, predicted in zip(y, yPredicted):
        predictedPlusEpsilon = predicted + epsilon
        term1 = actual * math.log(predictedPlusEpsilon)
        oneMinusActual = 1 - actual
        term2 = oneMinusActual * math.log(1 - predictedPlusEpsilon)
        logLossContribution = term1 + term2
        logLossSum -= logLossContribution
    log_loss = logLossSum / n
    return log_loss

    # print("Stub LogLoss in ", __file__)
    # return 0.0
