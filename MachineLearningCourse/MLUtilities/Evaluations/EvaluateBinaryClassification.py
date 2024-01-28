# This file contains stubs for evaluating binary classifications. You must complete these functions as part of your assignment.
#     Each function takes in: 
#           'y':           the arrary of 0/1 true class labels; 
#           'yPredicted':  the prediction your model made for the cooresponding example.


def __CheckEvaluationInput(y, yPredicted):
    # Check sizes
    if(len(y) != len(yPredicted)):
        raise UserWarning("Attempting to evaluate between the true labels and predictions.\n   Arrays contained different numbers of samples. Check your work and try again.")

    # Check values
    valueError = False
    for value in y:
        if value not in [0, 1]:
            valueError = True
    for value in yPredicted:
        if value not in [0, 1]:
            valueError = True

    if valueError:
        raise UserWarning("Attempting to evaluate between the true labels and predictions.\n   Arrays contained unexpected values. Must be 0 or 1.")

def Accuracy(y, yPredicted):
    __CheckEvaluationInput(y, yPredicted)

    correct = []
    for i in range(len(y)):
        if(y[i] == yPredicted[i]):
            correct.append(1)
        else:
            correct.append(0)

    return sum(correct)/len(correct)

def Precision(y, yPredicted):
    print("Stub precision in ", __file__)
    return 0.0

def Recall(y, yPredicted):
    print("Stub Recall in ", __file__)
    return 0.0

def FalseNegativeRate(y, yPredicted):
    print("Stub FalseNegativeRate in ", __file__)
    return 0.0

def FalsePositiveRate(y, yPredicted):
    print("Stub FalsePositiveRate in ", __file__)
    return 0.0

def ConfusionMatrix(y, yPredicted):
    # This function should return: [[<# True Negatives>, <# False Positives>], [<# False Negatives>, <# True Positives>]]
    #  Hint: writing this function first might make the others easier...
    
    print("Stub preConfusionMatrix in ", __file__)
    return None

def ExecuteAll(y, yPredicted):
    print("Confusion Matrix:", ConfusionMatrix(y, yPredicted))
    print("Accuracy: %.2f" % (Accuracy(y, yPredicted)))
    print("Precision: %.2f" % (Precision(y, yPredicted)))
    print("Recall: %.2f" % (Recall(y, yPredicted)))
    print("FPR: %.2f" % (FalsePositiveRate(y, yPredicted)))
    print("FNR: %.2f" %(FalseNegativeRate(y, yPredicted)))

