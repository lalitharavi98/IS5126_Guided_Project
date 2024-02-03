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
    confusion_matrix = ConfusionMatrix(y, yPredicted)
    true_positives = confusion_matrix[1][1]
    false_positives = confusion_matrix[0][1]
    if true_positives + false_positives == 0:
        return 0.0
    return true_positives / (true_positives + false_positives)

def Recall(y, yPredicted):
    print("Stub Recall in ", __file__)
    confusion_matrix = ConfusionMatrix(y, yPredicted)
    true_positives = confusion_matrix[1][1]
    false_negatives = confusion_matrix[1][0]

    if true_positives + false_negatives == 0:
        return 0.0
    return true_positives / (true_positives + false_negatives)

def FalseNegativeRate(y, yPredicted):
    print("Stub FalseNegativeRate in ", __file__)
    confusion_matrix = ConfusionMatrix(y, yPredicted)
    false_negatives = confusion_matrix[1][0]
    true_positives = confusion_matrix[1][1]
    if false_negatives + true_positives == 0:
        return 0.0

    return false_negatives / (false_negatives + true_positives)

def FalsePositiveRate(y, yPredicted):
    print("Stub FalsePositiveRate in ", __file__)
    confusion_matrix = ConfusionMatrix(y, yPredicted)
    false_positives = confusion_matrix[0][1]
    true_negatives = confusion_matrix[0][0]

    if false_positives + true_negatives == 0:
        return 0.0

    return false_positives / (false_positives + true_negatives)


def ConfusionMatrix(y, yPredicted):
    # This function should return: [[<# True Negatives>, <# False Positives>], [<# False Negatives>, <# True Positives>]]
    #  Hint: writing this function first might make the others easier...
    print("Stub preConfusionMatrix in ", __file__)
    true_positives = 0
    false_positives = 0
    true_negatives = 0
    false_negatives = 0
    for actual, predicted in zip(y, yPredicted):
        if actual == 0 and predicted == 0:
            true_negatives += 1
        elif actual == 0 and predicted == 1:
            false_positives += 1
        elif actual == 1 and predicted == 0:
            false_negatives += 1
        elif actual == 1 and predicted == 1:
            true_positives += 1

    return [[true_negatives, false_positives], [false_negatives, true_positives]]

def ExecuteAll(y, yPredicted):
    print("Confusion Matrix:", ConfusionMatrix(y, yPredicted))
    print("Accuracy: %.2f" % (Accuracy(y, yPredicted)))
    print("Precision: %.2f" % (Precision(y, yPredicted)))
    print("Recall: %.2f" % (Recall(y, yPredicted)))
    print("FPR: %.2f" % (FalsePositiveRate(y, yPredicted)))
    print("FNR: %.2f" %(FalseNegativeRate(y, yPredicted)))

