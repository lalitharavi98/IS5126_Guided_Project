import math

_zValues = { .5:.67, .68:1.0, .8:1.28, .9:1.64, .95:1.96, .98:2.33, .99:2.58 }

def GetAccuracyBounds(mean, sampleSize, confidence):
    if mean < 0.0 or mean > 1.0:
        raise UserWarning("mean must be between 0 and 1")

    if sampleSize <= 0:
        raise UserWarning("sampleSize should be positive")

    variance = mean * (1-mean)

    stdev = math.sqrt(variance)
    
    try:
        zValue = _zValues[confidence]
    except:
        raise UserWarning("Trying to get bounds with confidence %f, but that isn't in the _zValues table" % (confidence))

    lower = mean - zValue * (stdev / math.sqrt(sampleSize))
    upper = mean + zValue * (stdev / math.sqrt(sampleSize))
    return (lower, upper)