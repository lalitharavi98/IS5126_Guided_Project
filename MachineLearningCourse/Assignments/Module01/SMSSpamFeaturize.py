from collections import Counter
import numpy as np

class SMSSpamFeaturize(object):
    """A class to coordinate turning SMS spam strings into feature vectors."""

    def __init__(self, useHandCraftedFeatures=False):
        # use hand crafted features specified in _FeaturizeXForHandCrafted()
        self.useHandCraftedFeatures = useHandCraftedFeatures

        self.ResetVocabulary()

    def ResetVocabulary(self):
        self.vocabularyCreated = False
        self.vocabulary = []

    def Tokenize(self, xRaw):
        return str.split(xRaw)

    def FindMostFrequentWords(self, x, n):
        print("Stub FindMostFrequentWords in ", __file__)
        # Use Counter object to count word frequencies
        word_counts = Counter()
        for text in x:
            words = text.split()  # Tokenize by whitespace
            word_counts.update(words)

        # Return the top n most frequent words
        return [word for word, _ in word_counts.most_common(n)]

    def FindTopWordsByMutualInformation(self, x, y, n):
        print("Stub FindTopWordsByMutualInformation in ", __file__)

        """
        Finds the top n words with the highest mutual information with the labels.
        """

        # Get the statistics to calculate joint probability of word with the labels
        c = Counter()
        c0 = Counter()
        c1 = Counter()
        for i in range(len(x)):
            words = x[i].split()
            counted = set()
            for word in words:
                if word not in counted:
                    c[word] += 1
                    if y[i] == 0:
                        c0[word] += 1
                    elif y[i] == 1:
                        c1[word] += 1
                counted.add(word)
        # p(0) = count_sentences_with_label_0 / ( count_sentence_with_label_0 + count_sentence_with_label_1 )
        # ...
        p1 = sum(y) / len(y)
        p0 = 1 - p1
        mi = Counter()
        for word in c:
            # p(word) = count_sentences_with_word / ( count_sentence_with_word + count_sentence_without_word )
            pWord = (c[word] + 1) / (len(y) + 2)
            pNotWord = 1 - pWord
            # p(word, 0) = p(0|word) * p(word)
            pWordAnd0 = (c0[word] + 1) / (c0[word] + c1[word] + 2) * pWord
            pWordAnd1 = (c1[word] + 1) / (c0[word] + c1[word] + 2) * pWord
            # p(~word, 0) = p(0|~word) * p(~word) or p0 - p(word, 0)
            pNotWordAnd0 = p0 - pWordAnd0
            pNotWordAnd1 = p1 - pWordAnd1

            mi[word] = pWordAnd0 * np.log(pWordAnd0 / (p0 * pWord)) + pWordAnd1 * np.log(pWordAnd1 / (p1 * pWord)) + \
                       pNotWordAnd0 * np.log(pNotWordAnd0 / (p0 * pNotWord)) + \
                       pNotWordAnd1 * np.log(pNotWordAnd1 / (p1 * pNotWord))

        return [wordtoken[0] for wordtoken in mi.most_common(n)]

        # Question 4 : Updating the function CreateVocabulary

    def CreateVocabulary(self, xTrainRaw, yTrainRaw, numFrequentWords=0, numMutualInformationWords=0,
                         supplementalVocabularyWords=[]):
        if self.vocabularyCreated:
            raise UserWarning(
                "Calling CreateVocabulary after the vocabulary was already created. Call ResetVocabulary to reinitialize.")

        # This function will eventually scan the strings in xTrain and choose which words to include in the vocabulary.
        #   But don't implement that until you reach the assignment that requires it...
        top_frequent_words = self.FindMostFrequentWords(
            xTrainRaw, numFrequentWords)
        top_mutual_info_words = self.FindTopWordsByMutualInformation(
            xTrainRaw, yTrainRaw, numMutualInformationWords)
        self.vocabulary = top_frequent_words + top_mutual_info_words

        # For now, only use words that are passed in
        self.vocabulary = self.vocabulary + supplementalVocabularyWords
        
        self.vocabularyCreated = True

    def _FeaturizeXForVocabulary(self, xRaw):
        features = []
        
        # for each word in the vocabulary output a 1 if it appears in the SMS string, or a 0 if it does not
        tokens = self.Tokenize(xRaw)
        for word in self.vocabulary:
            if word in tokens:
                features.append(1)
            else:
                features.append(0)

        return features

    def _FeaturizeXForHandCraftedFeatures(self, xRaw):
        features = []

        # This function can produce an array of hand-crafted features to add on top of the vocabulary related features
        if self.useHandCraftedFeatures:
            # Have a feature for longer texts
            if (len(xRaw) > 40):
                features.append(1)
            else:
                features.append(0)

            # Have a feature for texts with numbers in them
            if (any(i.isdigit() for i in xRaw)):
                features.append(1)
            else:
                features.append(0)

        return features

    def _FeatureizeX(self, xRaw):
        return self._FeaturizeXForVocabulary(xRaw) + self._FeaturizeXForHandCraftedFeatures(xRaw)

    def Featurize(self, xSetRaw):
        return [self._FeatureizeX(x) for x in xSetRaw]

    def GetFeatureInfo(self, index):
        if index < len(self.vocabulary):
            return self.vocabulary[index]
        else:
            # return the zero based index of the heuristic feature
            return "Heuristic_%d" % (index - len(self.vocabulary))
