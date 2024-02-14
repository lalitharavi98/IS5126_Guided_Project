from collections import Counter

import math


class SMSSpamFeaturize(object):
    """A class to coordinate turning SMS spam strings into feature vectors."""

    def __init__(self, useHandCraftedFeatures=False):
        # use handcrafted features specified in _FeaturizeXForHandCrafted()
        self.useHandCraftedFeatures = useHandCraftedFeatures

        self.ResetVocabulary()

    def ResetVocabulary(self):
        self.vocabularyCreated = False
        self.vocabulary = []

    def Tokenize(self, xRaw):
        return str.split(xRaw)

    """
    Question 4a: Implementing the stub functions:
    -> FindMostFrequentWords(self, x, n):
    -> FindTopWordsByMutualInformation(self, x, y, n)

    """

    def FindMostFrequentWords(self, x, n):
        # print("Stub FindMostFrequentWords in ", __file__)
        # Use Counter object to count word frequencies
        word_counts = Counter()
        for text in x:
            words = text.split()  # Tokenize by whitespace
            word_counts.update(words)

        # Return the top n most frequent words
        return [word for word, _ in word_counts.most_common(n)]

        """
        Finds the top n words with the highest mutual information with the labels.
        """

    def FindTopWordsByMutualInformation(self, x, y, n):
        # print("Stub FindTopWordsByMutualInformation in ", __file__)
        # Count occurrences of words, class 0, and class 1 for mutual information calculation
        word_counts = Counter()
        class_0_counts = Counter()
        class_1_counts = Counter()

        # Initialize mutual information scores
        mi_scores = Counter()

        # Calculate class probabilities
        """
        Probability(spam) = (Number of messages that are labelled spam/(Number of messages that are labelled spam + Number of messages that are labelled non-spam))
        Similarly (non_spam) = (Number of messages that are labelled non-spam/(Number of messages that are labelled spam + Number of messages that are labelled non spam))

        """

        p_spam = sum(1 for label in y if label == 1) / len(y)
        p_non_spam = 1 - p_spam

        # Iterate over each example in the dataset to count word occurrences and class-specific word occurrences
        for i in range(len(x)):

            words = x[i].split()
            # Use a set to keep track of already counted words in the current text
            counted = set()

            for word in words:
                # Check if the word has not been counted yet in the current text
                if word not in counted:
                    # Increment the count of the word for overall occurrences
                    word_counts[word] += 1
                    # Increment the count of the word for class 0 if the label is 0
                    if y[i] == 0:
                        class_0_counts[word] += 1
                    else:
                        class_1_counts[word] += 1
                    counted.add(word)

        # Calculate probabilities for mutual information calculation
        """
         MI(word, label) = ∑ P(word, label) * log2(P(word, label) / (P(word) * P(label)))

         where :
         P(word, label): Joint probability of the word and the label occurring together.
         P(word): Probability of the word appearing in any message.
         P(label): Probability of the label occurring in any message.
         log2: Logarithm with base 2
        """
        for word in word_counts:
            # Calculate probabilities with smoothing
            p_word = (word_counts[word] + 1) / (len(y) + 2)
            p_not_word = 1 - p_word

            # Calculate conditional probabilities
            p_word_spam = (
                                  class_1_counts[word] + 1) / (class_0_counts[word] + class_1_counts[word] + 2) * p_word
            p_word_non_spam = (
                                      class_0_counts[word] + 1) / (
                                          class_0_counts[word] + class_1_counts[word] + 2) * p_word

            p_not_word_spam = p_spam - p_word_spam
            p_not_word_non_spam = p_non_spam - p_word_non_spam

            # Calculate mutual information score for the word
            mi_scores[word] = (
                    p_word_spam * math.log(p_word_spam / (p_spam * p_word)) +
                    p_word_non_spam * math.log(p_word_non_spam / (p_non_spam * p_word)) +
                    p_not_word_spam * math.log(p_not_word_spam / (p_spam * p_not_word)) +
                    p_not_word_non_spam *
                    math.log(p_not_word_non_spam / (p_non_spam * p_not_word))
            )

        # Return the top n words based on mutual information
        return [word for word, _ in mi_scores.most_common(n)]

    """
    Question 4 : Updating the function CreateVocabulary

    """

    def CreateVocabulary(self, xTrainRaw, yTrainRaw, numFrequentWords=0, numMutualInformationWords=0,
                         supplementalVocabularyWords=[]):
        if self.vocabularyCreated:
            raise UserWarning(
                "Calling CreateVocabulary after the vocabulary was already created. Call ResetVocabulary to reinitialize.")

        # This function will eventually scan the strings in xTrain and choose which words to include in the vocabulary.
        #   But don't implement that until you reach the assignment that requires it...

        # Calling the function FindMostFrequentWords to get the top frequent words
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

        # This function can produce an array of hand-crafted features to add on top of the vocabulary-related features
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
