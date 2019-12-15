
import numpy as np

# Representing each document as a bag of words.
class Document:
    @staticmethod
    def frequencies(doc, vocab):
        word_frequencies = np.zeros(len(vocab))
        for word in doc:
            word_frequencies[vocab[word]] += 1
        return word_frequencies

    def __init__(self, bag_of_words, label=None):
        if bag_of_words.shape[0] == 1:
            self.bag_of_words = np.ravel(bag_of_words[:, 0])
        else:
            self.bag_of_words = bag_of_words
            
        self.label = label

        # Is this document prelabelled, or not?
        if self.label is None:
            self.type = 'unlabelled'
        else:
            self.type = 'prelabelled'