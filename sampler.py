# External dependencies.
from numpy.random import beta, dirichlet, uniform, choice
import numpy as np
from collections import defaultdict


# Predicting classes (0 vs 1) by maximising the posterior distribution over each document.
class GibbsSampler:
    def __init__(self, docs, num_classes=2, random_seed=0):
        
        # Seed random number generator.
        np.random.seed(random_seed)

        # Assign variables.
        self.docs = docs
        self.num_words = docs[0].bag_of_words.shape[0]
        self.num_classes = num_classes

        # Hyperparameters.
        self.gamma_pi = np.ones(self.num_classes)
        self.gamma_theta = np.ones(self.num_words)

        # Initialize latent variables.
        self.init_pi = dirichlet(self.gamma_pi)
        self.thetas = [dirichlet(self.gamma_theta) for _ in range(num_classes)]
        
        # Initialize class-based aggregates required for sampling.
        self.class_count = np.zeros(self.num_classes)
        self.class_word_frequencies = np.zeros((self.num_classes, self.num_words))

        # Initialize labels.
        for doc in self.docs:
            if doc.label is None:
                doc.label = choice(self.num_classes, p=self.init_pi)
    
            self.class_count[doc.label] += 1
            self.class_word_frequencies[doc.label] += doc.bag_of_words

        # Initialize history.
        self.history = {'labels': defaultdict(list), 'thetas': []}
        self.update_history()
    
    # One iteration of Gibbs sampling over all unobserved variables.
    def sample(self):

        # Update document labels.
        for doc in self.docs:
            if doc.type == 'unlabelled':
                # Temporarily remove label.
                self.class_count[doc.label] -= 1
                self.class_word_frequencies[doc.label] -= doc.bag_of_words
                
                # Compute probabilities over labels.
                log_probabilities = np.zeros(self.num_classes)
                for label in range(self.num_classes):
                    log_probabilities[label] += np.log((self.class_count[label] + self.gamma_pi[label] - 1) / (len(self.docs) + np.sum(self.gamma_pi) - 1))
                    log_probabilities[label] += np.sum(np.multiply(doc.bag_of_words, np.log(self.thetas[label])))

                # Convert to probabilities, once we have cancelled out common factors. 
                log_probabilities -= np.max(log_probabilities)
                probabilities = np.exp(log_probabilities)

                # Normalize so that these probabilities sum up to 1.
                probabilities /= np.sum(probabilities)

                # Chose new label as weighted choice over labels' probabilities.
                doc.label = choice(self.num_classes, p=probabilities)

                # Update class counts.
                self.class_count[doc.label] += 1
                self.class_word_frequencies[doc.label] += doc.bag_of_words
        
        # Update latent thetas. Latent pis have been integrated over.
        self.thetas = []
        for label in range(self.num_classes):
            self.apparent_gamma_theta = self.gamma_theta.copy()
            self.apparent_gamma_theta += self.class_word_frequencies[label]
            self.thetas.append(dirichlet(self.apparent_gamma_theta))
        
        # Update history of variable values.
        self.update_history()


    # Update history of variable values.
    def update_history(self):
        for doc_index, doc in enumerate(self.docs):
            self.history['labels'][doc_index].append(doc.label)
        self.history['thetas'].append(self.thetas)


    # Predict a label for each document by taking the most-seen label over all iterations.
    def predict(self):
        labels = np.zeros(len(self.docs), dtype='int')
        for doc_index, _ in enumerate(self.docs):
            labels[doc_index] = np.argmax(np.bincount(self.history['labels'][doc_index]))
        
        return labels