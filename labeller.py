# External dependencies.
import numpy as np
from scipy.io import mmread
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix

# Internal dependencies.
from sampler import GibbsSampler
from document import Document

# Read BBC data.
# Term-frequencies for each document.
def doc_term_frequencies():
    return np.array(mmread('bbc/bbc.mtx').todense()).T

# Labels for each document.
def doc_labels():
    with open('bbc/bbc.classes', encoding='utf-8') as f:
        lines = f.readlines()

        # Remove comments.
        while lines[0][0] == '%':
            lines.pop(0)
        
        # Fill in.
        labels = np.zeros(len(lines), dtype='int')
        for index, line in enumerate(lines):
            labels[index] = int(line.split()[1])

    return labels

# Seed for randomness.
random_seed = 0
np.random.seed(random_seed)

# Load data.
term_frequencies = doc_term_frequencies()
labels = doc_labels()

# Keep only 20% of the labels.
visible_labels = [label if np.random.uniform(0, 1) < 0.2 else None for label in labels]

# Assign to documents.
docs = [Document(term_frequencies[index], label=visible_labels[index]) for index in range(len(labels))]

# Initialize sampler.
sampler = GibbsSampler(docs, num_classes=5, random_seed=random_seed)

# Gibbs sampling now!
num_iterations = 25
for iteration in range(num_iterations):
    sampler.sample()
    print('Iteration %d of Gibbs sampling complete!' % iteration)

# Obtain predicted labels.
predicted_labels = sampler.predict()
print('Predicted labels: %s' % predicted_labels)

# Plotting.
# Set styles.
sns.set_style('darkgrid')
sns.set(font='DejaVu Sans')

# Plot 2-dimensional tSNE representation of documents.
term_frequencies_reduced = TSNE(n_components=2).fit_transform(term_frequencies)
fig, axs = plt.subplots(ncols=2)
axs[0].scatter(term_frequencies_reduced[:, 0], term_frequencies_reduced[:, 1], c=labels)
axs[1].scatter(term_frequencies_reduced[:, 0], term_frequencies_reduced[:, 1], c=predicted_labels)
axs[0].set_xlabel('True Labels')
axs[1].set_xlabel('Predicted Labels')
plt.suptitle('tSNE Representation of Labelled Documents')
plt.show()

# Plot confusion matrix.
sns.heatmap(confusion_matrix(labels, predicted_labels), annot=True, fmt='d')
plt.show()