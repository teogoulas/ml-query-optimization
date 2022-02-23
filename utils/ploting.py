import random

import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE

from models.embeddings_model import EmbeddingsModel


def plot_embeddings(encoder: EmbeddingsModel, samples: int, title: str):
    vocab_size, embedding_size = encoder.model.wv.vectors.shape
    # Sample random words from model dictionary
    random_i = random.sample(range(vocab_size), samples)
    random_w = [encoder.model.wv.index_to_key[i] for i in random_i]

    # Generate Word2Vec embeddings of each word
    word_vecs = np.array([encoder.model.wv[w] for w in random_w])

    # Apply t-SNE to Word2Vec embeddings, reducing to 2 dims
    tsne = TSNE()
    tsne_e = tsne.fit_transform(word_vecs)

    # Plot t-SNE result
    plt.figure(figsize=(32, 32))
    plt.title(title)
    plt.scatter(tsne_e[:, 0], tsne_e[:, 1], marker='o', c=range(len(random_w)), cmap=plt.get_cmap('Spectral'))

    for label, x, y, in zip(random_w, tsne_e[:, 0], tsne_e[:, 1]):
        plt.annotate(label,
                     xy=(x, y), xytext=(0, 15),
                     textcoords='offset points', ha='right', va='bottom',
                     bbox=dict(boxstyle='round, pad=0.2', fc='yellow', alpha=0.1))
    plt.show()
