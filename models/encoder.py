from abc import ABC

import tensorflow as tf
from keras.layers import Embedding, GRU
from keras.models import Model

from utils.constants import EMBEDDING_DIM


class Encoder(Model, ABC):
    def __init__(self, vocab_size, embedding_matrix, max_length, num_hidden=256, num_embedding=EMBEDDING_DIM,
                 batch_size=16):
        super(Encoder, self).__init__()
        self.batch_size = batch_size
        self.num_hidden = num_hidden
        self.num_embedding = num_embedding
        self.embedding = Embedding(vocab_size, num_embedding, input_length=max_length, weights=[embedding_matrix],
                                   trainable=False)
        self.gru = GRU(num_hidden, return_sequences=True,
                       recurrent_initializer='glorot_uniform',
                       return_state=True)

    def call(self, x, hidden):
        embedded = self.embedding(x)  # converts integer tokens into a dense representation
        rnn_out, hidden = self.gru(embedded, initial_state=hidden)
        return rnn_out, hidden

    def init_hidden(self):
        return tf.zeros(shape=(self.batch_size, self.num_hidden))
