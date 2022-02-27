from abc import ABC

import tensorflow as tf
from keras.layers import Embedding, LSTM
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
        self.lstm = LSTM(num_hidden, return_sequences=True,
                         recurrent_initializer='glorot_uniform',
                         return_state=True)

    def call(self, input_sequence, states):
        embedded = self.embedding(input_sequence)  # converts integer tokens into a dense representation
        # print(f"embedded shape = {embedded.shape}")
        # print(f"hidden shape = {hidden.shape}")
        output, state_h, state_c = self.lstm(embedded, initial_state=states)
        # print(f"rnn_out shape = {rnn_out.shape}")
        # print(f"hidden shape = {hidden.shape}")
        return output, state_h, state_c

    def init_hidden(self, batch_size):
        # Return a all 0s initial states
        return (tf.zeros([self.batch_size, self.num_hidden]),
                tf.zeros([self.batch_size, self.num_hidden]))