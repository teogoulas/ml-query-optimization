from abc import ABC

import tensorflow as tf

from keras.layers import Dense
from keras.models import Model


class BahdanauAttention(Model, ABC):
    def __init__(self, units):
        super(BahdanauAttention, self).__init__()

        self.W1 = Dense(units)
        self.W2 = Dense(units)
        self.V = Dense(1)

    def call(self, encoder_out, hidden):
        # shape of encoder_out : batch_size, seq_length, hidden_dim (16, 10, 1024)
        # shape of encoder_hidden : batch_size, hidden_dim (16, 1024)

        hidden = tf.expand_dims(hidden, axis=1)  # out: (16, 1, 1024)

        score = self.V(tf.nn.tanh(self.W1(encoder_out) + self.W2(hidden)))  # out: (16, 10, 1)

        attn_weights = tf.nn.softmax(score, axis=1)

        context = attn_weights * encoder_out  # out: ((16,10,1) * (16,10,1024))=16, 10, 1024
        context = tf.reduce_sum(context, axis=1)  # out: 16, 1024
        return context, attn_weights
