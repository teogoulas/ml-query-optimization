from abc import ABC

import tensorflow as tf
from keras.layers import Embedding, Dense, LSTM
from keras.models import Model

from models.luong_attention import LuongAttention
from utils.constants import EMBEDDING_DIM


class Decoder(Model, ABC):
    def __init__(self, vocab_size, embedding_matrix, max_length, attention_func, dec_dim=256,
                 num_embedding=EMBEDDING_DIM, batch_size=16):
        super(Decoder, self).__init__()

        self.attn = LuongAttention(dec_dim, attention_func)
        self.embedding = Embedding(vocab_size, num_embedding, input_length=max_length, weights=[embedding_matrix],
                                   trainable=False)
        self.lstm = LSTM(dec_dim, recurrent_initializer='glorot_uniform',
                         return_sequences=True, return_state=True)
        self.fc = Dense(dec_dim, activation='tanh')
        self.fs = Dense(vocab_size)

    def call(self, input_sequence, state, encoder_output):
        # Remember that the input to the decoder
        # is now a batch of one-word sequences,
        # which means that its shape is (batch_size, 1)
        embed = self.embedding(input_sequence)

        # Therefore, the lstm_out has shape (batch_size, 1, hidden_dim)
        lstm_out, state_h, state_c = self.lstm(embed, initial_state=state)

        # Use self.attn to compute the context and alignment vectors
        # context vector's shape: (batch_size, 1, hidden_dim)
        # alignment vector's shape: (batch_size, 1, source_length)
        context, alignment = self.attn(lstm_out, encoder_output)

        # Combine the context vector and the LSTM output
        # Before combined, both have shape of (batch_size, 1, hidden_dim),
        # so let's squeeze the axis 1 first
        # After combined, it will have shape of (batch_size, 2 * hidden_dim)
        lstm_out = tf.concat(
            [tf.squeeze(context, 1), tf.squeeze(lstm_out, 1)], 1)

        # lstm_out now has shape (batch_size, hidden_dim)
        lstm_out = self.fc(lstm_out)

        # Finally, it is converted back to vocabulary space: (batch_size, vocab_size)
        logits = self.fs(lstm_out)

        return logits, state_h, state_c, alignment
