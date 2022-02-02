from abc import ABC

import tensorflow as tf

from keras.layers import Embedding, GRU, Dense
from keras.models import Model

from models.bahdanau_attention import BahdanauAttention
from utils.constants import EMBEDDING_DIM


class Decoder(Model, ABC):
    def __init__(self, vocab_size, embedding_matrix, max_length, dec_dim=256, num_embedding=EMBEDDING_DIM):
        super(Decoder, self).__init__()

        self.attn = BahdanauAttention(dec_dim)
        self.embedding = Embedding(vocab_size, num_embedding, input_length=max_length, weights=[embedding_matrix],
                                   trainable=False)
        self.gru = GRU(dec_dim, recurrent_initializer='glorot_uniform',
                       return_sequences=True, return_state=True)
        self.fc = Dense(vocab_size)

    def call(self, x, hidden, enc_out):
        # print("x shape: {}".format(x.shape))
        # print("enc_out shape: {}".format(enc_out.shape))
        # print("enc_hidden shape: {}".format(enc_hidden.shape))
        x = self.embedding(x)
        # print("embedding shape: {}".format(x.shape))
        context, attn_weights = self.attn(enc_out, hidden)
        # print("BahdanauAttention shape: {}".format(context.shape))
        x = tf.concat((tf.expand_dims(context, 1), x), -1)
        # x.shape = (16, 1, e_c_hidden_size + d_c_embedding_size)
        r_out, hidden = self.gru(x, initial_state=hidden)
        out = tf.reshape(r_out, shape=(-1, r_out.shape[2]))
        # print("out.shape: {}".format(out.shape))
        return self.fc(out), hidden, attn_weights
