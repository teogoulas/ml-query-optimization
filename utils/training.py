import numpy as np
import tensorflow as tf

from keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.optimizers import Adam

from utils.constants import SOS_token


def loss_fn(real, pred):
    criterion = SparseCategoricalCrossentropy(from_logits=True, reduction='none')
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    _loss = criterion(real, pred)
    mask = tf.cast(mask, dtype=_loss.dtype)
    _loss *= mask
    return tf.reduce_mean(_loss)


def train_step(input_tensor, target_tensor, enc_hidden, encoder, decoder):
    optimizer = Adam()
    loss = 0.0

    with tf.GradientTape() as tape:
        batch_size = input_tensor.shape[0]
        enc_output, enc_hidden = encoder(input_tensor, enc_hidden)

        SOS_tensor = np.array([SOS_token])
        dec_input = tf.squeeze(tf.expand_dims([SOS_tensor] * batch_size, 1), -1)
        dec_hidden = enc_hidden

        for tx in range(target_tensor.shape[1] - 1):
            dec_out, dec_hidden, _ = decoder(dec_input, dec_hidden,
                                             enc_output)
            loss += loss_fn(target_tensor[:, tx], dec_out)
            dec_input = tf.expand_dims(target_tensor[:, tx], 1)

    batch_loss = loss / target_tensor.shape[1]
    t_variables = encoder.trainable_variables + decoder.trainable_variables
    gradients = tape.gradient(loss, t_variables)
    optimizer.apply_gradients(zip(gradients, t_variables))
    return batch_loss


def checkpoint(model, name=None):
    if name is not None:
        model.save_weights('data/weights/{}.h5'.format(name))
    else:
        raise NotImplementedError
