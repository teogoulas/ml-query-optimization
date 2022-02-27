import numpy as np
import tensorflow as tf

from utils.constants import SOS_token


def train_step(input_tensor, target_tensor, enc_hidden, encoder, decoder, optimizer, loss_fn, acc_metric):
    loss = 0.0

    with tf.GradientTape() as tape:
        batch_size = input_tensor.shape[0]
        enc_output, en_state_h, en_state_c = encoder(input_tensor, enc_hidden)

        SOS_tensor = np.array([SOS_token])
        dec_input = tf.squeeze(tf.expand_dims([SOS_tensor] * batch_size, 1), -1)

        for tx in range(target_tensor.shape[1] - 1):
            dec_out, de_state_h, de_state_c, _ = decoder(dec_input, (en_state_h, en_state_c),
                                                         enc_output)
            loss += loss_fn(target_tensor[:, tx], dec_out)
            acc_metric.update_state(target_tensor[:, tx], dec_out)
            dec_input = tf.expand_dims(target_tensor[:, tx], 1)

    batch_loss = loss / target_tensor.shape[1]
    batch_accuracy = acc_metric.result()
    acc_metric.reset_states()
    t_variables = encoder.trainable_variables + decoder.trainable_variables
    gradients = tape.gradient(loss, t_variables)
    optimizer.apply_gradients(zip(gradients, t_variables))
    return batch_loss, batch_accuracy


def test_step(input_tensor, target_tensor, enc_hidden, encoder, decoder, loss_fn, acc_metric):
    loss = 0.0
    batch_size = input_tensor.shape[0]
    enc_output, en_state_h, en_state_c = encoder(input_tensor, enc_hidden)

    SOS_tensor = np.array([SOS_token])
    dec_input = tf.squeeze(tf.expand_dims([SOS_tensor] * batch_size, 1), -1)

    for tx in range(target_tensor.shape[1] - 1):
        dec_out, de_state_h, de_state_c, _ = decoder(dec_input, (en_state_h, en_state_c),
                                                     enc_output)
        loss += loss_fn(target_tensor[:, tx], dec_out)
        acc_metric.update_state(target_tensor[:, tx], dec_out)
        dec_input = tf.expand_dims(target_tensor[:, tx], 1)

    batch_loss = loss / target_tensor.shape[1]
    batch_accuracy = acc_metric.result()
    acc_metric.reset_states()
    return batch_loss, batch_accuracy
