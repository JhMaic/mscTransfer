import tensorflow as tf
from tensorflow import keras as keras
from tensorflow.keras import backend as K
from tensorflow.keras.layers import *
from tensorflow.keras import Model



class Linear(Layer):
    def __init__(self,
                 output_dim,
                 use_bias=True,
                 activation=None,
                 kernel_initializer=None,
                 **kwargs):
        super(Linear, self).__init__(**kwargs)
        self.activ = activation
        self.output_dim = output_dim
        self.use_bias = use_bias
        self.kernel_initializer = kernel_initializer

    def build(self, input_shape):
        self.w = self.add_weight(name='weight',
                                 shape=[input_shape[-1], self.output_dim],
                                 initializer=self.kernel_initializer,
                                 trainable=self.trainable,
                                 dtype='float32')

        self.b = self.add_weight(name='bias',
                                 shape=[self.output_dim],
                                 initializer='zeros',
                                 trainable=self.trainable,
                                 dtype='float32') if self.use_bias else 0.

    def call(self, inputs, **kwargs):
        return keras.activations.get(self.activ)(K.dot(inputs, self.w) + self.b)

class RNNReadout(Layer):
    def __init__(self,
                 gru_cell,
                 note_embedder,
                 note_decoder,
                 newstate=True,
                 **kwargs):
        super(RNNReadout, self).__init__(**kwargs)
        self.cell = gru_cell
        self.embedder = note_embedder
        self.decoder = note_decoder
        self.hdim = self.cell.units
        self.odim = 278
        self.newstate = newstate

    def step(self, inputs, states):
        lst_output, states = states[0], states[1]
        init_states = states

        time_notes = []
        for i in range(4):
            e_in = self.embedder(lst_output)
            x = tf.concat([e_in, inputs[:, i]], axis=-1)

            h, states = self.cell(x, states=states)
            notes = self.decoder(h, force=True)
            lst_output = notes
            time_notes += [notes[:, tf.newaxis, ...]]

        time_notes = tf.concat(time_notes, axis=1)

        return time_notes, [notes, init_states if self.newstate else states]

    def call(self, inputs, **kwargs):
        # cond_in = (b_a, m_a)
        o_note_in, cond_in = inputs
        batch_size = tf.shape(o_note_in)[0]

        init_state = [o_note_in[:, 0, 0], K.zeros([batch_size, self.hdim])]

        _, seq, lst_state = K.rnn(self.step, cond_in, init_state)

        return seq

    def compute_output_shape(self, input_shape):
        return input_shape[:-1] + [self.odim]