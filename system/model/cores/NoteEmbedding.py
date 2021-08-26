from system.model.cores.nn import *

class NoteEmbedding(Model):
    def __init__(self,
                 pitch_dim=61,  # (36,96)+rest
                 pitch_e_dim=30,
                 type_e_dim=30,
                 **kwargs):
        super(NoteEmbedding, self).__init__(**kwargs)
        self.pitch_dim = pitch_dim
        # self.embedded_dim = embedded_dim
        self.ped = pitch_e_dim
        self.ted = type_e_dim

    def build(self, input_shape):
        # input_shape = [None, dim]
        # dim = [l_p, l_t, l_s, r_p, r_t, r_s] (concat)
        input_dim = input_shape[-1]
        single_note_dim = (input_dim - 4) // 2
        type_dim = single_note_dim - self.pitch_dim
        state_dim = 4
        self.Wp = self.add_weight(name='W_pitch',
                                  shape=[self.pitch_dim, self.ped],
                                  initializer='glorot_normal',
                                  trainable=self.trainable)
        self.Wt = self.add_weight(name='W_type',
                                  shape=[type_dim, self.ted],
                                  initializer='glorot_normal',
                                  trainable=self.trainable)
        self.Wpt = self.add_weight(name='W_pitch_type',
                                   shape=[self.ped + self.ted, self.ped + self.ted],
                                   initializer='random_normal',
                                   trainable=self.trainable)
        self.Wlr = self.add_weight(name='W_l_r',
                                   shape=[(self.ped + self.ted) * 2 + state_dim, (self.ped + self.ted) * 2],
                                   initializer='random_normal',
                                   trainable=self.trainable)

    def call(self, inputs, **kwargs):
        # inputs is with dim of double notes
        lr, lr_state = inputs[..., :-4], inputs[..., -4:]
        l, r = tf.split(lr, 2, axis=-1)

        l_p, l_t = l[..., :self.pitch_dim], l[..., self.pitch_dim:]
        r_p, r_t = r[..., :self.pitch_dim], r[..., self.pitch_dim:]
        l_ep = K.dot(l_p, self.Wp)
        r_ep = K.dot(r_p, self.Wp)
        l_et = K.dot(l_t, self.Wt)
        r_et = K.dot(r_t, self.Wt)

        l_tp = K.dot(tf.concat([l_ep, l_et], axis=-1), self.Wpt)
        r_tp = K.dot(tf.concat([r_ep, r_et], axis=-1), self.Wpt)

        lr = K.dot(tf.concat([l_tp, r_tp, lr_state], axis=-1), self.Wlr)
        return lr

    def compute_output_shape(self, input_shape):
        return [input_shape[:-1]] + [(self.ped + self.ted) * 2]