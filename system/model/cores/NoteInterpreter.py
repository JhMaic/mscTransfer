from system.model.cores.nn import *


class NoteInterpreter(Model):
    def __init__(self,
                 pitch_dim=61,  # (36,96)+rest
                 type_dim=76,
                 state_dim=4,
                 gamma=0.001,
                 **kwargs):
        super(NoteInterpreter, self).__init__(**kwargs)
        self.pitch_dim = pitch_dim
        self.type_dim = type_dim
        self.state_dim = state_dim
        self.total_dim = (pitch_dim + type_dim) * 2 + state_dim
        self.gamma = gamma

        self.l1 = Linear(self.total_dim, activation=LeakyReLU(0.2), name='liner_1',
                         kernel_initializer='glorot_normal',
                         trainable=self.trainable)
        self.l_state = Linear(self.state_dim, name='state_exc',
                              kernel_initializer='glorot_normal',
                              trainable=self.trainable)
        self.l_lp = Linear(self.pitch_dim, name='lp_exc',
                           kernel_initializer='glorot_normal',
                           trainable=self.trainable)
        self.l_lt = Linear(self.type_dim, name='lt_exc',
                           kernel_initializer='glorot_normal',
                           trainable=self.trainable)
        self.l_rp = Linear(self.pitch_dim, name='rp_exc',
                           kernel_initializer='glorot_normal',
                           trainable=self.trainable)
        self.l_rt = Linear(self.type_dim, name='rt_exc',
                           kernel_initializer='glorot_normal',
                           trainable=self.trainable)

    def cal(self, x, training):
        x = self.l1(x)
        x = Dropout(0.5)(x, training=training)

        state = self.l_state(x)

        lp = self.l_lp(x)
        lt = self.l_lt(x)

        rp = self.l_rp(x)
        rt = self.l_rt(x)

        return [lp, lt, rp, rt, state]

    def call(self, inputs, force=False, training=None, mask=None):
        l_p, l_t, r_p, r_t, state = self.cal(inputs, training=training)

        if force:
            l_p /= self.gamma
            l_t /= self.gamma
            r_p /= self.gamma
            r_t /= self.gamma
            state /= self.gamma

        l_p = K.softmax(l_p)
        l_t = K.softmax(l_t)
        r_p = K.softmax(r_p)
        r_t = K.softmax(r_t)
        state = K.softmax(state)

        return tf.concat([l_p, l_t, r_p, r_t, state], axis=-1)

    def compute_output_shape(self, input_shape):
        return input_shape[:-1] + [self.total_dim]