from system.model.cores.nn import *


class EncoderQuery(Model):
    def __init__(self,
                 **kwargs):
        super(EncoderQuery, self).__init__(**kwargs)

    def build(self, input_shape):
        # (bs, time, dim)
        _, bq, _, _, mq, _ = input_shape
        hdim = bq[-1]
        self.scale = tf.sqrt(float(hdim))

        def getmodel(q_in, name):
            x = Linear(hdim, kernel_initializer='random_normal',
                       activation=LeakyReLU(0.2),
                       name='q_l_1')(q_in)
            x = Linear(hdim, kernel_initializer='random_normal',
                       activation='tanh',
                       name='q_l_2')(x)
            return Model(q_in, x, name=name, trainable=self.trainable)

        self.bq_model = getmodel(K.placeholder(bq), 'b2q_model')
        self.mq_model = getmodel(K.placeholder(mq), 'm2q_model')

        self.built = True

    def call(self, inputs, cross=False, **kwargs):
        # beats = (bs, time, dim)
        # a, b
        # a2b ==> notes of a + structure of b (1
        # b2a (2
        beats, b_q, b_k, measures, m_q, m_k = inputs

        bq = self.bq_model(b_q)
        mq = self.mq_model(m_q)

        b_atte = self.attention(bq, b_k, beats)
        m_atte = self.attention(mq, m_k, measures)

        return b_atte, m_atte

    def attention(self, q, k, v):
        score = tf.matmul(q, k, transpose_b=True) / self.scale

        score = tf.nn.softmax(score, axis=-1)
        if type(score) is not tf.Tensor:
            x = score.numpy().squeeze().T

        return tf.matmul(score, v)

    def compute_output_shape(self, input_shape):
        return input_shape
class EncoderQueryX(Model):
    # 只用作比较 Reconstruction only 用
    def __init__(self,
                 **kwargs):
        super(EncoderQueryX, self).__init__(**kwargs)
        self.tmp = []

    def build(self, input_shape):
        # (bs, time, dim)
        _, bq, _, _, mq, _ = input_shape
        hdim = bq[-1]
        self.scale = tf.sqrt(float(hdim))
        self.b_W_cross = self.add_weight('beat_cross_W', [hdim, hdim],
                                         initializer='random_normal')
        self.m_W_cross = self.add_weight('measure_cross_W', [hdim, hdim],
                                         initializer='random_normal')

        def getmodel(q_in, name):
            x = Linear(hdim, kernel_initializer='random_normal',
                       activation=LeakyReLU(0.2),
                       name='q_l_1')(q_in)
            x = Linear(hdim, kernel_initializer='random_normal',
                       activation='tanh',
                       name='q_l_2')(x)
            return Model(q_in, x, name=name, trainable=self.trainable)

        self.bq_model = getmodel(K.placeholder(bq), 'b2q_model')
        self.mq_model = getmodel(K.placeholder(mq), 'm2q_model')

        self.built = True

    def call(self, inputs, cross=False, **kwargs):
        # beats = (bs, time, dim)
        beats, b_q, b_k, measures, m_q, m_k = inputs

        bq = self.bq_model(b_q)
        mq = self.mq_model(m_q)

        if cross:
            # additional weighted mapping for stylization
            bq = K.dot(bq, self.b_W_cross)
            mq = K.dot(mq, self.m_W_cross)

        b_atte = self.attention(bq, b_k, beats, cross)
        m_atte = self.attention(mq, m_k, measures, cross)

        return b_atte, m_atte

    def attention(self, q, k, v, cross):
        score = tf.matmul(q, k, transpose_b=True) / self.scale

        score = tf.nn.softmax(score, axis=-1)
        if type(score) is not tf.Tensor:
            if len(self.tmp) >= 0:
                # a_k = tf.squeeze(self.tmp[0][1])
                # a_k = tf.transpose(a_k)
                # a_k = tf.linalg.inv(a_k)

                #xx = tf.matmul(a_k, k, transpose_b=True)

                xx = score.numpy().squeeze()
                if len(xx) > 40:
                    xx = xx[:44, :44]
                else:
                    xx = xx[:11, :11]

            else:self.tmp += [[q,k]]



        return tf.matmul(score, v)

    def compute_output_shape(self, input_shape):
        return input_shape