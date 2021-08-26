from system.model.cores.nn import *
from system.model.cores.NoteInterpreter import NoteInterpreter
from system.model.cores.EncoderQuery import EncoderQuery,EncoderQueryX
from system.model.cores.NoteEmbedding import NoteEmbedding
import os
import numpy as np

class HAutoEncoder():
    # version 2
    '''
        change from 1:
        use a unique decoder,
        rnn renew the state per 4 steps
    '''

    def __init__(self,
                 odim,
                 zdim,
                 nMeasure,
                 Embedder=None,
                 Interp=None,
                 savepath=None,
                 trainable=True,
                 cross_weight=True
                 ):
        self.zdim = zdim  # embedded dim
        self.odim = odim
        self.hdim = 200  # beat, measure dim
        self.nM = nMeasure
        self.embedder = Embedder if Embedder else NoteEmbedding(name='Embedding',trainable=False)
        self.inerp = Interp if Interp else NoteInterpreter(name='Interpreter',trainable=trainable)
        self.encoder = self.Encoder()
        self.d_gru = GRUCell(odim, name='gru_cell', trainable=trainable)
        self.encoderQuery = EncoderQuery(name='d_encoder_queries') if not cross_weight else \
            EncoderQueryX(name='d_encoder_queries')

        self.savepath = savepath if savepath else None

    def Encoder(self, name='Encoder', trainable=True):
        # extract overall msg without time-relations

        uw_reshape = Reshape([-1, 4, self.hdim], name='upward_reshape')
        # if use linear instead of conv1d ?
        n_conv1d = Conv1D(self.hdim, 1, kernel_initializer='glorot_normal', name='e_note_conv1d', trainable=trainable)
        b_conv1d = Conv1D(self.hdim, 1, kernel_initializer='glorot_normal', name='e_beat_conv1d', trainable=trainable)
        n_linear = Linear(self.hdim,
                          kernel_initializer='glorot_normal',
                          activation=LeakyReLU(0.2), name='note_linear',
                          trainable=trainable)
        b_linear_1 = Linear(self.hdim,
                            kernel_initializer='glorot_normal',
                            activation=LeakyReLU(0.2), name='e_beat_linear_1',
                            trainable=trainable)
        b_linear_2 = Linear(self.hdim,
                            kernel_initializer='glorot_normal',
                            activation=LeakyReLU(0.2), name='e_beat_linear_2',
                            trainable=trainable)
        b2k_linear = Linear(self.hdim,
                            kernel_initializer='glorot_normal',
                            use_bias=False,
                            name='e_beat2key_linear')
        b2q_linear = Linear(self.hdim,
                            kernel_initializer='glorot_normal',
                            use_bias=False,
                            name='e_beat2query_linear',
                            trainable=trainable)

        m_linear_1 = Linear(self.hdim,
                            kernel_initializer='glorot_normal',
                            activation=LeakyReLU(0.2), name='e_measure_linear_1',
                            trainable=trainable)
        m_linear_2 = Linear(self.hdim,
                            kernel_initializer='glorot_normal',
                            activation=LeakyReLU(0.2), name='e_measure_linear_2',
                            trainable=trainable)
        m2k_linear = Linear(self.hdim,
                            kernel_initializer='glorot_normal',
                            use_bias=False,
                            name='e_measure2key_linear')
        m2q_linear = Linear(self.hdim,
                            kernel_initializer='glorot_normal',
                            use_bias=False,
                            name='e_measure2query_linear',
                            trainable=trainable)

        note_seq = Input([None, self.zdim])

        x = n_linear(note_seq)
        x = uw_reshape(x)
        x = TimeDistributed(n_conv1d)(x)
        x = TimeDistributed(Flatten())(x)
        x = b_linear_1(x)
        x = LayerNormalization()(x)
        x = Dropout(0.3)(x)
        beats = b_linear_2(x)

        x = uw_reshape(beats)
        x = TimeDistributed(b_conv1d)(x)
        x = TimeDistributed(Flatten())(x)
        x = m_linear_1(x)
        x = LayerNormalization()(x)
        x = Dropout(0.3)(x)
        measures = m_linear_2(x)

        b_k = b2k_linear(beats)
        b_q = b2q_linear(beats)
        m_k = m2k_linear(measures)
        m_q = m2q_linear(measures)

        return Model([note_seq], [beats, b_q, b_k, measures, m_q, m_k], name=name)

    def Decoder(self, d_notes, beats, b_q, b_k, measures, m_q, m_k, cross=False):
        last_concat = Concatenate(-1)

        b_a, m_a = self.encoderQuery([beats, b_q, b_k, measures, m_q, m_k], cross=cross)
        r_m_a = tf.repeat(m_a, 16, axis=1)
        r_b_a = tf.repeat(b_a, 4, axis=1)

        x = last_concat([d_notes, r_b_a, r_m_a])
        x = Reshape([-1, 4, self.hdim * 2 + self.zdim])(x)

        return x, b_a, m_a

    def CreateModel(self, alpha, newstate=False):
        a_in = Input([self.nM * 16, self.odim])
        b_in = Input([self.nM * 16, self.odim])

        batch_concat = Concatenate(0, name='batch_concat')
        batch_split = Lambda(lambda x: tf.split(x, 2, 0),
                             name='batch_split')

        a_e_in = self.embedder(a_in)
        b_e_in = self.embedder(b_in)

        a_e_ret = self.encoder(a_e_in)
        b_e_ret = self.encoder(b_e_in)

        # decoder
        a_d_in = Input([self.nM * 16, self.odim])
        b_d_in = Input([self.nM * 16, self.odim])

        ea_d_in = self.embedder(a_d_in)
        eb_d_in = self.embedder(b_d_in)

        a_feed, a_b_atte, a_m_atte = self.Decoder(ea_d_in, *a_e_ret)
        b_feed, b_b_atte, b_m_atte = self.Decoder(eb_d_in, *b_e_ret)

        x = batch_concat([a_feed, b_feed])

        if newstate:
            ab_rec = TimeDistributed(RNN(self.d_gru, return_sequences=True))(x)
            ab_rec = Reshape([-1, self.odim])(ab_rec)
            ab_rec = self.inerp(ab_rec)
        else:
            ab_rec = Reshape([-1, 520])(x)
            ab_rec = RNN(self.d_gru, return_sequences=True)(ab_rec)
            ab_rec = self.inerp(ab_rec)

        ##
        # a2b, b2a
        a_beats, a_b_q, a_b_k, a_measures, a_m_q, a_m_k = a_e_ret
        b_beats, b_b_q, b_b_k, b_measures, b_m_q, b_m_k = b_e_ret

        # a2b ==> notes of a + structure of b
        axb_feed, a2b_b_atte, a2b_m_atte = self.Decoder(ea_d_in,
                                                         a_beats, b_b_q * alpha + (1 - alpha) * a_b_q, a_b_k,
                                                         a_measures, b_m_q * alpha + (1 - alpha) * a_m_q, a_m_k,
                                                        cross=True)

        # b2a ==> notes of b + structure of a
        bxa_feed, b2a_b_atte, b2a_m_atte = self.Decoder(eb_d_in,
                                                         b_beats, a_b_q * alpha + (1 - alpha) * b_b_q, b_b_k,
                                                         b_measures, a_m_q * alpha + (1 - alpha) * b_m_q, b_m_k,
                                                        cross=True)

        ori_b_atte = batch_concat([a_b_atte, b_b_atte])
        ori_m_atte = batch_concat([a_m_atte, b_m_atte])
        gen_b_atte = batch_concat([a2b_b_atte, b2a_b_atte])
        gen_m_atte = batch_concat([a2b_m_atte, b2a_m_atte])

        f_model = Model([a_in, b_in, a_d_in, b_d_in],
                        [ab_rec] +
                        [ori_b_atte, gen_b_atte] +
                        [ori_m_atte, gen_m_atte])

        ## self-feeding
        ro_rnn = RNNReadout(self.d_gru, self.embedder, self.inerp, newstate=newstate)

        x = batch_concat([axb_feed, bxa_feed])

        ab_cond = x[..., self.zdim:]
        ab_o_in = batch_concat([a_d_in, b_d_in])
        ab_o_in = Reshape([-1, 4, self.odim])(ab_o_in)

        ab_gen = ro_rnn([ab_o_in, ab_cond])
        ab_gen = Reshape([-1, self.odim])(ab_gen)

        a_gen, b_gen = batch_split(ab_gen)

        # rec_model = Model([a_in, b_in, a_d_in, b_d_in], [ab_rec])
        gen_model = Model([a_in, b_in, a_d_in, b_d_in], [a_gen, b_gen])

        train_model = Model([a_in, b_in, a_d_in, b_d_in],
                            [ab_rec] + [ab_gen])

        return f_model, gen_model

    def save_weights(self):
        os.makedirs(self.savepath + '\\Layers_npy', exist_ok=True)
        self.embedder.save_weights(self.savepath + '\\embedder')
        self.inerp.save_weights(self.savepath + '\\interpreter')
        self.encoder.save_weights(self.savepath + '\\encoder')
        self.encoderQuery.save_weights(self.savepath + '\\decoders\\decoder')
        np.save(self.savepath + '\\Layers_npy\\d_gru', self.d_gru.get_weights())

    def load_weights(self, path=None, readall=False):
        path = path if path else self.savepath
        self.embedder.load_weights(path + '\\embedder')
        if readall:
            self.inerp.load_weights(path + '\\interpreter')
            self.encoder.load_weights(path + '\\encoder')
            self.encoderQuery.load_weights(path + '\\decoders\\decoder')
            self.d_gru.set_weights(np.load(path + '\\Layers_npy\\d_gru.npy', allow_pickle=True))