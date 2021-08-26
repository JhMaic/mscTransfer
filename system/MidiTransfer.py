from system.model.HAutoEncoder import *
from datProcessing.TrainSetManager import *
from system.model.HAutoEncoder import *


class MidiTransfer:
    def __init__(self,
                 load=False,
                 train=True,
                 modelSavepath=None,
                 train_data_path=None,
                 chord_path=None,
                 readall=True,
                 odim=278,
                 nMeasure=25):
        self.nM = nMeasure
        self.train = train
        self.odim = odim
        self.data_path = train_data_path  # contains 2 different paths
        self.sChord = np.load(chord_path, allow_pickle=True)
        self.savepath = modelSavepath
        self.fit(load, readall)
        if train:
            self.datInit()
            self.fit()
            self.compile()

    def datInit(self):
        cd = self.sChord
        pr_v = TransferDataset().loadnpy([self.data_path[0]])[0]
        pr_c = TransferDataset().loadnpy([self.data_path[1]])[0]

        pr_v = self.musicNpyProcess(pr_v, len(cd), clip=self.nM * 4 * 4, roll=True)
        pr_c = self.musicNpyProcess(pr_c, len(cd), clip=self.nM * 4 * 4, roll=True)
        pr_cv = tf.concat([pr_c, pr_v], 0)

        full_data_dim = np.shape(pr_v)[-1]
        type_dim = (full_data_dim - 4) // 2 - 61

        self.l_t_idx = 61  # (61: 137) -> lt
        self.r_p_idx = 61 + type_dim  # (137: 198) -> rp
        self.r_t_idx = self.r_p_idx + 61  # (198: 274) -> rt
        self.BATCH_SIZE = 32
        self.train_dataset = tf.data.Dataset.from_tensor_slices(pr_cv).shuffle(pr_cv.shape[0]).batch(
            self.BATCH_SIZE * 2,
            drop_remainder=True)

    def fit(self,
            load_weight=False,
            load_all=True,
            embedding=None,
            noteInterpreter=None):

        Embedding = NoteEmbedding(name='Embedding', trainable=False) \
            if not embedding else embedding
        Interpreter = NoteInterpreter(name='Interpreter', trainable=True) \
            if not noteInterpreter else noteInterpreter

        model = HAutoEncoder(self.odim, 120, self.nM, Embedding, Interpreter,
                             savepath=self.savepath, trainable=True, cross_weight=True)
        mTrain, mPred = model.CreateModel(1, False)
        if load_weight: model.load_weights(readall=load_all)
        self.model = mTrain if self.train else mPred

    def compile(self,
                sys_optm=None):
        if not self.train:
            raise ConnectionError('prediction does not need this function')

        self.sys_optm = tf.keras.optimizers.Adam(learning_rate=0.0001, beta_1=0.3, beta_2=0.9) \
            if not sys_optm else sys_optm
        self.checkpoint = tf.train.Checkpoint(sys_optm=sys_optm, )

        # graph writer
        log_dir = 'logs/gradient_tape/' + '201113_corss_weights'
        self.rec_summary_writer = tf.summary.create_file_writer(log_dir + '/reconstruction')
        self.dis_summary_writer = tf.summary.create_file_writer(log_dir + '/discriminator')

        self.acc_lp_summ_writer = tf.summary.create_file_writer(log_dir + '/left_pitch_acc')
        self.acc_rp_summ_writer = tf.summary.create_file_writer(log_dir + '/right_pitch_acc')
        self.acc_state_summ_writer = tf.summary.create_file_writer(log_dir + '/state_acc')

        self.rec_loss = tf.metrics.Mean('rec_loss', 'float32')
        self.dis_loss = tf.metrics.Mean('dis_loss', 'float32')
        self.acc_lp = tf.metrics.CategoricalAccuracy()
        self.acc_rp = tf.metrics.CategoricalAccuracy()
        self.acc_state = tf.metrics.CategoricalAccuracy()

    def train(self):
        mvLoss = lambda x, y: tf.losses.MSE(K.mean(x, 1), K.mean(y, 1)) + \
                              tf.losses.MSE(K.sqrt(K.var(x, 1)), K.sqrt(K.var(y, 1)))
        rcloss = lambda x, y: tf.losses.categorical_crossentropy(x[..., 0:61], y[..., 0:61]) + \
                              tf.losses.categorical_crossentropy(x[..., 61:137], y[..., 61:137]) + \
                              tf.losses.categorical_crossentropy(x[..., 137:198], y[..., 137:198]) + \
                              tf.losses.categorical_crossentropy(x[..., 198:274], y[..., 198:274]) + \
                              tf.losses.categorical_crossentropy(x[..., 274:], y[..., 274:])

        for epoch in range(999999):
            # [0,1] for classical, [1,0] for v
            for step, data in enumerate(self.train_dataset):
                data_c, data_v = tf.split(data, 2, 0)
                data_in_c = self.get_decoder_input_data(data_c)
                data_in_v = self.get_decoder_input_data(data_v)

                with tf.GradientTape() as aex_tape:
                    ab_rec, ori_b_atte, gen_b_atte, \
                    ori_m_atte, gen_m_atte = self.model([data_c, data_v,
                                                         data_in_c, data_in_v], training=True)

                    total_data = tf.concat([data_c, data_v], axis=0)
                    D_loss = mvLoss(ori_m_atte, gen_m_atte) + mvLoss(ori_b_atte, gen_b_atte)
                    loss_rec = rcloss(ab_rec, total_data)
                    gen_losses = loss_rec + D_loss[..., tf.newaxis]

                self.rec_loss(loss_rec)
                # dis_loss(b_disc_loss + m_disc_loss)
                self.dis_loss(D_loss)

                grad_aex = aex_tape.gradient(gen_losses, self.model.trainable_variables)
                self.sys_optm.apply_gradients(zip(grad_aex, self.model.trainable_variables))

                self.acc_lp(ab_rec[..., :self.l_t_idx], total_data[..., :self.l_t_idx])
                self.acc_rp(ab_rec[..., self.r_p_idx:self.r_t_idx], total_data[..., self.r_p_idx:self.r_t_idx])
                self.acc_state(ab_rec[..., -4:], total_data[..., -4:])

                with self.rec_summary_writer.as_default():
                    tf.summary.scalar('loss', self.rec_loss.result(), step=epoch)
                with self.dis_summary_writer.as_default():
                    tf.summary.scalar('loss', self.dis_loss.result(), step=epoch)
                with self.acc_lp_summ_writer.as_default():
                    tf.summary.scalar('accuracy', self.acc_lp.result(), step=epoch)
                with self.acc_rp_summ_writer.as_default():
                    tf.summary.scalar('accuracy', self.acc_rp.result(), step=epoch)
                with self.acc_state_summ_writer.as_default():
                    tf.summary.scalar('accuracy', self.acc_state.result(), step=epoch)

                print('EPOCH:{}, STEP:{}, training_loss = {}\t\t\t, dis_loss = {} '.format(epoch, step,
                                                                                           self.rec_loss.result(),
                                                                                           self.dis_loss.result()))

            self.model.save_weights()
            self.checkpoint.save(self.savepath)

    def predict(self, a, b, fmt='tensor'):
        assert not self.train
        assert fmt in ['tensor', 'mscPath']
        if fmt != 'tensor':
            a_token = XmlDatParse(a, self.sChord,readonly=True).convert(combine=True)
            b_token = XmlDatParse(b, self.sChord,readonly=True).convert(combine=True)
            a = self.musicNpyProcess(a_token, len(self.sChord), clip=self.nM * 4 * 4, roll=False)[:1]
            b = self.musicNpyProcess(b_token, len(self.sChord), clip=self.nM * 4 * 4, roll=False)[:1]

        a_b, b_a = self.model([a,
                               b,
                               self.get_decoder_input_data(a),
                               self.get_decoder_input_data(b)], training=False)
        return a_b.numpy(), b_a.numpy() # tensor

    def get_decoder_input_data(self, x):
        # input is musical token
        p = tf.concat(
            [tf.one_hot([61], 61), tf.one_hot([76], 76), tf.one_hot([61], 61), tf.one_hot([76], 76),
             tf.one_hot([1], 4)],
            axis=-1)
        p = tf.expand_dims(p, 1)
        p = tf.repeat(p, tf.shape(x)[0], axis=0)
        a = K.concatenate([p, x], axis=1)

        return a[:, :-1]

    def get_info(self, x):
        # input is musical tensor for training
        lp = tf.argmax(x[..., 0:61], axis=-1)
        lt = tf.argmax(x[..., 61:137], axis=-1)
        rp = tf.argmax(x[..., 137:198], axis=-1)
        rt = tf.argmax(x[..., 198:274], axis=-1)
        state = tf.argmax(x[..., 274:], axis=-1)

        lp = tf.add(lp, 35)
        lp = tf.where(lp == 35, -1, lp)
        rp = tf.add(rp, 35)
        rp = tf.where(rp == 35, -1, rp)

        return state, lp, rp, lt, rt

    def musicNpyProcess(self, pr_np, ntype, clip=None, roll=False):
        # TOKENS TO TENSOR
        # shape = (2,time,3)
        # pr = tf.reshape(pr_np,[tf.shape(pr_np)[1], 6])
        # ASSUME RANGE IS (36,96) -> 60 notes (+1 rest note)
        def state_convert(pr_np):
            '''
            [L,R] onset(1) or keep(0)
            [0,0]-> 0
            [1,1]-> 1
            [1,0]-> 2
            [0,1]-> 3
            '''
            l, r = pr_np[0], pr_np[1]
            l_s = l[:, 2:]
            r_s = r[:, 2:]
            lr_s = tf.concat([l_s, r_s], axis=-1)
            new_state = tf.zeros_like(l)
            new_state = K.sum(new_state, axis=1)

            cond_1 = tf.constant([[0, 0, ]])
            cond_2 = tf.constant([[1, 1, ]])
            cond_3 = tf.constant([[1, 0, ]])
            cond_4 = tf.constant([[0, 1, ]])

            cond_1 = tf.cast(tf.equal(lr_s, cond_1), tf.int8)
            cond_1 = K.sum(cond_1, axis=1)
            new_state = tf.where(cond_1 == 2, 0, new_state)

            cond_2 = tf.cast(tf.equal(lr_s, cond_2), tf.int8)
            cond_2 = K.sum(cond_2, axis=1)
            new_state = tf.where(cond_2 == 2, 1, new_state)

            cond_3 = tf.cast(tf.equal(lr_s, cond_3), tf.int8)
            cond_3 = K.sum(cond_3, axis=1)
            new_state = tf.where(cond_3 == 2, 2, new_state)

            cond_4 = tf.cast(tf.equal(lr_s, cond_4), tf.int8)
            cond_4 = K.sum(cond_4, axis=1)
            new_state = tf.where(cond_4 == 2, 3, new_state)

            return new_state

        def Clip(tpr):
            times = np.shape(tpr)[1] // clip
            if times == 1:
                return tpr
            tpr = tpr[:, :times * clip]
            tpr = tf.concat(tf.split(tpr, times, axis=1), axis=0)
            return tpr

        new_state = state_convert(pr_np)

        l, r = pr_np[0], pr_np[1]
        l_p = l[:, 0]
        r_p = r[:, 0]
        l_p = tf.where(l_p == -1, 35, l_p) - 35  # 0 - 60, where 0 is rest
        r_p = tf.where(r_p == -1, 35, r_p) - 35

        l_p = tf.one_hot(l_p, 61)
        r_p = tf.one_hot(r_p, 61)
        l_t = tf.one_hot(l[:, 1], ntype)
        r_t = tf.one_hot(r[:, 1], ntype)
        lr_s = tf.one_hot(new_state, 4)

        l = tf.concat([l_p, l_t], axis=-1)
        r = tf.concat([r_p, r_t], axis=-1)

        pr = tf.concat([l, r, lr_s], axis=-1)
        pr = tf.cast(pr, 'float32')

        prs = []
        if clip:
            pr = tf.expand_dims(pr, 0)
            if roll:
                for _ in range(4):
                    prs += [Clip(pr)]
                    pr = tf.roll(pr, 32, axis=1)  # 32 = 2measures
                pr = tf.concat(prs, axis=0)
            else:
                pr = Clip(pr)
        return pr
