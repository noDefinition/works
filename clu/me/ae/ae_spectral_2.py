from clu.me.ae.ae_spectral import *


# noinspection PyAttributeOutsideInit
class AeSpectral2(AeSpectral):
    # 试着用阈值过滤合适的sample，训练其purity
    # 试着先预训练encode输出为mean pooling，和decode的tfidf输出；几轮后训练谱聚类
    def get_hyper_params(self, args):
        self.lr: float = args[C.lr]
        self.dim_h: int = args[C.hd]
        self.c_init: int = args[C.cini]
        self.w_init: int = args[C.wini]
        self.sigma: float = args[C.sigma]
        self.c_spectral: float = args[C.cspec]
        self.c_decode: float = args[C.cdecd]
        self.c_purity: float = args[C.cpurt]
        self.pre_train_step: int = args[C.ptn]
        # self.pre_train_comb: int = args[C.ptncmb]
        # self.entropy_thr = 0

    # def define_inputs(self):
    #     super(AeSpectral2, self).define_inputs()
    #     self.ph_entropy_thr = tf.placeholder(f32, (), name='ph_entropy_thr')

    def forward(self):
        # p_stop = tf.stop_gradient(self.p_lkup, name='p_stop')  # (bs, dw)
        # p_lkup = tf.cond(self.ph_is_pretrain, true_fn=lambda: p_stop, false_fn=lambda: self.p_lkup)
        p_mean = self.get_mean_pooling(self.p_lkup, self.p_mask, name='p_mean')  # (bs, dw)
        uas = [(self.dim_w * 2, relu), (self.dim_w, None)]
        p_encode = instant_denses(p_mean, uas, name='p_encode')  # (bs, dw)
        uas = [(self.num_c, None)]
        pred_denses = build_denses(uas, name='pred_denses')
        pd_score = postpone_denses(p_encode, pred_denses, name='pd_score')  # (bs, cn)
        pd_probs = softmax(pd_score, axis=-1, name='pd_probs')  # (bs, cn)
        # pd_score = instant_denses(p_encode, uas, name='pd_score')  # (bs, cn)

        # with tf.name_scope('purity_loss'):
        #     # tf.cond(self.ph_pre, true_fn=lambda: p_stop, false_fn=lambda: self.p_lkup)
        #     entropy_thr = - np.log(1 / self.num_c) * self.ph_entropy_thr
        #     neg_p_log_p = - tf.log(pd_probs + 1e-32) * pd_probs  # (bs, cn)
        #     entropy = tf.reduce_sum(neg_p_log_p, axis=-1, keepdims=True)  # (bs, 1)
        #     entropy_mask = tf.cast(tf.less(entropy, entropy_thr), f32)  # (bs, 1)
        #     pred_mut_mask = tf.matmul(entropy_mask, entropy_mask, transpose_b=True)  # (bs, bs)
        #     purity_loss = tf.reduce_mean(entropy * entropy_mask)

        with tf.name_scope('spectral_loss'):
            p_encode_s = tf.stop_gradient(p_encode, name='p_encode_s')
            pd_score_s = postpone_denses(p_encode_s, pred_denses, name='pd_score_s')  # (bs, cn)
            pd_probs_s = softmax(pd_score_s, axis=-1, name='pd_probs_s')  # (bs, cn)
            pmut_norm = self.mutual_norm(p_encode_s, p_encode_s, name='pmut_norm')  # (bs, bs)
            radius = - 2 * (self.sigma ** 2)
            pmut_kernel = tf.exp(pmut_norm / radius)
            pd_probs_s = pd_probs
            pred_norm = self.mutual_norm(pd_probs_s, pd_probs_s, name='pred_norm')  # (bs, bs)
            # spectral_loss = tf.reduce_mean(pmut_kernel * pred_norm, name='spectral_loss')
            spectral_loss = - tf.reduce_mean(pred_norm)

        with tf.name_scope('encode_loss'):
            # mean_encode_l2 = tf.square(p_mean - p_encode)  # (bs, dw)
            # mean_encode_dist = tf.reduce_sum(mean_encode_l2, axis=-1)  # (bs, )
            mean_encode_sim = self.cos_sim(p_mean, p_encode, name='mean_encode_sim')
            encode_loss = - tf.reduce_mean(mean_encode_sim)

        with tf.name_scope('decode_loss'):
            # decode_input = p_encode
            decode_input = tf.concat([p_encode, pd_probs], axis=1)
            uas = [(self.dim_w * 2, relu), (self.num_w, None)]
            decode_tfidf = instant_denses(decode_input, uas, name='decode_tfidf')  # (bs, nw)
            # decode_dist = tf.square(self.ph_tfidf - decode_tfidf)
            # decode_loss = tf.reduce_mean(tf.sqrt(decode_dist))
            decode_cos_sim = self.cos_sim(self.ph_tfidf, decode_tfidf, name='decode_cos_sim')
            decode_loss = - tf.reduce_mean(decode_cos_sim, name='decode_loss')

        # self.encode_loss = tf.add_n([encode_loss, ], name='encode_loss')
        # self.decode_loss = tf.add_n([decode_loss, ], name='encode_loss')
        # self.ende_loss = tf.add_n([
        #     encode_loss,
        #     decode_loss * self.c_decode,
        # ], name='ende_loss')
        # self.train_loss = tf.add_n([
        #     purity_loss * self.c_purity,
        #     spectral_loss * self.c_spectral,
        # ], name='train_loss')
        self.encode_loss = encode_loss
        self.decode_loss = decode_loss
        self.train_loss = spectral_loss
        self.pred_denses = pred_denses
        self.pc_probs = pd_probs

        self.merge_mid = su.merge([
            histogram(name='pmut_norm', values=pmut_norm, family='spectral'),
            histogram(name='pmut_kernel', values=pmut_kernel, family='spectral'),
            histogram(name='pred_norm', values=pred_norm, family='spectral'),
            # histogram(name='neg_p_log_p', values=neg_p_log_p, family='purity'),
            # histogram(name='entropy', values=entropy, family='purity'),
            histogram(name='p_encode', values=p_encode, family='encode'),
            histogram(name='pd_probs', values=pd_probs, family='encode'),
            histogram(name='decode_tfidf', values=decode_tfidf, family='decode'),
            # histogram(name='decode_dist', values=decode_dist, family='decode'),
            histogram(name='decode_cos_sim', values=decode_cos_sim, family='decode'),
        ])
        self.merge_loss = su.merge([
            # scalar(name='purity_loss', tensor=purity_loss, family='loss'),
            scalar(name='spectral_loss', tensor=spectral_loss, family='loss'),
            scalar(name='decode_loss', tensor=decode_loss, family='loss'),
            scalar(name='encoen_loss', tensor=encode_loss, family='loss'),
            scalar(name='train_loss', tensor=self.train_loss, family='loss'),
        ])

    def define_optimizer(self):
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr, name='AAdam_spec')
        self.train_op = self.optimizer.minimize(self.train_loss, name='train_op')
        self.encode_op = self.optimizer.minimize(self.encode_loss, name='encode_op')
        self.decode_op = self.optimizer.minimize(self.decode_loss, name='decode_op')

    def train_step(self, docarr: List[Document], epoch_i: int, batch_i: int, **kwargs):
        if epoch_i == 0 and batch_i == 0:
            print('fffffuck pre assign')
            # ref = [v for v in tf.trainable_variables() if v.name ][0]
            ref = self.pred_denses[-1].kernel
            print(ref.name, ref.shape)
            pre_assign = tf.assign(ref, tf.transpose(self.c_embed))
            self.sess.run(pre_assign)
        fd = self.get_fd(docarr, is_train=True)
        if epoch_i <= 3:
            self.sess.run(self.encode_op, feed_dict=fd)
        self.sess.run(self.decode_op, feed_dict=fd)
        if epoch_i > self.pre_train_step:
            self.sess.run(self.train_op, feed_dict=fd)
        self.global_step += 1
        # self.entropy_thr = min(0.8, self.global_step / 5000)
        # print_on_first(epoch_i, f'  --entropy_thr = {self.entropy_thr}')
        # fd[self.ph_entropy_thr] = self.entropy_thr
        # if epoch_i < self.pre_train_step:
        #     print_on_first(epoch_i, f'  --phase 1--  pre-training on epoch {epoch_i}')
        #     fd[self.ph_is_pretrain] = True
        #     self.sess.run(self.pre_train_op, feed_dict=fd)
        # else:
        #     print_on_first(epoch_i, f'  --phase 2--  formal training on epoch {epoch_i}')

    def run_merge(self, docarr: List[Document], merge):
        fd = self.get_fd(docarr, is_train=False)
        # fd[self.ph_entropy_thr] = self.entropy_thr
        return self.sess.run(merge, feed_dict=fd)
