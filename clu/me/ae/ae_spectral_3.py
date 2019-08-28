from clu.me.ae.ae_base import *


# noinspection PyAttributeOutsideInit
class AeSpectral3(AEBase):
    def get_hyper_params(self, args: CluArgs):
        super(AeSpectral3, self).get_hyper_params(args)
        self.alpha: float = args.alpha
        self.beta: float = args.beta
        self.gamma: float = args.gamma
        self.pretrain_step: int = args.ptn

    def define_inputs(self):
        super(AeSpectral3, self).define_inputs()
        self.wid_hot = tf.one_hot(self.p_wids, depth=self.num_w, name='wid_hot')
        self.p_tf = tf.reduce_sum(self.wid_hot, axis=1, keepdims=False, name='p_tf')  # (bs, tn, nw)

    @staticmethod
    def mutual_norm(a, b, name: str):
        assert len(a.shape) == len(b.shape) == 2
        with tf.name_scope(name):
            a = tf.expand_dims(a, axis=1)  # (bs, 1, dw)
            b = tf.expand_dims(b, axis=0)  # (1, bs, dw)
            ab_sq = tf.square(a - b)  # (bs, bs, dw)
            ab_sum = tf.reduce_sum(ab_sq, axis=-1)  # (bs, bs)
            # ab_norm = tf.sqrt(ab_sum + 1e-12)  # (bs, bs)
            ab_norm = ab_sum
        return ab_norm

    @staticmethod
    def mutual_cos_sim(a, b, name: str):
        with tf.name_scope(name):
            axis = -1
            a_n = tf.nn.l2_normalize(a, axis=axis)  # (bs, dw)
            b_n = tf.nn.l2_normalize(b, axis=axis)  # (bs, dw)
            a_n_e = tf.expand_dims(a_n, axis=1)  # (bs, 1, dw)
            b_n_e = tf.expand_dims(b_n, axis=0)  # (1, bs, dw)
            dot = tf.reduce_sum(a_n_e, b_n_e, axis=axis)  # (bs, bs)
        return dot

    def forward(self):
        with tf.name_scope('encoder'):
            # p_tf = tf.one_hot(self.p_wids, self.num_w, name='p_tf')
            p_mean = self.get_mean_pooling(self.p_lkup, self.p_mask, name='p_mean')  # (bs, dw)
            pc_score = tf.matmul(p_mean, self.c_embed, transpose_b=True)  # (bs, cn)
            pc_probs = tf.nn.softmax(pc_score, axis=-1, name='pc_prods')  # (bs, cn)

        # with tf.name_scope('spectral_loss'):
        #     pmut_norm = self.mutual_norm(p_mean, p_mean, name='pmut_norm')  # (bs, bs)
        #     radius = - 2 * (self.sigma ** 2)
        #     pmut_kernel = tf.exp(pmut_norm / radius)
        #     pred_dist = 1 - self.mutual_cos_sim(pc_probs, pc_probs, name='pred_dist')
        #     # pred_dist = self.mutual_norm(pc_probs, pc_probs, name='pred_dist')  # (bs, bs)
        #     spectral_loss = tf.reduce_mean(pmut_kernel * pred_dist, name='spectral_loss')

        with tf.name_scope('vector_quantize'):
            z = p_mean
            z_i = tf.argmax(pc_probs, axis=1)  # (bs, )
            z_p = tf.nn.embedding_lookup(self.c_embed, z_i)  # (bs, dw)
            # z_e = tf.stop_gradient(z_p + z) - z  # (bs, dw)
            z_e = self.gamma * z_p + (1 - self.gamma) * z  # (bs, dw)
            zzp_cos = self.cos_sim(z, z_p, name='zzp_cos')
            zzp_loss = - tf.reduce_mean(zzp_cos)
            # zzp_cos_1 = self.cos_sim(tf.stop_gradient(z), z_p, name='zzp_cos_1')  # (bs, )
            # zpz_cos_2 = self.cos_sim(tf.stop_gradient(z_p), z, name='zpz_cos_2')  # (bs, )
            # zzp_loss_1 = - tf.reduce_mean(zzp_cos_1)
            # zpz_loss_2 = - tf.reduce_mean(zpz_cos_2)

        with tf.name_scope('decoder'):
            # uas = [(self.num_w, None)]
            uas = [(self.dim_w * 2, tanh), (self.num_w, None)]
            decode_dense = build_denses(uas, 'decode_dense')
            decode_score = postpone_denses(z_e, decode_dense, name='decode_score')  # (bs, nw)
            decode_tf = tf.nn.sigmoid(decode_score)  # (bs, nw)
            # decode_score = instant_denses(z_e, uas, name='decode_score')  # (bs, nw)
            # decode_ce = tf.log(tf.maximum(decode_tf, 1e-32)) * self.p_tf  # (bs, nw)
            # decode_loss = - tf.reduce_mean(tf.reduce_sum(decode_ce, axis=1))
            decode_cos_sim = self.cos_sim(self.p_tf, decode_tf, name='decode_cos_sim')
            decode_loss = - tf.reduce_mean(decode_cos_sim, name='decode_loss')

        with tf.name_scope('decoder_pretrain'):
            p_lkup = tf.stop_gradient(self.p_lkup)
            pred_log = postpone_denses(p_lkup, decode_dense, name='pred_log')  # (bs, tn, nw)
            hot_tf_cos = self.cos_sim(pred_log, self.wid_hot, name='hot_tf_cos')  # (bs, tn)
            hot_tf_loss = - tf.reduce_mean(hot_tf_cos)
            # hot_tf_sim = pred_log * self.p_tf  # (bs, tn)
            # hot_tf_loss = - tf.reduce_mean(hot_tf_sim)

            self.pre_loss = hot_tf_loss
            self.train_loss = tf.add_n([
                self.alpha * zzp_loss if self.alpha > 1e-6 else 0.,
                self.beta * decode_loss if self.beta > 1e-6 else 0.
            ], name='train_loss')

            self.pc_probs = pc_probs
            self.merge_some = su.merge([
                histogram(name='pc_probs', values=pc_probs, family='encode'),
                histogram(name='z_i', values=z_i, family='vq'),
                histogram(name='zzp_cos', values=zzp_cos, family='vq'),
                histogram(name='decode_tf', values=decode_tf, family='decode'),
                histogram(name='decode_score', values=decode_score, family='decode'),
            ])
            self.merge_batch = su.merge([
                scalar(name='hot_tf_loss', tensor=hot_tf_loss, family='pre'),
                scalar(name='zzp_loss', tensor=zzp_loss, family='loss'),
                scalar(name='decode_loss', tensor=decode_loss, family='loss'),
                scalar(name='train_loss', tensor=self.train_loss, family='loss'),
            ])

    def define_optimizer(self):
        pre_oper = tf.train.AdamOptimizer(learning_rate=self.lr, name='ADAM_pre')
        self.pre_op = pre_oper.minimize(self.pre_loss, name='pre_op')
        train_oper = tf.train.AdamOptimizer(learning_rate=self.lr, name='AAdam_train')
        self.train_op = train_oper.minimize(self.train_loss, name='train_op')

    def train_step(self, docarr: List[Document], epoch_i: int, batch_i: int, **kwargs):
        fd = self.get_fd(docarr, is_train=True)
        if epoch_i < self.pretrain_step:
            self.sess.run(self.pre_op, feed_dict=fd)
        else:
            self.sess.run(self.train_op, feed_dict=fd)
        self.global_step += 1
