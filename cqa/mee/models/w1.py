from .v1 import *


# noinspection PyAttributeOutsideInit
class W1(V1):
    """ 给回答表示叠加 **生成噪声** """

    def __init__(self, args):
        super(W1, self).__init__(args)
        self.eps = args[K.eps]

    def forward(self):
        a_emb = self.embed_rep(self.a_lkup, self.a_mask, 'a_emb', k=10)
        au_cat = tf.concat([a_emb, self.u_lkup], axis=-1, name='au_cat')
        uas = [(self.dim_w, relu), (self.dim_w, None)]
        au_ffn = instant_denses(au_cat, uas, name='au_ffn', w_reg=self.l2_reg)

        if self.eps > 1e-9:
            # Q = instant_denses(au_cat, [(self.dim_w, relu)], name='Q', w_reg=self.l2_reg)
            # K = instant_denses(au_cat, [(self.dim_w, relu)], name='K', w_reg=self.l2_reg)
            Q = K = au_ffn
            QK = tf.matmul(Q, K, transpose_b=True, name='QK')
            # QK = tf.divide(QK, tf.sqrt(tf.cast(Q.shape[-1], f32)), name='QK_scale')
            QK = mask_diagly(QK, name='QK_mask_diagly')
            QK_attn = tf.nn.softmax(QK, axis=1, name='QK_attn')
            au_res = tf.matmul(QK_attn, au_cat, name='au_res')
            generator = [
                # tf.layers.Dense(u, a, kernel_regularizer=self.l2_reg, name='gen_{}'.format(i))
                tf.layers.Dense(u, a, name='gen_{}'.format(i))
                for i, (u, a) in enumerate([(self.dim_w, relu), (self.dim_w, None)])
            ]
            # q_emb = self.embed_rep(self.q_lkup, self.q_mask, 'q_emb')
            # noise = postpone_denses(generator, q_emb, name='au_noise')
            noise = postpone_denses(generator, au_res, name='au_noise')
            noise = self.eps * noise
            au_final = tf.add(au_ffn, noise, name='au_final')
        else:
            au_final = au_ffn
        uas = [(self.dim_w, relu), (1, None)]
        preds = instant_denses(au_final, uas, name='preds', w_reg=self.l2_reg)
        self.pred_probs = self.pred_scores = tf.squeeze(preds, axis=1, name='pred_scores')

    def define_optimizer(self):
        trues_pw = pairwise_sub(self.true_scores, name='trues_pw')
        ones_upper = tf.matrix_band_part(tf.ones_like(trues_pw), 0, -1, name='ones_upper')
        true_sign = tf.sign(trues_pw * ones_upper, name='true_sign')
        preds_pw = pairwise_sub(self.pred_scores, name='preds_pw')
        with tf.name_scope('margin_pw'):
            margin_pw = tf.maximum(1. - preds_pw * true_sign, 0.)

        self.margin_loss = tf.reduce_sum(margin_pw, name='margin_loss')
        if self.l2_reg is not None:
            reg_vars = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
            self.reg_loss = tf.add_n(reg_vars, name='reg_loss')
        else:
            self.reg_loss = tf.constant(0., name='reg_loss')

        self.total_loss = tf.add_n([self.margin_loss, self.reg_loss], name='total_loss')
        opt = tf.train.AdamOptimizer(learning_rate=self.lr, name='adam_opt')
        gvs_dis = [(g, v) for g, v in opt.compute_gradients(self.total_loss)
                   if 'gen_' not in v.name and g is not None]
        gvs_gen = list()
        for g, v in opt.compute_gradients(self.margin_loss):
            if 'gen_' in v.name and g is not None:
                print(v.name, 'is generator to min margin')
                gvs_gen.append((-g, v))  # gen should make margin loss larger

        print('len dis:{}, len gen:{}'.format(len(gvs_dis), len(gvs_gen)))
        self.train_dis = opt.apply_gradients(gvs_dis, name='train_dis')
        if len(gvs_gen) > 0:
            self.train_gen = opt.apply_gradients(gvs_gen, name='train_gen')
        else:
            self.train_gen = tf.constant(0., name='train_gen_holder')

    def get_loss(self, qwid, awids, uints, votes):
        fd = self.get_fd(qwid, awids, uints, votes)
        losses = self.sess.run([self.margin_loss, self.reg_loss, self.total_loss], feed_dict=fd)
        return dict(zip(['margin loss', 'reg loss', 'total loss'], losses))

    def train_step(self, qwid, awids, uints, votes, *args, **kwargs):
        fd = self.get_fd(qwid, awids, uints, votes)
        fd[self.is_train] = True

        self.sess.run(self.train_dis, feed_dict=fd)
        if kwargs['epoch'] >= 3:
            print('train gen')
            self.sess.run(self.train_gen, feed_dict=fd)
