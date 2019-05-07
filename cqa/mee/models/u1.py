from .v1 import *


# noinspection PyAttributeOutsideInit
class U1(V1):
    """ 用户/单词加 **对抗噪声** """

    def __init__(self, args):
        super(U1, self).__init__(args)
        self.woru = args[K.woru]
        self.eps = args[K.eps]
        self.use_adv = True if self.woru in {1, 2, 3} else False

    def define_word_embed(self, word_embed):
        super(U1, self).define_word_embed(word_embed)
        init, shape = self.x_init, self.word_embed.shape
        self.word_noise = tf.get_variable('word_noise', initializer=init, shape=shape)
        self.wemb_with_noise = self.word_embed + self.word_noise

    def define_user_embed(self, user_embed):
        super(U1, self).define_user_embed(user_embed)
        init, shape = self.x_init, self.user_embed.shape
        self.user_noise = tf.get_variable('user_noise', initializer=init, shape=shape)
        # self.uemb_with_noise = self.user_embed + self.user_noise

    def define_inputs(self):
        super(U1, self).define_inputs()
        lookup = tf.nn.embedding_lookup
        self.a_lkup_nis = lookup(self.wemb_with_noise, self.awid_seq, name='a_lkup_nis')
        self.q_lkup_nis = lookup(self.wemb_with_noise, self.qwid_exp, name='q_lkup_nis')
        self.u_lkup_nis = lookup(self.user_noise, self.uint_seq, name='u_lkup_nis')

    def forward(self):
        a_emb = self.embed_rep(self.a_lkup, self.a_mask, 'a_emb')
        dense_layers = [
            tf.layers.Dense(u, a, kernel_regularizer=self.l2_reg, name='ffn_{}'.format(i))
            for i, (u, a) in enumerate([(self.dim_w, relu), (1, None)])
        ]

        if self.use_adv:
            a_emb_nis = self.embed_rep(self.a_lkup_nis, self.a_mask, 'a_emb_nis')
            au_sel = [
                None, (a_emb_nis, self.u_lkup),
                (a_emb + self.u_lkup_nis, self.u_lkup + self.u_lkup_nis),
            ][self.woru]
            # au_sel = [a_emb + self.u_lkup_nis, self.u_lkup + self.u_lkup_nis]
            au_nis = tf.concat(au_sel, axis=-1, name='au_nis')
            self.pred_scores_nis = postpone_denses(dense_layers, au_nis, name='pred_scores_nis')
            self.pred_scores_nis = tf.squeeze(self.pred_scores_nis, axis=1)

        au_cat = tf.concat([a_emb, self.u_lkup], axis=-1, name='au_cat')
        self.pred_scores = postpone_denses(dense_layers, au_cat, name='pred_score')
        self.pred_probs = self.pred_scores = tf.squeeze(self.pred_scores, axis=1)

    def define_optimizer(self):
        trues_pw = pairwise_sub(self.true_scores, name='trues_pw')
        ones_upper_tri = tf.matrix_band_part(tf.ones_like(trues_pw), 0, -1)
        true_sign = tf.sign(trues_pw * ones_upper_tri, name='true_sign')

        preds_pw = pairwise_sub(self.pred_scores, name='preds_pw')
        margin_pw = tf.maximum(1. - preds_pw * true_sign, 0., name='margin_pw')
        self.total_loss = tf.reduce_sum(margin_pw, name='margin_loss')

        if self.use_adv:
            preds_pw_nis = pairwise_sub(self.pred_scores_nis, name='preds_pw_nis')
            margin_pw_nis = tf.maximum(1. - preds_pw_nis * true_sign, 0., name='margin_pw_nis')
            self.total_loss += tf.reduce_sum(margin_pw_nis, name='margin_loss_nis')
        if self.l2_reg is not None:
            self.total_loss += tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
            print('reged vars:', len(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)))

        opt = tf.train.AdamOptimizer(learning_rate=self.lr)
        gvs = opt.compute_gradients(self.total_loss, tf.trainable_variables())
        noises = {self.word_noise, self.user_noise}
        self.assign_ops = list()
        for g, v in gvs:
            if v in noises and g is not None:
                assign_op = tf.assign(v, - self.eps * tf.nn.l2_normalize(g))
                self.assign_ops.append(assign_op)
        # for noise in noises:
        #     if noise in v2g:
        #         print('noise var: ', noise.name)
        #         assign_op = tf.assign(noise, - self.eps * tf.nn.l2_normalize(v2g[noise]))
        #         self.reassign_ops.append(assign_op)
        print('woru', self.woru, self.assign_ops)
        gvs = [(g, v) for g, v in gvs if v not in noises]
        self.train_op = opt.apply_gradients(gvs, name='adv_op')

    def train_step(self, qwid, awids, uints, votes, *args, **kwargs):
        fd = self.get_fd(qwid, awids, uints, votes)
        fd[self.is_train] = True
        self.sess.run([self.train_op] + self.assign_ops, feed_dict=fd)
