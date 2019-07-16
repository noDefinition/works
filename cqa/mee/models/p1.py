from .v1 import *


# noinspection PyAttributeOutsideInit
class P1(V1):
    """ 完全pairwise """

    def forward(self):
        a_emb = self.embed_rep(self.a_lkup, self.a_mask, name='a_emb', k=10)
        au_cat = tf.concat([a_emb, self.u_lkup], axis=-1, name='au_cat')  # (bs, dw)

        # Q = instant_denses(au_cat, [(self.dim_w, relu)], name='Q', w_reg=self.l2_reg)
        # K = instant_denses(au_cat, [(self.dim_w, relu)], name='K', w_reg=self.l2_reg)
        # QK = tf.matmul(Q, K, transpose_b=True, name='QK')
        # QK = tf.divide(QK, tf.constant(np.sqrt(int(Q.shape[-1])), dtype=f32), name='QK_div')
        # QK = mask_diagly(QK, name='QK_mask_diagly')
        # QK_attn = tf.nn.softmax(QK, axis=1, name='QK_attn')
        # au_res = tf.matmul(QK_attn, au_cat, name='au_res')
        # self.QK_attn = QK_attn

        # au_final = tf.concat([au_cat, au_res, au_cat * au_res], axis=-1, name='au_final')
        au_final = au_cat
        uas = [(self.dim_w, relu), (self.dim_w, relu), (1, None)]
        preds = instant_denses(au_final, uas, name='preds', w_reg=self.l2_reg)
        self.pred_scores = tf.squeeze(preds, axis=1, name='pred_scores')

    def define_optimizer(self):
        true_pw = pairwise_sub(self.true_scores, name='true_pw')
        ones_upper_tri = tf.matrix_band_part(tf.ones_like(true_pw), 0, -1, name='ones_upper_tri')
        true_sign = tf.sign(true_pw * ones_upper_tri, name='true_sign')
        pred_pw = pairwise_sub(self.pred_scores, name='pred_pw')
        margin_pw = tf.maximum(1. - pred_pw * true_sign, 0., name='margin_pw')

        margin_loss = tf.reduce_sum(margin_pw, name='margin_loss')
        if self.l2_reg is not None:
            reg_vars = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
            reg_loss = tf.add_n(reg_vars, name='reg_loss')
        else:
            reg_loss = tf.constant(0., name='reg_loss')

        self.total_loss = tf.add(margin_loss, reg_loss, name='total_loss')
        opt = tf.train.AdamOptimizer(learning_rate=self.lr, name='adam_opt')
        self.train_op = opt.minimize(self.total_loss, name='train_op')


# noinspection PyAttributeOutsideInit
class P1m(P1):
    """ 用真实vote差作为margin """

    def forward(self):
        a_emb = self.embed_rep(self.a_lkup, self.a_mask, name='a_emb')
        au_cat = tf.concat([a_emb, self.u_lkup], axis=-1, name='au_cat')  # (bs, dw)
        uas = [(self.dim_w, relu), (self.dim_w, relu), (1, None)]
        preds = instant_denses(au_cat, uas, name='preds', w_reg=self.l2_reg)
        self.pred_probs = self.pred_scores = tf.squeeze(preds, axis=1, name='pred_scores')

    def define_optimizer(self):
        true_pw = pairwise_sub(self.true_scores, name='true_pw')
        ones_upper = tf.matrix_band_part(tf.ones_like(true_pw), 0, -1, name='ones_upper')
        true_sign = tf.sign(true_pw * ones_upper, name='true_sign')
        true_margin = tf.sqrt(tf.abs(true_pw, name='true_margin'))

        pred_pw = pairwise_sub(self.pred_scores, name='pred_pw')
        margin_pw = tf.maximum(true_margin - pred_pw * true_sign, 0., name='margin_pw')
        margin_loss = tf.reduce_sum(margin_pw, name='margin_loss')

        if self.l2_reg is not None:
            reg_vars = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
            reg_loss = tf.add_n(reg_vars, name='regular')
        else:
            reg_loss = tf.constant(0., name='regular')

        self.total_loss = tf.add_n([margin_loss, reg_loss], name='total_loss')
        opt = tf.train.AdamOptimizer(learning_rate=self.lr, name='adam_opt')
        self.train_op = opt.minimize(self.total_loss, name='train_op')


# noinspection PyAttributeOutsideInit
class R1(P1):
    """ 衡量重构差距 """

    def __init__(self, args):
        super(R1, self).__init__(args)
        self.eps = args[K.eps]

    def forward(self):
        a_emb = self.embed_rep(self.a_lkup, self.a_mask, name='a_emb')
        au_cat = tf.concat([a_emb, self.u_lkup], axis=-1, name='au_cat')  # (bs, dw)

        Q = instant_denses(au_cat, [(self.dim_w, relu)], name='Q', w_reg=self.l2_reg)
        K = instant_denses(au_cat, [(self.dim_w, relu)], name='K', w_reg=self.l2_reg)
        QK = tf.matmul(Q, K, transpose_b=True, name='QK')
        QK = tf.divide(QK, tf.constant(np.sqrt(int(Q.shape[-1])), dtype=f32), name='QK_div')
        QK = mask_diagly(QK, name='QK_mask_diagly')
        QK_attn = tf.nn.softmax(QK, axis=1, name='QK_attn')
        au_res = tf.matmul(QK_attn, au_cat, name='au_res')

        if self.eps > 1e-9:
            self.decode_loss = self.eps * tf.reduce_sum(au_cat - au_res, name='decode_loss')
        else:
            self.decode_loss = tf.constant(0.)
        uas = [(self.dim_w, relu), (1, None)]
        preds = instant_denses(au_res, uas, name='preds', w_reg=self.l2_reg)
        self.pred_probs = self.pred_scores = tf.squeeze(preds, axis=1, name='pred_scores')

    def define_optimizer(self):
        true_pw = pairwise_sub(self.true_scores, name='true_pw')
        ones_upper = tf.matrix_band_part(tf.ones_like(true_pw), 0, -1, name='ones_upper')
        true_sign = tf.sign(true_pw * ones_upper, name='true_sign')

        pred_pw = pairwise_sub(self.pred_scores, name='pred_pw')
        margin_pw = tf.maximum(1. - pred_pw * true_sign, 0., name='margin_pw')
        self.margin_loss = tf.reduce_sum(margin_pw, name='margin_loss')
        if self.l2_reg is not None:
            reg_vars = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
            self.reg_loss = tf.add_n(reg_vars, name='reg_loss')
        else:
            self.reg_loss = tf.constant(0., name='reg_loss')

        self.total_loss = tf.add_n(
            [self.margin_loss, self.reg_loss, self.decode_loss], name='total_loss')
        opt = tf.train.AdamOptimizer(learning_rate=self.lr, name='adam_opt')
        self.train_op = opt.minimize(self.total_loss, name='train_op')

    def get_loss(self, qwid, awids, uints, votes):
        fd = self.get_fd(qwid, awids, uints, votes)
        losses = self.sess.run(
            [self.decode_loss, self.margin_loss, self.reg_loss, self.total_loss],
            feed_dict=fd
        )
        return dict(zip(['decode gen2', 'margin gen2', 'reg gen2', 'total gen2'], losses))
