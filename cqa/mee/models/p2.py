from .p1 import *


# noinspection PyAttributeOutsideInit
class P2(P1):
    """ [qa, a, u]三者attention """

    def forward(self):
        q_emb = self.embed_rep(self.q_lkup, self.q_mask, name='q_emb')
        a_emb = self.embed_rep(self.a_lkup, self.a_mask, name='a_emb', k=10)
        u_emb = self.u_lkup
        # au_cat = tf.concat([a_emb, self.u_lkup], axis=-1, name='au_cat')  # (bs, dw)

        qa_hdm = tf.multiply(q_emb, a_emb, name='qa_hdm')
        qu_hdm = tf.multiply(q_emb, u_emb, name='qu_hdm')
        # u_proj = instant_denses(u_emb, [(self.dim_w // 2, tanh)], name='u_proj')
        uas = [(self.dim_w, relu)]
        a_proj = instant_denses(a_emb, uas, name='a_proj')
        qa_proj = instant_denses(qa_hdm, uas, name='qa_proj')
        qu_proj = instant_denses(qu_hdm, uas, name='qu_proj')
        with tf.name_scope('qau_att'):
            qau_mul = tf.concat([a_proj, qa_proj, qu_proj], axis=-1)
            qau_scr = instant_denses(qau_mul, [(3, None)], name='qau_scr')  # (bs, 3)
            qau_att = softmax(qau_scr, axis=-1, name='qau_att')  # (bs, 3)
            qau_att_exp = tf.expand_dims(qau_att, axis=1, name='qau_att_exp')  # (bs, 1, 3)
        with tf.name_scope('qau_cat'):
            exp_proj = [a_proj, qa_proj, qu_proj]
            exp_proj = [tf.expand_dims(m, axis=1) for m in exp_proj]  # (bs, 1, dw // 2)
            qau_cat = tf.concat(exp_proj, axis=1, name='qau_cat')  # (bs, 3, dw // 2)
        with tf.name_scope('qau_att_sum'):
            qau_avg = tf.matmul(qau_att_exp, qau_cat)
            qau_avg = tf.squeeze(qau_avg, axis=1)

        uas = [(self.dim_w, relu), (1, None)]
        preds = instant_denses(qau_avg, uas, name='preds', w_reg=self.l2_reg)
        self.pred_probs = self.pred_scores = tf.squeeze(preds, axis=1, name='pred_scores')

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
