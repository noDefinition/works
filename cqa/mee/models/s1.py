from .v1 import *


# noinspection PyAttributeOutsideInit
class S1(V1):
    """ 相似度加成 """

    def __init__(self, args):
        super(S1, self).__init__(args)
        assert K.mix in args and args[K.mix] is not None
        self.mix = args[K.mix]

    def forward(self):
        a_emb = self.embed_rep(self.a_lkup, self.a_mask, name='a_emb', k=10)
        au_cat = tf.concat([a_emb, self.u_lkup], axis=-1, name='au_cat')  # (bs, dw)
        # au_final = tf.concat([au_cat, au_res, au_cat * au_res], axis=-1, name='au_final')

        uas = [(self.dim_w, relu), (self.dim_w, None)]
        au_rep = instant_denses(au_cat, uas, name='au_rep')

        uas = [(self.dim_w, tanh), (self.dim_w, None)]
        au_fc = instant_denses(au_rep, uas, name='au_fc')
        au_fc_sum = tf.reduce_sum(au_fc, axis=0, keepdims=True, name='au_fc_sum')
        au_fc_sum_sub = tf.subtract(au_fc_sum, au_fc, name='au_fc_sum_sub')
        au_hrd = tf.multiply(au_fc, au_fc_sum_sub, name='au_hrd')

        uas = [(1, tanh)]
        sim_preds = instant_denses(au_hrd, uas, name='sim_preds')
        uas = [(1, None)]
        org_preds = instant_denses(au_rep, uas, name='org_preds')

        preds = tf.add(self.mix * sim_preds, org_preds, name='preds')
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
