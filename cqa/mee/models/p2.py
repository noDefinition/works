from .p1 import *


# noinspection PyAttributeOutsideInit
class P2(P1):
    """ [qa, a, u]三者attention """

    def __init__(self, args):
        super(P2, self).__init__(args)
        self.qtp = args.get(K.qtp, None)
        self.topk = args.get(K.topk, None)
        assert isinstance(self.qtp, int)
        assert isinstance(self.topk, int) and 1 <= self.topk

    def forward(self):
        q_topk = self.embed_rep(self.q_lkup, self.q_mask, name='q_topk', k=self.topk)  # (1, dw)
        a_topk = self.embed_rep(self.a_lkup, self.a_mask, name='a_topk', k=self.topk)  # (bs, dw)
        u_emb = self.u_lkup
        a_shape = tf.shape(a_topk)
        q_topk_tile = tf.tile(q_topk, multiples=[a_shape[0], 1])  # (bs, dw)

        if self.qtp == 0:
            qau = tf.concat([q_topk_tile, a_topk], axis=-1)  # (bs, dw * 2)
        elif self.qtp == 1:
            qau = tf.concat([q_topk_tile, a_topk, u_emb], axis=-1)  # (bs, dw * 3)
        elif self.qtp == 2:
            uas = [(self.dim_w, tanh)]
            a_proj = instant_denses(self.a_lkup, uas, name='a_proj')  # (bs, tn, dw)
            q_topk_tile_exp = tf.expand_dims(q_topk_tile, axis=-1)  # (bs, dw, 1)
            aq_score = tf.matmul(a_proj, q_topk_tile_exp)  # (bs, tn, 1)
            aq_probs = tf.nn.softmax(aq_score, axis=1)  # (bs, tn, 1)
            aq_probs_T = tf.transpose(aq_probs, perm=[0, 2, 1])  # (bs, 1, tn)
            aq_recon = tf.matmul(aq_probs_T, a_proj)  # (bs, 1, dw)
            qau = tf.squeeze(aq_recon, axis=1)  # (bs, dw)
        elif self.qtp == 3:
            qa_hdm = tf.multiply(q_topk, a_topk, name='qa_hdm')  # (bs, dw)
            qu_hdm = tf.multiply(q_topk, u_emb, name='qu_hdm')  # (bs, dw)
            uas = [(self.dim_w, relu)]
            a_proj = instant_denses(a_topk, uas, name='a_proj')  # (bs, dw)
            qa_proj = instant_denses(qa_hdm, uas, name='qa_proj')  # (bs, dw)
            qu_proj = instant_denses(qu_hdm, uas, name='qu_proj')  # (bs, dw)
            # with tf.name_scope('qau_probs'):
            qau_concat = tf.concat([a_proj, qa_proj, qu_proj], axis=-1)  # (bs, dw * 3)
            qau_score = instant_denses(qau_concat, [(3, None)], name='qau_score')  # (bs, 3)
            qau_probs = tf.nn.softmax(qau_score, axis=-1, name='qau_probs')  # (bs, 3)
            qau_probs_exp = tf.expand_dims(qau_probs, axis=1, name='qau_probs_exp')  # (bs, 1, 3)
            # with tf.name_scope('qau_tile'):
            exp_proj = [a_proj, qa_proj, qu_proj]
            exp_proj = [tf.expand_dims(m, axis=1) for m in exp_proj]  # (bs, 1, dw) * 3
            qau_tile = tf.concat(exp_proj, axis=1, name='qau_tile')  # (bs, 3, dw)
            # with tf.name_scope('qau_att_sum'):
            qau = tf.matmul(qau_probs_exp, qau_tile)  # (bs, 1, dw)
            qau = tf.squeeze(qau, axis=1)  # (bs, dw)
        else:
            raise ValueError('FUCK qtp', self.qtp)

        uas = [(self.dim_w, relu), (self.dim_w, relu), (1, None)]
        preds = instant_denses(qau, uas, name='preds')  # (bs, 1)
        self.pred_scores = tf.squeeze(preds, axis=1, name='pred_scores')
