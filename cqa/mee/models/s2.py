from .s1 import *


# noinspection PyAttributeOutsideInit
class S2(S1):
    """ 相似度加成(改) """

    def __init__(self, args):
        super(S2, self).__init__(args)
        self.act = args.get(K.act, None)
        self.dpt = args.get(K.dpt, None)
        self.mix = args.get(K.mix, None)
        assert self.act is not None and self.act in {0, 1, 2}
        assert self.dpt is not None and self.dpt >= 1
        assert self.mix is not None

    def forward(self):
        act = [relu, tanh, sigmoid][self.act]
        a_emb = self.embed_rep(self.a_lkup, self.a_mask, name='a_emb', k=10)
        au_cat = tf.concat([a_emb, self.u_lkup, a_emb * self.u_lkup], axis=-1, name='au_cat')

        if self.mix > 1e-9:
            uas = [(self.dim_w, act)] * self.dpt + [(1, None)]
            pop_preds = instant_denses(au_cat, uas, name='pop_preds')

            uas = [(self.dim_w, act)] * self.dpt
            au_fc = instant_denses(au_cat, uas, name='au_fc')
            # au_fc_sum = tf.reduce_sum(au_fc, axis=0, keepdims=True, name='au_fc_sum')
            # au_fc_sum = tf.subtract(au_fc_sum, au_fc, name='au_fc_sum')
            # au_hrd = tf.concat(au_fc * au_fc_sum, axis=-1, name='au_hrd')
            au_fc_sum = tf.reduce_mean(au_fc, axis=0, keepdims=True, name='au_fc_sum')
            au_hrd = tf.multiply(au_fc, au_fc_sum, name='au_hrd')
            uas = [(self.dim_w, act)] + [(1, None)]
            sim_preds = instant_denses(au_hrd, uas, name='sim_preds')

            preds = tf.add(pop_preds, self.mix * sim_preds, name='preds')
        else:
            uas = [(self.dim_w, act)] * (self.dpt * 2) + [(1, None)]
            preds = instant_denses(au_cat, uas, name='preds')

        self.pred_probs = self.pred_scores = tf.squeeze(preds, axis=1, name='pred_scores')
