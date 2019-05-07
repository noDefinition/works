from .s1 import *


# noinspection PyAttributeOutsideInit
class S3(S1):
    """ 相似度加成(改) """

    def __init__(self, args):
        super(S3, self).__init__(args)
        self.act = args.get(K.act, None)
        self.dpt = args.get(K.dpt, None)
        self.mix = args.get(K.mix, None)
        assert self.act is not None and self.act in {0, 1, 2}
        assert self.dpt is not None and self.dpt >= 1
        assert self.mix is not None

    def forward(self):
        # act = [relu, tanh, sigmoid][self.act]
        act = relu
        to_scalar = [(1, None)]
        a_emb = self.embed_rep(self.a_lkup, self.a_mask, name='a_emb', k=10)
        au_cat = tf.concat([a_emb, self.u_lkup, a_emb * self.u_lkup], axis=-1, name='au_cat')

        if np.abs(self.mix) > 1e-9:
            uas = [(self.dim_w, act)] * self.dpt + to_scalar
            pop_preds = instant_denses(au_cat, uas, name='pop_preds')

            uas = [(self.dim_w, act)] * self.dpt
            a_emb_fc = instant_denses(a_emb, uas, name='a_emb_fc')
            # a_emb_sum = tf.reduce_sum(a_emb_fc, axis=0, keepdims=True, name='a_emb_sum')
            # a_emb_sum = tf.subtract(a_emb_sum, a_emb_fc, name='a_emb_sum')
            # au_hrd = tf.concat(a_emb_fc * a_emb_sum, axis=-1, name='au_hrd')
            a_emb_sum = tf.reduce_mean(a_emb_fc, axis=0, keepdims=True, name='a_emb_sum')
            au_hrd = tf.multiply(a_emb_fc, a_emb_sum, name='au_hrd')
            uas = [(self.dim_w, act)] + to_scalar
            sim_preds = instant_denses(au_hrd, uas, name='sim_preds')

            preds = tf.add(pop_preds, self.mix * sim_preds, name='preds')
        else:
            uas = [(self.dim_w, act)] * (self.dpt * 2) + to_scalar
            preds = instant_denses(au_cat, uas, name='preds')

        self.pred_scores = tf.squeeze(preds, axis=1, name='pred_scores')
