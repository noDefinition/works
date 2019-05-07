from .p1 import *


# noinspection PyAttributeOutsideInit
class P3(P1):
    """ 更深的pariwise """

    def __init__(self, args):
        super(P3, self).__init__(args)
        self.depth = args.get(K.dpt, None)
        self.act = args.get(K.act, None)
        assert self.depth is not None and self.depth >= 1
        assert self.act is not None and self.act in {0, 1, 2}

    def forward(self):
        a_emb = self.embed_rep(self.a_lkup, self.a_mask, name='a_emb', k=10)
        au_cat = tf.concat([a_emb, self.u_lkup], axis=-1, name='au_cat')  # (bs, dw)
        activation = [relu, tanh, sigmoid][self.act]
        uas = [(self.dim_w, activation)] * self.depth + [(1, None)]
        preds = instant_denses(au_cat, uas, name='preds')
        self.pred_probs = self.pred_scores = tf.squeeze(preds, axis=1, name='pred_scores')
