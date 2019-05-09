from .p1 import *


# noinspection PyAttributeOutsideInit
class P3(P1):
    """ 更深的pariwise """

    def __init__(self, args):
        super(P3, self).__init__(args)
        self.w_or_u = args.get(K.woru, None)
        self.topk = args.get(K.topk, None)
        assert isinstance(self.w_or_u, int)
        assert isinstance(self.topk, int) and 1 <= self.topk

    def forward(self):
        uas = [(self.dim_w, tanh)]
        if self.w_or_u == 0:
            a_emb = self.embed_rep(self.a_lkup, self.a_mask, name='a_emb', k=self.topk)
            au_cat = instant_denses(a_emb, uas, name='a_emb_p')
        elif self.w_or_u == 1:
            u_emb = self.u_lkup
            au_cat = instant_denses(u_emb, uas, name='u_emb_p')
        elif self.w_or_u == 2:
            a_emb = self.embed_rep(self.a_lkup, self.a_mask, name='a_emb', k=self.topk)
            au_cat = tf.concat([a_emb, self.u_lkup], axis=-1, name='au_cat')  # (bs, dw)
        else:
            raise ValueError('w_or_u invalid {}'.format(self.w_or_u))
        # activation = [relu, tanh, sigmoid][self.act]
        uas = [(self.dim_w, tanh)] * 2 + [(1, None)]
        preds = instant_denses(au_cat, uas, name='preds')
        self.pred_scores = tf.squeeze(preds, axis=1, name='pred_scores')
