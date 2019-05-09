from .p1 import *
from tensorflow.keras.layers import GRU, Bidirectional, Concatenate, RNN


# noinspection PyAttributeOutsideInit
class P4(P1):
    """ LSTM """

    def __init__(self, args):
        super(P4, self).__init__(args)
        self.w_or_u = args.get(K.woru, None)
        self.topk = args.get(K.topk, None)
        assert isinstance(self.w_or_u, int)
        assert isinstance(self.topk, int) and 1 <= self.topk

    def embed_gru(self, inputs):
        with tf.variable_scope('recurrent', reuse=tf.AUTO_REUSE):
            dropout = self.dropout_keep_prob
            gru_cell = GRU(self.dim_w, dropout=dropout, recurrent_dropout=dropout)
            init_state = gru_cell.zero_state(tf.shape(inputs)[0], f32)
            init_state = tf.tile(tf.zeros([1, ]))
            gru_out = Bidirectional(gru_cell).__call__(inputs=inputs, initial_state=init_state)
        return

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
