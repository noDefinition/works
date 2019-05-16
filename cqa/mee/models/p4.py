from tensorflow.keras.layers import Bidirectional, GRU, Masking, Input

from .p1 import *


# noinspection PyAttributeOutsideInit
class P4(P1):
    """ with/without LSTM + user attention """

    def __init__(self, args):
        super(P4, self).__init__(args)
        self.att_type = args.get(K.atp)
        self.text_type = args.get(K.ttp)
        self.topk = args.get(K.topk, None)
        assert self.att_type in {0, 1}
        assert self.text_type in {0, 1}

    def bi_gru_output(self, inputs):
        with tf.variable_scope('recurrent', reuse=tf.AUTO_REUSE):
            cell = GRU(
                units=self.dim_w,
                return_sequences=True,
                dropout=self.dropout,
                recurrent_dropout=self.dropout,
            )
            gru_out = Bidirectional(cell)(
                inputs=Masking()(inputs),
                training=self.is_train
            )
            # print('gru_out.shape', gru_out.shape)
        return gru_out

    def forward(self):
        if self.text_type == 0:
            uas = [(self.dim_w, tanh)]
            a_emb = instant_denses(self.a_lkup, uas, name='a_emb')  # (bs, la, dw)
        elif self.text_type == 1:
            a_gru = self.bi_gru_output(self.a_lkup)  # (bs, la, dh * 2)
            uas = [(self.dim_w, tanh)]
            a_emb = instant_denses(a_gru, uas, name='a_emb')  # (bs, la, dw)
        else:
            raise ValueError('invalid att_type:{}'.format(self.att_type))

        if self.att_type == 0:
            if self.text_type == 0:
                assert isinstance(self.topk, int) and 1 <= self.topk
                a_recon = self.top_k_mean(a_emb, k=self.topk)  # (bs, dw)
            elif self.text_type == 1:
                a1 = a_emb[:, 0, self.dim_w:]  # (bs, dw)
                a2 = a_emb[:, -1, :self.dim_w]  # (bs, dw)
                a_recon = tf.concat([a1, a2], axis=-1)  # (bs, dw * 2)
                print(a_recon.shape)
            else:
                raise ValueError('fuck 2')
            au_recon = tf.concat([a_recon, self.u_lkup], axis=-1)  # (bs, dw * ?)
        elif self.att_type == 1:
            uas = [(self.dim_w, tanh)]
            u_lkup = instant_denses(self.u_lkup, uas, name='u_lkup')
            u_lkup_exp = tf.expand_dims(u_lkup, axis=-1)  # (bs, dw, 1)
            au_mult = tf.matmul(a_emb, u_lkup_exp)  # (bs, la, 1)
            au_mask = (1.0 - self.a_mask) * (-1e9)  # (bs, la, 1)
            au_score = tf.add(au_mult, au_mask)  # (bs, la, 1)
            au_score = tf.transpose(au_score, perm=[0, 2, 1])  # (bs, 1, la)
            au_atten = tf.nn.softmax(au_score, axis=-1)  # (bs, 1, la)
            au_recon = tf.matmul(au_atten, a_emb)  # (bs, 1, dw)
            au_recon = tf.squeeze(au_recon, axis=1)  # (bs, dw)
        else:
            raise ValueError('fuck 3')

        uas = [(self.dim_w, relu)] * 2 + [(1, None)]
        preds = instant_denses(au_recon, uas, name='preds')
        self.pred_scores = tf.squeeze(preds, axis=1, name='pred_scores')
