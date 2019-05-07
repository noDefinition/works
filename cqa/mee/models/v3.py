from .v1 import *


# noinspection PyAttributeOutsideInit
class V3(V1):
    file = __file__

    def forward(self):
        a_emb = self.embed_rep(self.a_lkup, self.a_mask, 'a_emb')
        q_emb = self.embed_rep(self.q_lkup, self.q_mask, 'q_emb')
        u_emb = self.u_lkup
        au_cat = tf.concat([a_emb, u_emb], axis=-1)

        uas = [(self.dim_w, relu), (self.dim_w, None)]
        au_ffn = instant_denses(au_cat, uas, name='au_ffn')
        qau_cat = tf.concat([q_emb, au_ffn], axis=0)
        mut_sup = tf.matmul(qau_cat, tf.transpose(qau_cat))
        mut_att = tf.nn.softmax(mask_diagly(mut_sup, name='mask_mut_sup'), axis=1)
        qau_res = tf.matmul(mut_att, qau_cat)
        qau_final = tf.concat([qau_cat, qau_res, qau_cat * qau_res], axis=-1)
        uas = [(self.dim_w, relu), (1, None)]
        preds = instant_denses(qau_final, uas, name='preds')

        self.pred_probs = tf.nn.softmax(tf.squeeze(preds, axis=1)[1:])
        self.true_probs = tf.nn.softmax(tf.cast(self.vote_seq, dtype=f32))
