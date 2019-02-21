from .v1 import *


# noinspection PyAttributeOutsideInit
class V3(V1):
    """ 类似Transformer的问题答案交互 """
    file = __file__

    def forward(self):
        a_emb = self.get_doc_embed(self.awid_seq, name='a_embed')
        q_emb = self.get_doc_embed(tf.expand_dims(self.qwid, axis=0), name='q_embed')
        u_emb = tf.nn.embedding_lookup(self.user_embed, self.uint_seq, name='u_lookup')

        au_cat = tf.concat([a_emb, u_emb], axis=-1)
        uas = [(self.w_dim, relu), (self.w_dim, None)]
        au_ffn = denses(au_cat, uas, name='au_ffn')

        qau_cat = tf.concat([q_emb, au_ffn], axis=0)
        qau_mut = tf.matmul(qau_cat, tf.transpose(qau_cat))
        mut_att = tf.nn.softmax(mask_diagly(qau_mut), axis=1)
        qau_res = tf.matmul(mut_att, qau_cat)
        qau_final = tf.concat([qau_cat, qau_res, qau_cat * qau_res], axis=-1)
        uas = [(self.w_dim, relu), (1, None)]
        qau_score = denses(qau_final, uas, name='qau_score')

        self.pred_probs = tf.nn.softmax(tf.squeeze(qau_score, axis=1)[1:])
        self.true_probs = tf.nn.softmax(tf.cast(self.vote_seq, dtype=f32))
        self.kl_loss = get_kl_loss(self.true_probs, self.pred_probs)
