from dc.models.v1 import *


# noinspection PyAttributeOutsideInit
class V3(V1):
    """ 更好地问题交互 """
    file = __file__

    def forward(self):
        u_emb = tf.nn.embedding_lookup(self.u_embed, self.uint_seq, name='user_lookup')
        a_emb = self.get_ans_embed(self.awid_seq, name='a_lookup_mask')
        au_rep = tf.concat([a_emb, u_emb], axis=-1)
        mu_ua_list = [(512, relu), (256, relu), (au_rep.shape[1], None)]
        au_rep_d = denses(au_rep, mu_ua_list, name='au_rep_denses')

        mut_sup = tf.matmul(au_rep, tf.transpose(au_rep_d))
        # mut_mask = - 1e32 * tf.eye(tf.shape(mut_sup)[0])
        mut_mask = - tf.diag(tf.diag_part(mut_sup))
        mut_att = tf.nn.softmax(mut_sup + mut_mask, axis=1)
        au_res = tf.matmul(mut_att, au_rep)
        mut_probs = tf.reduce_mean(mut_att, axis=0, keepdims=False)

        au_rep_final = tf.concat([au_rep, au_res, au_rep * au_res], axis=1)
        re_ua_list = [(512, relu), (256, relu), (1, None)]
        res_scr = denses(au_rep_final, re_ua_list, name='res_scr')
        res_probs = tf.squeeze(tf.nn.softmax(res_scr), axis=1)

        self.true_probs = tf.nn.softmax(tf.cast(self.vote_seq, dtype=f32))
        self.pred_probs = {0: mut_probs, 1: res_probs}[self.prob_type]
        self.kl_loss = self.get_kl_loss(self.true_probs, self.pred_probs)
        self.mut_att = mut_att
