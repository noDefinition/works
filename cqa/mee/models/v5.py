from .v1 import *


# noinspection PyAttributeOutsideInit
class V4(V1):
    """ transformer式 用户-单词 + 问题-答案 """
    file = __file__

    def forward(self):
        u_emb = self.u_emb  # (bs, ud)
        q_emb, q_mask = self.lookup_and_mask(self.qwid_exp, name='q_lookup_mask')  # (1, nq, wd)
        a_emb, a_mask = self.lookup_and_mask(self.awid_seq, name='a_lookup_mask')  # (bs, na, wd)

        u_emb = tf.expand_dims(u_emb, axis=1)  # (bs, 1, ud)

        aq_emb = self.get_aq_emb(q_emb, a_emb)
        aqu_rep = tf.concat([aq_emb, u_emb], axis=-1)
        ua_list = [(512, relu), (256, relu), (1, None)]
        res_scr = denses(aqu_rep, ua_list, 'res_scr')
        self.pred_probs = tf.nn.softmax(tf.squeeze(res_scr, axis=1))
        self.true_probs = tf.nn.softmax(tf.cast(tf.maximum(self.vote_seq, 0), dtype=f32))
        self.kl_loss = get_kl_loss(self.true_probs, self.pred_probs)
