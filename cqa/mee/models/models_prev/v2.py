from dc.models.v1 import *


# noinspection PyAttributeOutsideInit
class V2(V1):
    """ 带问题 """
    file = __file__

    def __init__(self, args: dict):
        super(V2, self).__init__(args)
        self.aqm = args[aqm_]

    def get_aq_emb(self, q_emb, a_emb):
        if self.aqm == 0:
            aq_emb = a_emb
        elif self.aqm == 1:
            ua_list = [(self.w_dim, relu)]
            aq_emb = denses(q_emb, ua_list, name='q_new') + denses(a_emb, ua_list, name='a_new')
        elif self.aqm == 2:
            aq_emb = q_emb + a_emb
        elif self.aqm == 3:
            aq_emb = q_emb * a_emb
        else:
            raise ValueError('unsupported aqm:{}'.format(self.aqm))
        return aq_emb

    def forward(self):
        u_emb = tf.nn.embedding_lookup(self.u_embed, self.uint_seq, name='user_lookup')
        q_emb = self.get_ans_embed(tf.expand_dims(self.qwid, axis=0), name='q_lookup_mask')
        a_emb = self.get_ans_embed(self.awid_seq, name='a_lookup_mask')
        aq_emb = self.get_aq_emb(q_emb, a_emb)
        aqu_rep = tf.concat([aq_emb, u_emb], axis=-1)
        ua_list = [(512, relu), (256, relu), (1, None)]
        res_scr = denses(aqu_rep, ua_list, 'res_scr')
        self.pred_probs = tf.nn.softmax(tf.squeeze(res_scr, axis=1))
        self.true_probs = tf.nn.softmax(tf.cast(tf.maximum(self.vote_seq, 0), dtype=f32))
        self.kl_loss = self.get_kl_loss(self.true_probs, self.pred_probs)
