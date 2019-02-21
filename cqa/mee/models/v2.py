from .v1 import *


# noinspection PyAttributeOutsideInit
class V2(V1):
    file = __file__

    def __init__(self, args):
        super(V2, self).__init__(args)
        self.prob_type = args[ptp_]
        self.mut_type = args[mtp_]
        self.ans_type = args[atp_]

    def forward(self):
        u_emb = tf.nn.embedding_lookup(self.user_embed, self.uint_seq, name='u_lookup')
        a_emb = self.get_doc_embed(self.awid_seq, name='a_embed')
        au_cat = tf.concat([a_emb, u_emb], axis=-1)

        # au_ffn = au_cat
        uas = [(512, relu), (256, relu), (self.w_dim, None)]
        # uas = [(au_cat.shape[1], relu), (self.w_dim, None)]
        au_ffn = denses(au_cat, uas, name='au_ffn')
        mut_sup = tf.matmul(au_ffn, tf.transpose(au_ffn))
        mut_att = tf.nn.softmax(mask_diagly(mut_sup), axis=1)

        if self.prob_type == 0:
            pred_probs = tf.reduce_mean(mut_att, axis=0, keepdims=False)
        elif self.prob_type == 1:
            if self.ans_type == 0:
                au_final = au_cat
            elif self.ans_type in {1, 2}:
                mut_att_choice = [mut_att, tf.transpose(mut_att)][self.mut_type]
                au_res = tf.matmul(mut_att_choice, au_ffn)
                au_res_cat = tf.concat([au_ffn, au_res, au_ffn * au_res], axis=1)
                au_final = [None, au_res, au_res_cat][self.ans_type]
            else:
                raise ValueError('invalid answer type {}'.format(self.ans_type))
            uas = [(512, relu), (256, relu), (1, None)]
            # uas = [(self.w_dim, relu), (1, None)]
            pred_score = denses(au_final, uas, name='pred_score')
            pred_probs = tf.nn.softmax(tf.squeeze(pred_score, axis=1))
        else:
            raise ValueError('invalid probability type {}'.format(self.prob_type))

        self.pred_probs = pred_probs
        self.true_probs = tf.nn.softmax(tf.cast(tf.maximum(self.vote_seq, 0), dtype=f32))
        self.kl_loss = get_kl_loss(self.true_probs, self.pred_probs)
