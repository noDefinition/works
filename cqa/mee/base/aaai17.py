from .common import *


# noinspection PyAttributeOutsideInit
class AAAI17(CqaBaseline):
    def mask_sum(self, t, mask, name='mask_sum'):
        with tf.variable_scope(name):
            s = tf.reduce_sum(t * mask, axis=1, keepdims=False)
        return s

    def mask_mean(self, t, mask, name='mask_mean'):
        with tf.variable_scope(name):
            s = self.mask_sum(t, mask)
            d = tf.reduce_sum(mask, axis=[1, 2], keepdims=True)
            d = tf.squeeze(d, axis=-1)
            print('s.shape', s.shape, 'd.shape', d.shape)
        # return s / d
        return s

    def forward(self):
        q_emb = self.mask_mean(self.q_lkup, self.q_mask)  # (1, dw)
        a_emb = self.mask_mean(self.a_lkup, self.a_mask)  # (bs, dw)
        print(q_emb.shape, a_emb.shape)
        qw = tf.get_variable(name='qw', shape=(self.dim_w, self.dim_w), initializer=self.x_init)
        q_proj = tf.matmul(q_emb, qw)  # (1, dw)
        # a_emb_proj = instant_denses(a_emb, [(self.dim_w, tanh)], name='a_emb_proj')
        # u_emb_proj = instant_denses(self.u_lkup, [(self.dim_w, tanh)], name='u_lkup_proj')
        qa = self.dot(q_proj, a_emb)  # (bs,)
        qu = self.dot(q_emb, self.u_lkup)  # (bs,)
        self.pred_scores = qa * qu
        # print(o.shape)
        # uas = [(self.dim_w, relu), (self.dim_w, relu), (1, None)]
        # preds = instant_denses(o, uas, name='preds')
        # self.pred_scores = tf.squeeze(preds, axis=1, name='pred_scores')
