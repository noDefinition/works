from .common import *


# noinspection PyAttributeOutsideInit
class AAAI15(CqaBaseline):
    def forward(self):
        print('***')
        with tf.variable_scope('S-matrix'):
            q_lkup_exp = tf.expand_dims(self.q_lkup, 2)  # (1, lq, 1, dw)
            a_lkup_exp = tf.expand_dims(self.a_lkup, 1)  # (bs, 1, la, dw)
            m = self.cos(q_lkup_exp, a_lkup_exp)
            m = tf.expand_dims(m, -1)
            print(q_lkup_exp.shape, a_lkup_exp.shape, m.shape)
        with tf.variable_scope('CNN'):
            m = tf.layers.conv2d(m, 20, 5, kernel_initializer=self.x_init)
            m = tf.layers.max_pooling2d(m, (2, 3), (2, 3))
            m = tf.layers.conv2d(m, 50, 5, kernel_initializer=self.x_init)
            m = tf.layers.max_pooling2d(m, (2, 3), (2, 3))
            o = tf.layers.flatten(m)
            print(m.shape, o.shape)
        print('***')
        uas = [(self.dim_w, relu), (self.dim_w, relu), (1, None)]
        preds = instant_denses(o, uas, name='preds')
        self.pred_scores = tf.squeeze(preds, axis=1, name='pred_scores')
