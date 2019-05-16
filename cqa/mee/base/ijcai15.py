from .common import *


# noinspection PyAttributeOutsideInit
class IJCAI15(CqaBaseline):
    def cnn_topk(self, t, maxlen, ns):
        with tf.variable_scope('CNN_tk', reuse=tf.AUTO_REUSE):
            t = tf.layers.conv1d(t, filters=10, kernel_size=3, kernel_initializer=self.x_init)
            t = self.top_k(t, ns + (maxlen - ns) // 2)
            t = tf.layers.conv1d(t, filters=5, kernel_size=3, kernel_initializer=self.x_init)
            t = self.top_k(t, ns)
            o = tf.layers.flatten(t)
        return o

    def forward(self):
        q = self.cnn_topk(self.q_lkup, self.len_q, ns=4)  # (1, ns)
        a = self.cnn_topk(self.a_lkup, self.len_a, ns=4)  # (bs, ns)
        print(q.shape, a.shape)
        with tf.variable_scope('match'):
            ns, r = 20, 5
            M = tf.get_variable(name='M', shape=(ns, ns, r), initializer=self.x_init)
            V = tf.get_variable(name='V', shape=(2 * ns, r), initializer=self.x_init)
            b = tf.get_variable(name='b', shape=(1, r), initializer=self.x_init)

            qm = tf.tensordot(q, M, axes=[1, 0])  # (1, ns, r)
            qm_tile = tf.tile(qm, [tf.shape(a)[0], 1, 1])
            a_exp = tf.expand_dims(a, 1)  # (bs, 1, ns)
            qma = tf.matmul(a_exp, qm_tile)  # (bs, 1, r)
            qma = tf.squeeze(qma, axis=1)  # (bs, r)

            # q_tile = tf.tile(q, [tf.shape(a)[0], 1])  # (bs, ns)
            q_tile = tf.ones_like(a) * q
            qa = tf.concat([q_tile, a], axis=1)  # (bs, ns * 2)
            qav = tf.matmul(qa, V)  # (bs, r)

            o = relu(qma + qav + b)
        uas = [(self.dim_w, relu), (self.dim_w, relu), (1, None)]
        preds = instant_denses(o, uas, name='preds')
        self.pred_scores = tf.squeeze(preds, axis=1, name='pred_scores')

    # def define_optimizer(self):
    #     true_pw = pairwise_sub(self.true_scores, name='true_pw')
    #     ones_upper_tri = tf.matrix_band_part(tf.ones_like(true_pw), 0, -1, name='ones_upper_tri')
    #     true_sign = tf.sign(true_pw * ones_upper_tri, name='true_sign')
    #     pred_pw = pairwise_sub(self.pred_scores, name='pred_pw')
    #     margin_pw = tf.maximum(1. - pred_pw * true_sign, 0., name='margin_pw')
    #     margin_pw_drop = tf.layers.dropout(margin_pw, rate=0.5, training=self.is_train)
    #     margin_loss = tf.reduce_sum(margin_pw_drop, name='margin_loss')
    #     self.total_loss = margin_loss
    #     opt = tf.train.AdagradOptimizer(learning_rate=0.1, name='adam_opt')
    #     self.train_op = opt.minimize(self.total_loss, name='train_op')
