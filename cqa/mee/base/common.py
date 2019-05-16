from cqa.mee.models.v1 import *


# noinspection PyAttributeOutsideInit
class CqaBaseline(V1):
    def __init__(self, args):
        super(CqaBaseline, self).__init__(args)
        self.pred_scores = None

    def dot(self, a, b, axis=-1):
        return tf.reduce_sum(a * b, axis)

    def cos(self, a, b, axis=-1):
        ab = self.dot(a, b, axis)
        aa = tf.sqrt(self.dot(a, a, axis))
        bb = tf.sqrt(self.dot(b, b, axis))
        return ab / (aa * bb + 1e-7)

    def define_optimizer(self):
        true_pw = pairwise_sub(self.true_scores, name='true_pw')
        ones_upper_tri = tf.matrix_band_part(tf.ones_like(true_pw), 0, -1, name='ones_upper_tri')
        true_sign = tf.sign(true_pw * ones_upper_tri, name='true_sign')
        pred_pw = pairwise_sub(self.pred_scores, name='pred_pw')
        margin_pw = tf.maximum(1. - pred_pw * true_sign, 0., name='margin_pw')
        margin_pw_drop = tf.layers.dropout(margin_pw, rate=0.5, training=self.is_train)
        margin_loss = tf.reduce_sum(margin_pw_drop, name='margin_loss')
        self.total_loss = margin_loss
        opt = tf.train.AdamOptimizer(learning_rate=self.lr, name='adam_opt')
        self.train_op = opt.minimize(self.total_loss, name='train_op')
