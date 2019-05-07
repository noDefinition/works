from cqa.mee import K
from utils.deep.layers import *
from .funcs import *


# noinspection PyAttributeOutsideInit
class B1(object):
    def __init__(self, args):
        self.lr = args[K.lr]
        self.x_init = tf.contrib.layers.xavier_initializer()

    def build(self, user_embed):
        self.define_user_embed(user_embed)
        self.define_inputs()
        self.forward()
        self.define_optimizer()

    def define_user_embed(self, user_embed):
        shape = self.num_u, self.dim_u = user_embed.shape
        init = tf.constant_initializer(user_embed)
        # init = tf.truncated_normal_initializer(0., 2.)
        self.user_embed = tf.get_variable('user_embed', initializer=init, shape=shape)

    def define_inputs(self):
        a_num = None
        # fea_len = 1024
        fea_len = 768
        self.a_pf16 = tf.placeholder(f16, shape=(a_num, fea_len))
        self.uint_seq = tf.placeholder(i32, shape=(a_num,), name='uint_seq')
        self.vote_seq = tf.placeholder(i32, shape=(a_num,), name='vote_seq')
        lookup = tf.nn.embedding_lookup
        self.a_pool = tf.cast(self.a_pf16, f32, name='a_pool')
        self.u_lkup = lookup(self.user_embed, self.uint_seq, name='u_lkup')  # (bs, dw)
        self.true_scores = tf.cast(self.vote_seq, dtype=f32, name='true_scores')

    def forward(self):
        au_cat = tf.concat([self.a_pool, self.u_lkup], axis=-1, name='au_cat')
        uas = [(256, tanh), (512, tanh), (1, None)]
        preds = instant_denses(au_cat, uas, name='pred_score', w_init=self.x_init)
        self.pred_scores = tf.squeeze(preds, axis=1, name='pred_scores')

    def define_optimizer(self):
        true_pw = pairwise_sub(self.true_scores, name='true_pw')
        pred_pw = pairwise_sub(self.pred_scores, name='pred_pw')
        ones_upper_tri = tf.matrix_band_part(tf.ones_like(true_pw), 0, -1, name='ones_upper_tri')
        true_sign = tf.sign(true_pw * ones_upper_tri, name='true_sign')
        margin_pw = tf.maximum(1. - pred_pw * true_sign, 0., name='margin_pw')
        margin_loss = tf.reduce_sum(margin_pw, name='margin_loss')
        self.total_loss = margin_loss
        opt = tf.train.AdamOptimizer(learning_rate=self.lr, name='adam_opt')
        grads_vars = opt.compute_gradients(self.total_loss)
        self.train_op = opt.apply_gradients(grads_vars, name='train_op')

    """ runtime below """

    def set_session(self, sess):
        self.sess = sess

    def get_fd(self, apool, uints, votes=None):
        fd = {self.a_pf16: apool, self.uint_seq: uints}
        if votes is not None:
            fd[self.vote_seq] = votes
        return fd

    def train_step(self, apool, uints, votes, *args, **kwargs):
        fd = self.get_fd(apool, uints, votes)
        self.sess.run(self.train_op, feed_dict=fd)

    def predict(self, apool, uints):
        fd = self.get_fd(apool, uints)
        return self.sess.run(self.pred_scores, feed_dict=fd)

    def save(self, file):
        tf.train.Saver().save(self.sess, file)

    def load(self, file):
        tf.train.Saver().restore(self.sess, file)
