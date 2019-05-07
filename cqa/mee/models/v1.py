from cqa.mee import K
from utils.deep.layers import *
from .funcs import *


# noinspection PyAttributeOutsideInit
class V1(object):
    def __init__(self, args: dict):
        self.lr = args[K.lr]
        self.len_a = self.len_q = None
        self.x_init = tf.contrib.layers.xavier_initializer()
        if args.get(K.reg, 0) > 1e-9:
            print('regularize coef', args[K.reg])
            self.l2_reg = tf.contrib.layers.l2_regularizer(scale=args[K.reg])
        else:
            print('no regularize')
            self.l2_reg = None
        if args.get(K.temp, 0) > 1e-9:
            print('temperature', args[K.temp])
            self.temper = args[K.temp]
        else:
            print('no temperature')
            self.temper = 1.
        if args.get(K.drp, 0) > 1e-9:
            print('dropout', args[K.drp])
            self.dropout = args[K.drp]
        else:
            print('no dropout')
            self.dropout = 0
        self.summaries = list()

    def define_word_embed(self, word_embed):
        shape = self.num_w, self.dim_w = word_embed.shape
        init = tf.constant_initializer(word_embed)
        with tf.variable_scope('word_embed'):
            emb = tf.get_variable(name='w_embed', initializer=init, shape=shape)
            # emb = instant_denses(emb, [(self.dim_w, relu)], name='w_dense')
            pad = tf.zeros(shape=(1, tf.shape(emb)[1]), name='w_padding', dtype=f32)
            self.word_embed = tf.concat([pad, emb], axis=0, name='w_embed_concat')

    def define_user_embed(self, user_embed):
        shape = self.num_u, self.dim_u = user_embed.shape
        init = tf.constant_initializer(user_embed)
        self.user_embed = tf.get_variable('user_embed', initializer=init, shape=shape)

    def define_inputs(self):
        self.is_train = tf.placeholder_with_default(False, shape=())
        ans_num = None
        self.qwid = tf.placeholder(i32, (self.len_q,), name='qwid')
        self.uint_seq = tf.placeholder(i32, (ans_num,), name='uint_seq')
        self.vote_seq = tf.placeholder(i32, (ans_num,), name='vote_seq')
        self.awid_seq = tf.placeholder(i32, (ans_num, self.len_a), name='awid_seq')

        lookup = tf.nn.embedding_lookup
        self.qwid_exp = tf.expand_dims(self.qwid, axis=0, name='qwid_exp')
        self.q_mask = self.get_mask(self.qwid_exp, name='q_mask')  # (1, lq, 1)
        self.a_mask = self.get_mask(self.awid_seq, name='a_mask')  # , (bs, la, 1)
        self.q_lkup = lookup(self.word_embed, self.qwid_exp, name='q_lkup')  # (1, lq, dw)
        self.a_lkup = lookup(self.word_embed, self.awid_seq, name='a_lkup')  # (bs, la, dw)
        self.u_lkup = lookup(self.user_embed, self.uint_seq, name='u_lkup')  # (bs, dw)
        self.true_scores = tf.cast(self.vote_seq, dtype=f32, name='true_scores')

    @staticmethod
    def get_mask(wids, name):
        with tf.name_scope(name):
            mask = tf.cast(tf.not_equal(wids, 0), dtype=f32)  # (bs, tn)
            mask = tf.expand_dims(mask, axis=-1)  # (bs, tn, 1)
        return mask

    def embed_rep(self, lookup, mask, name, k=5):
        with tf.name_scope(name):
            return self.top_k_mean(lookup, k)

    def forward(self):
        a_emb = self.embed_rep(self.a_lkup, self.a_mask, 'a_emb')
        au_cat = tf.concat([a_emb, self.u_lkup], axis=-1)
        uas = [(512, relu), (256, relu), (1, None)]
        preds = instant_denses(au_cat, uas, name='preds')
        self.pred_probs = tf.nn.softmax(tf.squeeze(preds, axis=1))
        self.true_probs = tf.nn.softmax(self.true_scores / self.temper)
        self.pred_scores = self.pred_probs

    def define_optimizer(self):
        total_loss = get_kl_loss(self.true_probs, self.pred_probs)
        if self.l2_reg is not None:
            total_loss += tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
        opt = tf.train.AdamOptimizer(learning_rate=self.lr)
        self.total_loss = total_loss
        self.train_op = opt.minimize(total_loss)

    def build(self, word_embed, user_embed):
        self.define_word_embed(word_embed)
        self.define_user_embed(user_embed)
        self.define_inputs()
        self.forward()
        self.define_optimizer()

    def add_summary(self, tensor, name, family, scalar=False, histogram=False):
        import tensorflow.summary as su
        if not (scalar ^ histogram):
            raise ValueError('must specify one of scalar or histogram')
        if scalar:
            self.summaries.append(su.scalar(name=name, tensor=tensor, family=family))
        elif histogram:
            self.summaries.append(su.histogram(name=name, values=tensor, family=family))

    """ funcs """

    def mask_max(self, t, mask, name='mask_max'):
        with tf.variable_scope(name):
            t = t * mask - (1 - mask) * 1e30
            m = tf.reduce_max(t, -2)
        return m

    def mask_sum(self, t, mask, name='mask_sum'):
        with tf.variable_scope(name):
            s = tf.reduce_sum(t * mask, -2)
        return s

    def mask_mean(self, t, mask, name='mask_mean'):
        with tf.variable_scope(name):
            s = self.mask_sum(t, mask)
            d = tf.reduce_sum(mask, -2)
        return s / d

    def top_k(self, t, k, name='TopK'):
        with tf.variable_scope(name):
            t = tf.transpose(t, [0, 2, 1])
            t = tf.nn.top_k(t, k).values
            t = tf.transpose(t, [0, 2, 1])
        return t

    def top_k_sum(self, t, k, name='TopKSum'):
        with tf.variable_scope(name):
            t = self.top_k(t, k)
            s = tf.reduce_sum(t, -2)
        return s

    def mask_top_k_sum(self, t, mask, k, name='MaskTopKSum'):
        with tf.variable_scope(name):
            t = t * mask - (1 - mask) * 1e30
            t = self.top_k(t, k)
            m = self.top_k(mask, k)
            t = t * m
            s = tf.reduce_sum(t, -2)
        return s

    def top_k_mean(self, t, k, name='TopKMean'):
        with tf.variable_scope(name):
            t = self.top_k(t, k)
            s = tf.reduce_mean(t, -2)
        return s

    def mask_top_k_mean(self, t, mask, k, name='MaskTopKMeam'):
        with tf.variable_scope(name):
            t = t * mask - (1 - mask) * 1e30
            t = self.top_k(t, k)
            m = self.top_k(mask, k)
            s = self.mask_mean(t, m)
        return s

    """ runtime below """

    def set_session(self, sess):
        self.sess = sess

    def get_fd(self, qwid, awids, uints, votes=None):
        fd = {self.qwid: qwid, self.awid_seq: awids, self.uint_seq: uints}
        if votes is not None:
            fd[self.vote_seq] = votes
        return fd

    def predict(self, qwid, awids, uints):
        fd = self.get_fd(qwid, awids, uints)
        return self.sess.run(self.pred_scores, feed_dict=fd)

    def train_step(self, qwid, awids, uints, votes, *args, **kwargs):
        fd = self.get_fd(qwid, awids, uints, votes)
        fd[self.is_train] = True
        self.sess.run(self.train_op, feed_dict=fd)

    def get_loss(self, qwid, awids, uints, votes):
        fd = self.get_fd(qwid, awids, uints, votes)
        loss = self.sess.run(self.total_loss, feed_dict=fd)
        return {'total_loss': loss}

    def get_summary(self, qwid, awids, uints, votes):
        fd = self.get_fd(qwid, awids, uints, votes)
        return self.sess.run(self.summaries, feed_dict=fd)

    def save(self, file):
        tf.train.Saver().save(self.sess, file)

    def load(self, file):
        tf.train.Saver().restore(self.sess, file)
