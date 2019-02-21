from cqa.mee import *
from utils.deep.layers import *


# q_maxlen = 20
# a_maxlen = 200


def denses(t, uas, name, kernel=None):
    for i, (u, a) in enumerate(uas):
        t = tf.layers.dense(t, u, a, kernel_initializer=kernel, name='{}_{}'.format(name, i))
    return t


def get_kl_loss(true_probs, pred_probs):
    import tensorflow.distributions as tfd
    true = tfd.Categorical(probs=true_probs + 1e-32)
    pred = tfd.Categorical(probs=pred_probs + 1e-32)
    pointwise_diverge = tfd.kl_divergence(true, pred, allow_nan_stats=False)
    kl_loss = tf.reduce_sum(pointwise_diverge)
    return kl_loss


def mask_diagly(t, value=-1e32):
    return t + value * tf.eye(num_rows=tf.shape(t)[0])


# noinspection PyAttributeOutsideInit
class V1:
    file = __file__

    def __init__(self, args: dict):
        self.lr = args[lr_]
        self.m_dim = args[md_]
        self.w_init = args[winit_]
        self.w_train = args[wtrn_]
        self.opt_type = args[otp_]
        self.x_init = tf.contrib.layers.xavier_initializer()
        # self.n_init = tf.random_normal_initializer(mean=0., stddev=args[sc_])

    def define_user_embed(self, user_embed):
        self.n_users, self.u_dim = user_embed.shape
        initer = tf.constant_initializer(user_embed)
        self.user_embed = tf.get_variable('u_embed', initializer=initer, shape=user_embed.shape)

    def define_word_embed(self, word_embed):
        self.n_words, self.w_dim = word_embed.shape
        self.n_words += 1
        # initer = [self.x_init, tf.constant_initializer(w_embed)][self.w_init]
        initer = tf.constant_initializer(word_embed)
        emb = tf.get_variable(name='w_embed', initializer=initer, shape=word_embed.shape)
        # emb = denses(emb, [(self.w_dim, relu)], name='map_w', kernel=self.x_init)
        pad = tf.zeros(shape=(1, tf.shape(emb)[1]), name='w_pad', dtype=f32)
        self.word_embed = tf.concat([pad, emb], axis=0)

    def define_inputs(self):
        ans_num = q_maxlen = a_maxlen = None
        self.qwid = tf.placeholder(i32, (q_maxlen,), name='qwid')
        self.uint_seq = tf.placeholder(i32, (ans_num,), name='uint_seq')
        self.vote_seq = tf.placeholder(i32, (ans_num,), name='vote_seq')
        self.awid_seq = tf.placeholder(i32, (ans_num, a_maxlen), name='awid_seq')

        self.qwid_exp = tf.expand_dims(self.qwid, axis=0)
        self.u_emb = tf.nn.embedding_lookup(self.user_embed, self.uint_seq, name='u_lookup')

    def lookup_and_mask(self, wid_seq, name):
        with tf.name_scope(name):
            lookup = tf.nn.embedding_lookup(self.word_embed, wid_seq)
            mask = tf.expand_dims(tf.cast(tf.not_equal(wid_seq, 0), dtype=f32), axis=-1)
        return lookup, mask

    def embed_from_lookup(self, lookup, mask, name):
        with tf.name_scope(name):
            return self.top_k_mean(lookup, k=5)
            # return self.mask_top_k_mean(lookup, mask, 5)
            #     k = 5
            #     if self.text_mode == 0:
            #         o = self.top_k_sum(w_lookup_seq, k)
            #     elif self.text_mode == 1:
            #         o = self.top_k_mean(w_lookup_seq, k)
            #     elif self.text_mode == 2:
            #         o = self.mask_top_k_sum(w_lookup_seq, w_mask_seq, k)
            #     elif self.text_mode == 3:
            #         o = self.mask_top_k_mean(w_lookup_seq, w_mask_seq, k)
            #     else:
            #         raise ValueError('unexpected text mode {}'.format(self.text_mode))
            #     return o

    def get_doc_embed(self, wid_seq, name):
        lookup, mask = self.lookup_and_mask(wid_seq, name + '_lookup_mask')
        embed = self.embed_from_lookup(lookup, mask, name + '_embed')
        return embed

    def forward(self):
        u_emb = self.u_emb
        a_emb = self.get_doc_embed(self.awid_seq, name='a_emb')
        au_cat = tf.concat([a_emb, u_emb], axis=-1)
        pred_score = denses(au_cat, [(512, relu), (256, relu), (1, None)], name='pred_score')
        pred_probs = tf.nn.softmax(tf.squeeze(pred_score, axis=1))

        self.pred_probs = pred_probs
        self.true_probs = tf.nn.softmax(tf.cast(tf.maximum(self.vote_seq, 0), dtype=f32))
        self.kl_loss = get_kl_loss(self.true_probs, self.pred_probs)

    def define_optimizer(self):
        # opt_class = [tf.train.AdagradOptimizer, tf.train.AdamOptimizer][self.opt_type]
        opt = tf.train.AdamOptimizer(learning_rate=self.lr)
        var = tf.trainable_variables()
        grads_vars = opt.compute_gradients(self.kl_loss, var_list=var)
        self.train_op = opt.apply_gradients(grads_vars)

    def build(self, u_embed, w_embed):
        self.define_user_embed(u_embed)
        self.define_word_embed(w_embed)
        self.define_inputs()
        self.forward()
        self.define_optimizer()

    """ runtime below """

    def get_fd(self, qwid, awid_seq, uint_seq, vote_seq=None):
        fd = {self.qwid: qwid, self.awid_seq: awid_seq, self.uint_seq: uint_seq}
        if vote_seq is not None:
            fd[self.vote_seq] = vote_seq
        return fd

    def predict(self, sess, qwid, awid_seq, uint_seq):
        fd = self.get_fd(qwid, awid_seq, uint_seq)
        return sess.run(self.pred_probs, feed_dict=fd)

    def train_step(self, sess, qwid, awid_seq, uint_seq, vote_seq):
        fd = self.get_fd(qwid, awid_seq, uint_seq, vote_seq)
        _, loss = sess.run([self.train_op, self.kl_loss], feed_dict=fd)
        return loss

    ''' funcs begin '''

    def mask_max(self, t, mask):
        with tf.variable_scope('mask_max'):
            t = t * mask - (1 - mask) * 1e30
            m = tf.reduce_max(t, -2)
        return m

    def mask_sum(self, t, mask):
        with tf.variable_scope('mask_sum'):
            s = tf.reduce_sum(t * mask, -2)
        return s

    def mask_mean(self, t, mask):
        with tf.variable_scope('mask_mean'):
            s = self.mask_sum(t, mask)
            d = tf.reduce_sum(mask, -2)
        return s / d

    def top_k(self, t, k):
        with tf.variable_scope('TopK'):
            t = tf.transpose(t, [0, 2, 1])
            t = tf.nn.top_k(t, k).values
            t = tf.transpose(t, [0, 2, 1])
        return t

    def top_k_sum(self, t, k):
        with tf.variable_scope('TopKSum'):
            t = self.top_k(t, k)
            s = tf.reduce_sum(t, -2)
        return s

    def mask_top_k_sum(self, t, mask, k):
        with tf.variable_scope('MaskTopKSum'):
            t = t * mask - (1 - mask) * 1e30
            t = self.top_k(t, k)
            m = self.top_k(mask, k)
            t = t * m
            s = tf.reduce_sum(t, -2)
        return s

    def top_k_mean(self, t, k):
        with tf.variable_scope('TopKMean'):
            t = self.top_k(t, k)
            s = tf.reduce_mean(t, -2)
        return s

    def mask_top_k_mean(self, t, mask, k):
        with tf.variable_scope('MaskTopKMeam'):
            t = t * mask - (1 - mask) * 1e30
            t = self.top_k(t, k)
            m = self.top_k(mask, k)
            s = self.mask_mean(t, m)
        return s
