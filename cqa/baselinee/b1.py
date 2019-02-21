import tensorflow as tf

from cqa.baselinee import lr_


# q_maxlen = 20
# a_maxlen = 200


# noinspection PyAttributeOutsideInit
class B1:
    def __init__(self, args: dict):
        self.lr = args[lr_]
        # self.m_dim = args[md_]
        self.t_mode = 1
        self.x_init = tf.contrib.layers.xavier_initializer()

    def user_emb(self, u):
        with tf.variable_scope('UserEmb'):
            initer = self.x_init
            u_shape = (self.n_user, self.u_dim)
            emb_u = tf.get_variable(name='emb_u', shape=u_shape, initializer=initer)
            o = tf.nn.embedding_lookup(emb_u, u)
        return o

    def text_emb(self, t):
        with tf.variable_scope('TextEmb'):
            initer = tf.constant_initializer(self.word_vec)
            emb_w = tf.get_variable(name='emb_w', initializer=initer)
            emb_w = tf.layers.dense(emb_w, self.w_dim, name='MapWordVec')
            pad = tf.zeros(shape=(1, self.w_dim))
            emb_w = tf.concat([pad, emb_w], axis=0)
        o = tf.nn.embedding_lookup(emb_w, t)
        mask = tf.expand_dims(tf.cast(tf.not_equal(t, 0), tf.float32), -1)
        return o, mask

    ''' *********************************** funcs begin ***********************************'''

    def mask_max(self, t, mask):
        # (bs, tl, k), (bs, tl)
        with tf.variable_scope('mask_max'):
            t = t * mask - (1 - mask) * 1e30
            m = tf.reduce_max(t, -2)
        return m

    def mask_sum(self, t, mask):
        # (bs, tl, k), (bs, tl)
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
            t = self.top_k(t, k)  # (bs, k, dim_k)
            s = tf.reduce_sum(t, -2)  # (bs, dim_k)
        return s

    def mask_top_k_sum(self, t, mask, k):
        # t:(bs, tl, dim_k), mask: (bs, tl, 1)
        with tf.variable_scope('MaskTopKSum'):
            t = t * mask - (1 - mask) * 1e30
            t = self.top_k(t, k)  # (bs, k, dim_k)
            m = self.top_k(mask, k)  # (bs, k, 1)
            t = t * m
            s = tf.reduce_sum(t, -2)  # (bs, dim_k)
        return s

    def top_k_mean(self, t, k):
        with tf.variable_scope('TopKMean'):
            t = self.top_k(t, k)  # (bs, k, dim_k)
            s = tf.reduce_mean(t, -2)  # (bs, dim_k)
        return s

    def mask_top_k_mean(self, t, mask, k):
        # t:(bs, tl, dim_k), mask: (bs, tl, 1)
        with tf.variable_scope('MaskTopKMeam'):
            t = t * mask - (1 - mask) * 1e30
            t = self.top_k(t, k)  # (bs, k, dim_k)
            m = self.top_k(mask, k)  # (bs, k, 1)
            s = self.mask_mean(t, m)  # (bs, dim_k)
        return s

    def text_rep(self, t, mask):
        if self.t_mode == 0:
            t = self.top_k_sum(t, 5)
        elif self.t_mode == 1:
            t = self.top_k_mean(t, 5)
        elif self.t_mode == 2:
            t = self.mask_top_k_sum(t, mask, 5)
        elif self.t_mode == 3:
            t = self.mask_top_k_mean(t, mask, 5)
        return t

    ''' ****************************** funcs over ******************************'''

    def top_layers(self, t, name='TopLayer'):
        fc = (512, 256)
        with tf.variable_scope(name):
            o = t
            for i, n in enumerate(fc):
                # o = tf.layers.batch_normalization(o)
                o = tf.layers.dense(o, n, activation=tf.nn.relu, name='dense_{}'.format(i))
            # o = tf.layers.batch_normalization(o)
            o = tf.layers.dense(o, 1, name='dense_{}'.format(len(fc)))
            o = tf.reshape(o, (-1,))
        return o

    def rep(self, ques, answ, user):
        a, a_mask = self.text_emb(answ)  # (bs, al, k)
        a_rep = self.text_rep(a, a_mask)
        u_rep = self.user_emb(user)
        return tf.concat([a_rep, u_rep], axis=-1)

    def define_word_vec(self, word_vec):
        self.word_vec = word_vec
        self.n_word, self.w_dim = word_vec.shape

    def define_user_vec(self, user_vec):
        self.user_vec = user_vec
        self.n_user, self.u_dim = user_vec.shape

    def define_inputs(self):
        q_maxlen = a_maxlen = batch_size = None
        self.inp_q0 = tf.placeholder(tf.int32, (q_maxlen,), name='question')
        self.inp_a0 = tf.placeholder(tf.int32, (batch_size, a_maxlen), name='pos_ans')
        self.inp_u0 = tf.placeholder(tf.int32, (batch_size,), name='pos_user')

        self.inp_q1 = tf.placeholder(tf.int32, (q_maxlen,), name='question')
        self.inp_a1 = tf.placeholder(tf.int32, (batch_size, a_maxlen), name='neg_ans')
        self.inp_u1 = tf.placeholder(tf.int32, (batch_size,), name='neg_user')

        self.train_inp = [self.inp_a0, self.inp_u0, self.inp_a1, self.inp_u1]
        self.pred_inp = [self.inp_a0, self.inp_u0]

    def forward(self):
        with tf.variable_scope("train", reuse=tf.AUTO_REUSE):
            score0 = self.top_layers(self.rep(self.inp_q0, self.inp_a0, self.inp_u0))
            score1 = self.top_layers(self.rep(self.inp_q1, self.inp_a1, self.inp_u1))
        diff_score = score0 - score1
        # self.loss = tf.reduce_mean(tf.maximum(1 - diff_score, 0.))
        self.loss = tf.reduce_mean(tf.maximum(1 - diff_score, tf.zeros_like(diff_score)))
        self.pred = score0

    def build(self, user_vec, word_vec):
        # tf.set_random_seed(918762)
        self.define_user_vec(user_vec)
        self.define_word_vec(word_vec)
        self.define_inputs()
        self.forward()
        self.minimize = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss)

    """ runtime below """

    def get_fd(self, q0, a0, u0, q1=None, a1=None, u1=None):
        if a1 is not None and u1 is not None:
            return dict(zip(self.train_inp, [a0, u0, a1, u1]))
        else:
            return dict(zip(self.pred_inp, [a0, u0]))

    def predict(self, sess, q0, a0, u0):
        fd = self.get_fd(None, a0, u0)
        return sess.run(self.pred, fd)

    def train_step(self, sess, q0, a0, u0, q1, a1, u1):
        fd = self.get_fd(None, a0, u0, None, a1, u1)
        _, loss = sess.run([self.minimize, self.loss], feed_dict=fd)
        return loss
