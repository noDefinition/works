import tensorflow as tf

from cqa.baselinee import J


# noinspection PyAttributeOutsideInit
class B1:
    def __init__(self, args: dict):
        self.lr = args[J.lr]
        self.t_mode = 1
        self.x_init = tf.contrib.layers.xavier_initializer()
        self.l2_reg = None
        if J.reg in args and args[J.reg] > 1e-8:
            self.l2_reg = tf.contrib.layers.l2_regularizer(scale=args[J.reg])

    def user_emb(self, u):
        with tf.variable_scope('UserEmb'):
            shape = self.user_vec.shape
            # initer = self.x_init
            initer = tf.constant_initializer(self.user_vec)
            emb_u = tf.get_variable(name='emb_u', initializer=initer, shape=shape)
            o = tf.nn.embedding_lookup(emb_u, u)
        return o

    def text_emb(self, t):
        with tf.variable_scope('TextEmb'):
            shape = self.word_vec.shape
            initer = tf.constant_initializer(self.word_vec)
            emb_w = tf.get_variable(name='emb_w', initializer=initer, shape=shape)
            emb_w = tf.layers.dense(emb_w, self.w_dim, name='MapWordVec')
            pad = tf.zeros(shape=(1, self.w_dim))
            emb_w = tf.concat([pad, emb_w], axis=0)
        o = tf.nn.embedding_lookup(emb_w, t)
        mask = tf.expand_dims(tf.cast(tf.not_equal(t, 0), tf.float32), -1)
        return o, mask

    def top_layers(self, t, name='TopLayer'):
        fc = (512, 256)
        k_init, k_reg = None, self.l2_reg
        with tf.variable_scope(name):
            o = t
            for i, d in enumerate(fc):
                o = tf.layers.dense(o, d, tf.nn.relu, name='dense_{}'.format(i),
                                    kernel_initializer=k_init, bias_initializer=k_init,
                                    kernel_regularizer=k_reg, bias_regularizer=k_reg)
                # o = tf.layers.batch_normalization(o)
            o = tf.layers.dense(o, 1, name='dense_{}'.format(len(fc)),
                                kernel_initializer=k_init, bias_initializer=k_init,
                                kernel_regularizer=k_reg, bias_regularizer=k_reg)
            # o = tf.layers.batch_normalization(o)
            o = tf.reshape(o, (-1,))
        return o

    def rep(self, ques, answ, user):
        a, a_mask = self.text_emb(answ)  # (bs, la, dw)
        a_rep = self.text_rep(a, a_mask)
        u_rep = self.user_emb(user)
        return tf.concat([a_rep, u_rep], axis=-1)

    def define_word_vec(self, word_vec):
        self.word_vec = word_vec
        self.n_word, self.w_dim = word_vec.shape

    def define_user_vec(self, user_vec):
        self.user_vec = user_vec
        self.n_user, self.u_dim = user_vec.shape

    def define_inputs(self, len_q=None, len_a=None):
        self.is_train = tf.placeholder_with_default(False, shape=())
        n_answ = None
        self.inp_q0 = tf.placeholder(tf.int32, (len_q,), name='ques_0')
        self.inp_a0 = tf.placeholder(tf.int32, (n_answ, len_a), name='answ_0')
        self.inp_u0 = tf.placeholder(tf.int32, (n_answ,), name='user_0')

        self.inp_q1 = tf.placeholder(tf.int32, (len_q,), name='ques_1')
        self.inp_a1 = tf.placeholder(tf.int32, (n_answ, len_a), name='answ_1')
        self.inp_u1 = tf.placeholder(tf.int32, (n_answ,), name='user_1')
        # self.train_inp = [self.inp_a0, self.inp_u0, self.inp_a1, self.inp_u1]
        # self.pred_inp = [self.inp_a0, self.inp_u0]

    def forward(self):
        with tf.variable_scope("train", reuse=tf.AUTO_REUSE):
            score0 = self.top_layers(self.rep(self.inp_q0, self.inp_a0, self.inp_u0))
            score1 = self.top_layers(self.rep(self.inp_q1, self.inp_a1, self.inp_u1))
        diff_score = score0 - score1
        self.loss = tf.reduce_mean(tf.maximum(1 - diff_score, tf.zeros_like(diff_score)))
        self.pred = score0

    def define_optimizer(self):
        final_loss = self.loss
        if self.l2_reg is not None:
            final_loss += tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
        optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
        self.minimize = optimizer.minimize(final_loss)

    def build(self, user_vec, word_vec, len_q, len_a):
        self.define_user_vec(user_vec)
        self.define_word_vec(word_vec)
        self.define_inputs(len_q, len_a)
        self.forward()
        self.define_optimizer()

    """ runtime below """

    def set_session(self, sess):
        self.sess = sess

    def get_fd(self, q0, a0, u0, q1=None, a1=None, u1=None):
        fd = {self.inp_a0, self.inp_u0}
        if a1 is not None and u1 is not None:
            fd.update({self.inp_a1: a1, self.inp_u1: u1})
        return fd
        # self.train_inp = [self.inp_a0, self.inp_u0, self.inp_a1, self.inp_u1]
        # return dict(zip(self.train_inp, [a0, u0, a1, u1]))
        # else:
        #     return dict(zip(self.pred_inp, [a0, u0]))

    def predict(self, q0, a0, u0):
        fd = self.get_fd(None, a0, u0)
        return self.sess.run(self.pred, fd)

    def train_step(self, q0, a0, u0, q1, a1, u1):
        fd = self.get_fd(None, a0, u0, None, a1, u1)
        fd[self.is_train] = True
        _, loss = self.sess.run([self.minimize, self.loss], feed_dict=fd)
        return {'gen2', loss}

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
        k=5
        t = self.top_k_mean(t, k)
        # if self.t_mode == 0:
        #     t = self.top_k_sum(t, k)
        # elif self.t_mode == 1:
        #     t = self.top_k_mean(t, k)
        # elif self.t_mode == 2:
        #     t = self.mask_top_k_sum(t, mask, k)
        # elif self.t_mode == 3:
        #     t = self.mask_top_k_mean(t, mask, k)
        return t

    ''' ****************************** funcs over ******************************'''
