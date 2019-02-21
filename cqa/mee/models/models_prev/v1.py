from dc import *
from me.layers import *

q_maxlen = 20
a_maxlen = 200
dense = tf.layers.dense


def denses(t, ua_list, name, kernel=None):
    for i, (u, a) in enumerate(ua_list):
        t = dense(t, u, a, kernel_initializer=kernel, name='{}_{}'.format(name, i))
    return t


# noinspection PyAttributeOutsideInit
class V1:
    file = __file__

    def __init__(self, args: dict):
        self.lr = args[lr_]
        self.m_dim = args[md_]
        self.w_init = args[wit_]
        self.w_train = args[wtn_]
        self.text_mode = args[tmd_]

        self.prob_type = args[ptp_]
        self.opt_type = args[otp_]
        self.mask_type = args[mtp_]
        self.final_type = args[ftp_]

        self.normal_initer = tf.random_normal_initializer(mean=0., stddev=args[sc_], dtype=f32)

    def define_user_embed(self, u_init):
        self.n_users, self.u_dim = u_init.shape
        initer = tf.constant_initializer(u_init)
        self.u_embed = tf.get_variable('u_embed', initializer=initer, shape=u_init.shape)

    def define_word_embed(self, w_init):
        self.n_words, self.w_dim = w_init.shape
        initer = {0: self.normal_initer, 1: tf.constant_initializer(w_init)}[self.w_init]
        emb = tf.get_variable(name='w_embed', initializer=initer, shape=w_init.shape)
        emb = denses(emb, [(self.w_dim, relu)], name='map_w', kernel=self.normal_initer)
        pad = tf.zeros(shape=(1, self.w_dim), name='w_pad', dtype=f32)
        self.w_embed = tf.concat([pad, emb], axis=0)

    def define_inputs(self):
        ans_num = None
        self.qwid = tf.placeholder(tf.int32, (q_maxlen,), name='question')
        self.uint_seq = tf.placeholder(tf.int32, (ans_num,), name='u_seq_ints')
        self.vote_seq = tf.placeholder(tf.int32, (ans_num,), name='a_seq_vote')
        self.awid_seq = tf.placeholder(tf.int32, (ans_num, a_maxlen), name='a_seq_wids')

    def lookup_and_mask(self, w_id_seq, name):
        with tf.name_scope(name):
            lkup = tf.nn.embedding_lookup(self.w_embed, w_id_seq)
            mask = tf.cast(tf.not_equal(w_id_seq, 0), f32)
            mask = tf.expand_dims(mask, -1)
        return lkup, mask

    def get_ans_embed(self, w_id_seq, name):
        lkup, mask = self.lookup_and_mask(w_id_seq, name)
        return self.doc_embed(lkup, mask)

    def doc_embed(self, w_lookup_seq, w_mask_seq):
        k = 5
        if self.text_mode == 0:
            o = self.top_k_sum(w_lookup_seq, k)
        elif self.text_mode == 1:
            o = self.top_k_mean(w_lookup_seq, k)
        # elif self.text_mode == 2:
        #     o = self.mask_top_k_sum(w_lookup_seq, w_mask_seq, k)
        # elif self.text_mode == 3:
        #     o = self.mask_top_k_mean(w_lookup_seq, w_mask_seq, k)
        # else:
        #     raise ValueError('unexpected text mode {}'.format(self.text_mode))
        return o

    def get_kl_loss(self, true_probs, pred_probs):
        import tensorflow.distributions as tfd
        true = tfd.Categorical(probs=true_probs + 1e-32)
        pred = tfd.Categorical(probs=pred_probs + 1e-32)
        point_diverge = tfd.kl_divergence(true, pred, allow_nan_stats=False)
        kl_loss = tf.reduce_sum(point_diverge)
        return kl_loss

    def forward(self):
        u_emb = tf.nn.embedding_lookup(self.u_embed, self.uint_seq, name='user_lookup')
        a_lkup, a_mask = self.lookup_and_mask(self.awid_seq, name='a_lookup_mask')
        a_emb = self.doc_embed(a_lkup, a_mask)
        au_rep = tf.concat([a_emb, u_emb], axis=-1)
        ua_list = [(512, relu), (256, relu), (self.w_dim, None)]
        au_rep = denses(au_rep, ua_list, name='au_rep_new')

        mut_sup = tf.matmul(au_rep, tf.transpose(au_rep))
        if self.mask_type == 0:
            mut_mask = - 1e32 * tf.eye(tf.shape(mut_sup)[0])
        elif self.mask_type == 1:
            mut_mask = - tf.diag(tf.diag_part(mut_sup))
        else:
            mut_mask = None
        mut_att = tf.nn.softmax(mut_sup + mut_mask, axis=1)
        au_res = tf.matmul(mut_att, au_rep)
        mut_probs = tf.reduce_mean(mut_att, axis=0, keepdims=False)

        if self.final_type == 0:
            au_final = au_res
        elif self.final_type == 1:
            au_final = tf.concat([au_rep, au_res, au_rep * au_res], axis=1)
        else:
            au_final = None
        ua_list = [(512, relu), (256, relu), (1, None)]
        res_scr = denses(au_final, ua_list, name='res_scr')
        res_probs = tf.squeeze(tf.nn.softmax(res_scr), axis=1)

        self.true_probs = tf.nn.softmax(tf.cast(tf.maximum(self.vote_seq, 0), dtype=f32))
        self.pred_probs = {0: mut_probs, 1: res_probs}[self.prob_type]
        self.kl_loss = self.get_kl_loss(self.true_probs, self.pred_probs)

    def define_optimizer(self):
        opt_cls = {0: tf.train.AdagradOptimizer, 1: tf.train.AdamOptimizer}[self.opt_type]
        opt = opt_cls(learning_rate=self.lr)
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
