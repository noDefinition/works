from me.gen1.v1 import *


# noinspection PyAttributeOutsideInit
class S1(V1):
    """  使用等长序列(padding) """
    file = __file__

    def define_word_embed(self, w_init, w_train=True):
        w_shape = w_init.shape
        w_initer = tf.constant_initializer(w_init)
        self.v_num, self.w_dim = w_shape
        self.w_embed = tf.get_variable(
            name='w_embed', shape=w_shape, initializer=w_initer, trainable=w_train)

    def define_cluster_embed(self, c_init, c_train=True):
        c_shape = c_init.shape
        c_initer = tf.constant_initializer(c_init)
        self.c_num, self.c_dim = c_shape
        self.c_embed = tf.get_variable(
            name='c_embed', shape=c_shape, initializer=c_initer, trainable=c_train)

    def define_inputs(self):
        self.dropout = tf.placeholder_with_default(1., ())
        self.p_seq = tf.placeholder(tf.int32, (None, None))
        self.p_len = tf.placeholder(tf.int32, (None,))
        self.p_emb = tf.nn.embedding_lookup(self.w_embed, self.p_seq)
        self.n_seq = tf.placeholder(tf.int32, (None, None))
        self.n_len = tf.placeholder(tf.int32, (None,))
        self.n_emb = tf.nn.embedding_lookup(self.w_embed, self.n_seq)

    def doc_embed_max_len(self, w_embed, l_mask):
        w_proj, q_proj = self.W_1.apply(w_embed), self.W_2.apply(self.Q)
        wq_score = self.W_3.apply(tf.nn.sigmoid(w_proj + q_proj))
        # batch_size * max_seq_len
        mask = tf.sequence_mask(l_mask, tf.shape(w_embed)[1], tf.float32)
        mask = (expand_dims(mask, axes=2) - tf.constant(1., tf.float32)) * 1e32
        wq_alpha = tf.nn.softmax(wq_score + mask, axis=1)
        wq_apply = tf.multiply(wq_alpha, w_embed)
        return mask, tf.reduce_sum(wq_apply, axis=1, keepdims=False)

    def forward(self, *hyperparams):
        l1, l2, l3, l4, _ = hyperparams
        c_embed = self.c_embed
        with tf.name_scope('doc_embed'):
            # batch_size * embed_dim
            self.ppp, p_doc = self.doc_embed_max_len(self.p_emb, self.p_len)
            # (neg sample size) * embed_dim
            _, n_doc = self.doc_embed_max_len(self.n_emb, self.n_len)
        with tf.name_scope('clu_assign'):
            # batch_size * embed_dim / clu_num * embed_dim
            # batch_size * clu_num
            c_scores = tensor_dot(p_doc, tf.transpose(c_embed))
            c_probs = tf.nn.softmax(c_scores, axis=1)
            # batch_size * embed_dim
            r_doc = tensor_dot(c_probs, c_embed)
            r_doc = self.D_r.apply(r_doc)
        # with tf.name_scope('cluster_word_get_closer'):
        #     p_cw_loss = self.clu_word_sim_loss(c_weights, c_embed, p_wq_apply)
        with tf.name_scope('losses'):
            c_embed, p_doc, r_doc, n_doc = l2_norm_tensors(c_embed, p_doc, r_doc, n_doc)
            'restore and positive similarity'
            # batch_size * 1
            pr_sim = inner_dot(p_doc, r_doc, keepdims=True)
            # pr_sim_v = element_wise(p_doc, r_doc, reduce=tf.reduce_sum, axis=1, keepdims=True)
            p_cw_loss = tf.reduce_mean(pr_sim)
            'restore and negative similarity'
            # batch_size * (neg sample size)
            rn_sim = tensor_dot(r_doc, tf.transpose(n_doc))
            # batch_size * 1
            rn_sim_v = tf.reduce_mean(rn_sim, axis=1, keepdims=True)
            'gen2 cluster mutual similarity'
            # batch_size * batch_size
            c_sim = tf.matmul(c_embed, tf.transpose(c_embed))
            c_sim_loss = tf.nn.l2_loss(c_sim - tf.diag(tf.diag_part(c_sim)))
            'regularization losses'
            reg_loss = sum([w.get_norm(order=1) for w in [self.W_1, self.W_2, self.W_3]])
            'total gen2'
            loss = tf.reduce_mean(tf.maximum(0.0, 1.0 - pr_sim + rn_sim_v * l1))
            loss += (c_sim_loss * l2)
            loss -= (p_cw_loss * l3)
            loss += (reg_loss * l4)

        self.c_probs = c_probs
        self.loss = loss
        # tf.summary.scalar(name='gen2', tensor=gen2)
        tf.summary.histogram(name='rn_sim_v', values=rn_sim_v, family='sim')
        tf.summary.histogram(name='pr_sim_v', values=pr_sim, family='sim')
        # tf.summary.histogram(name='p_doc', values=p_doc, family='doc')
        # tf.summary.histogram(name='r_doc', values=r_doc, family='doc')
        # tf.summary.histogram(name='n_doc', values=n_doc, family='doc')

    def get_fd_by_batch(self, p_batch, n_batch=None, dropout=1.):
        p_seq = [d.pos_fill for d in p_batch]
        p_len = [len(d.tokenids) for d in p_batch]
        pairs = [(self.p_seq, p_seq), (self.p_len, p_len), (self.dropout, dropout)]
        if n_batch is not None:
            n_seq = [d.neg_fill for d in n_batch]
            n_len = [len(d.tokenids) for d in n_batch]
            pairs += [(self.n_seq, n_seq), (self.n_len, n_len)]
        return dict(pairs)
