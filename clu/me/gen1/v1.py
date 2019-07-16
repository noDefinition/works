from utils.deep.layers import *
from clu.me import *


# noinspection PyAttributeOutsideInit
class V1:
    """  multi-word multi-cluster gen2 (submit version) """
    file = __file__

    def __init__(self, args):
        self.m_dim = args[C.md]
        self.scale = args[C.sc]
        self.n_num = args[C.ns]
        self.margin = args[C.mgn]
        self.l_reg = [args[C.l1], args[C.l2], args[C.l3], args[C.l4]]
        self.pc_probs = self.loss = None

    def define_embed(self, w_init, c_init, w_train=True, c_train=True):
        assert w_init is not None and c_init is not None
        self.v_size, w_dim = w_init.shape
        self.c_num, c_dim = c_init.shape
        self.e_dim = w_dim
        assert w_dim == c_dim
        with tf.variable_scope('embeds'):
            self.w_embed = parse_variable(w_init, trainable=w_train, name='w_embed')
            self.c_embed = parse_variable(c_init, trainable=c_train, name='c_embed')

    def define_inputs(self):
        self.dropout = tf.placeholder_with_default(1., (), name='dropout')
        ph, lk = tf.placeholder, tf.nn.embedding_lookup
        self.p_seq = ph(tf.int32, (None, None), name='p_seq')
        self.n_seqs = [ph(tf.int32, (None, None), name='n_seq_{}'.format(i))
                       for i in range(self.n_num)]
        with tf.name_scope('lookup'):
            self.p_emb = lk(self.w_embed, self.p_seq, name='p_emb')
            self.n_embs = [lk(self.w_embed, n, name='n_emb_{}'.format(i))
                           for i, n in enumerate(self.n_seqs)]

    def define_denses(self):
        ed, sc = self.e_dim, self.scale
        with tf.variable_scope('define_dense'):
            self.D_r = MyDense.normal(ed, ed, sc, name='D_r')
            self.W_1 = MyDense.normal(ed, ed, sc, name='W_1')
            self.W_2 = MyDense.normal(ed, ed, sc, name='W_2')
            self.W_3 = MyDense.normal(ed, ed, sc, name='W_3')
            self.Q = parse_variable(normal((1, 1, ed), sc), name='Q')
            self.W_doc = [self.W_1, self.W_2, self.W_3]

    def define_optimizer(self):
        optimizer = tf.train.AdamOptimizer(tf.train.exponential_decay(
            learning_rate=4e-3, global_step=0, decay_steps=100, decay_rate=0.95, staircase=True))
        grads_and_vars = optimizer.compute_gradients(self.loss)
        self.train_op = optimizer.apply_gradients(grads_and_vars)

    def clu_word_sim_loss(self, c_probs, c_embed, d_embed):
        # batch_size * clu_num * embed_dim
        clu_weight_expand = expand_dims(c_probs, axes=-1)
        # 1 * clu_num * embed_dim
        clu_embed_expand = expand_dims(c_embed, axes=0)
        # batch_size * token_num * 1 * embed_dim
        doc_embed_expand = expand_dims(d_embed, axes=2)
        # batch_size * 1 * clu_num * embed_dim
        clu_apply_expand = expand_dims(element_wise(clu_weight_expand, clu_embed_expand), axes=1)
        # batch_size * token_num * clu_num * embed_dim
        clu_word_loss = tf.reduce_mean(element_wise(clu_apply_expand, doc_embed_expand))
        return clu_word_loss

    def doc_embed(self, w_embed):
        # batch_size * token_num * embed_dim | batch_size/1 * 1 * embed_dim
        w_proj = self.W_1.apply(w_embed)
        q_proj = self.W_2.apply(self.Q)
        # batch_size * token_num * embed_dim
        wq_score = self.W_3.apply(tf.nn.sigmoid(w_proj + q_proj))
        wq_alpha = tf.nn.softmax(wq_score, axis=1)
        wq_apply = element_wise(wq_alpha, w_embed)
        # batch_size * embed_dim
        d_mdatt = tf.reduce_sum(wq_apply, axis=1, keepdims=False)
        return wq_alpha, wq_apply, d_mdatt

    def forward(self):
        l1, l2, l3, l4 = self.l_reg
        with tf.name_scope('get_c_embed'):
            c_embed = self.c_embed
        with tf.name_scope('doc_embed'):
            # batch_size * embed_dim
            p_wq_alpha, p_wq_apply, p_doc = self.doc_embed(self.p_emb)
            # (neg sample size) * embed_dim
            n_doc = tf.concat([self.doc_embed(n)[-1] for n in self.n_embs], axis=0)
        with tf.name_scope('clu_assign'):
            # batch_size * clu_num
            c_scores = tensor_dot(p_doc, tf.transpose(c_embed))
            c_probs = tf.nn.softmax(c_scores, axis=1)
            # batch_size * embed_dim
            r_doc = tensor_dot(c_probs, c_embed)
            r_doc = self.D_r.apply(r_doc)

        # with tf.name_scope('cluster_word_closer'):
        #     pointwise = self.clu_word_sim_loss(c_probs, c_embed, p_wq_apply)
        with tf.name_scope('losses'):
            c_embed, p_doc, r_doc, n_doc = l2_norm_tensors(c_embed, p_doc, r_doc, n_doc)
            'restore and positive similarity'
            # batch_size * 1
            pr_sim = inner_dot(p_doc, r_doc, keepdims=True, name='pr_sim')
            pointwise = tf.reduce_mean(pr_sim)
            'restore and negative similarity'
            # batch_size * (neg sample size)
            rn_sim = tensor_dot(r_doc, tf.transpose(n_doc))
            # batch_size * 1
            # rn_sim_v = tf.reduce_mean(rn_sim, axis=1, keepdims=True)
            'gen2 cluster mutual similarity'
            # batch_size * batch_size
            # c_sim = tf.matmul(c_embed, tf.transpose(c_embed))
            # c_sim_loss = tf.nn.l2_loss(c_sim - tf.diag(tf.diag_part(c_sim)))
            'regularization losses'
            ws = [self.W_1, self.W_2, self.W_3]
            reg_loss = sum(w.get_norm(order=2, add_bias=False) for w in ws)
            'total gen2'
            pairwise = tf.reduce_mean(tf.maximum(0., self.margin - pr_sim + rn_sim))
            loss = pairwise
            # gen2 += (c_sim_loss * l2)
            loss -= (pointwise * l3)
            loss += (reg_loss * l4)

        self.pc_probs = c_probs
        self.loss = loss
        tf.summary.scalar(name='gen2', tensor=loss)
        tf.summary.histogram(name='rn_sim', values=rn_sim, family='sim')
        tf.summary.histogram(name='pr_sim', values=pr_sim, family='sim')

    def build(self, w_embed, c_embed):
        self.define_embed(w_embed, c_embed)
        self.define_denses()
        self.define_inputs()
        self.forward()
        self.define_optimizer()

    """ runtime below """

    def set_session(self, sess):
        self.sess = sess

    def get_fd_by_batch(self, p_batch, n_batches=None, dropout=1.):
        p_seq = [d.tokenids for d in p_batch]
        pairs = [(self.p_seq, p_seq), (self.dropout, dropout)]
        if n_batches is not None:
            n_seqs = [[d.tokenids for d in n_batch] for n_batch in n_batches]
            pairs += list(zip(self.n_seqs, n_seqs))
        return dict(pairs)

    def train_step(self, sess, fd, epoch, max_epoch, batch_id, max_batch_id):
        sess.run(self.train_op, feed_dict=fd)

    def predict(self, sess, fd):
        return np.argmax(sess.run(self.pc_probs, feed_dict=fd), axis=1).reshape(-1)

    def get_loss(self, sess, fd):
        return round(float(sess.run(self.loss, feed_dict=fd)), 4)

    def evaluate(self, sess, batches):
        from utils import au
        clusters, topics = list(), list()
        for p_batch in batches:
            clusters.extend(self.predict(sess, self.get_fd_by_batch(p_batch)))
            topics.extend(d.topic for d in p_batch)
        return au.scores(topics, clusters, au.eval_scores)
