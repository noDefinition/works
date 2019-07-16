from me.layers import *
from me.gen1 import *


# noinspection PyAttributeOutsideInit,PyMethodMayBeStatic,PyPep8Naming
class G1(V1):
    """ loss加gan，判别器区分正例与重构 """

    def __init__(self, args):
        super(G1, self).__init__(args)
        self.margin = args.mgn
        self.noise = args.nis
        self.pre_train = args.ptn
        self.dis_batch_span = args.dbs
        self.gen_batch_span = args.gbs

    def define_denses(self):
        super(G1, self).define_weight_params()
        sc, ed, md = self.scale, self.e_dim, self.m_dim
        self.D_dis_1 = MyDense.normal(ed, md, sc, name='D_dis_1')
        self.D_dis_2 = MyDense.normal(ed, md, sc, name='D_dis_2')
        self.D_dis_3 = MyDense.normal(ed, md, sc, name='D_dis_3')
        self.D_dis = [self.D_dis_1, self.D_dis_2, self.D_dis_3]

    def doc_embed(self, w_embed):
        w, q = self.W_1(w_embed, name='get_w'), self.W_2(self.Q, name='get_q')
        wq_score = self.W_3(tf.nn.sigmoid(w + q), name='get_wq_score')
        wq_alpha = tf.nn.softmax(wq_score, axis=1, name='get_wq_alpha')
        return element_wise(wq_alpha, w_embed, reduce='sum', axis=1, keepdims=False)

    def get_original_cpn(self):
        with tf.name_scope('get_c_emb'):
            c = self.c_embed
        with tf.name_scope('get_p_emb'):
            p = self.doc_embed(self.p_emb)
        with tf.name_scope('get_n_emb'):
            n = tf.concat([self.doc_embed(e) for e in self.n_embs], axis=0)
        return c, p, n

    def forward(self):
        l1, l2, l3, l4 = self.l_reg
        margin, noise = self.margin, self.noise
        c, p, n = self.get_original_cpn()

        with tf.name_scope('get_pc_probs'):
            pc_score = tensor_dot(p, tf.transpose(c), name='pc_score')
            pc_probs = tf.nn.softmax(pc_score, axis=1, name='pc_probs')
        with tf.name_scope('get_p_res'):
            r = tensor_dot(pc_probs, c)
            r = self.D_r(r)
            # r = tf.nn.relu(r)

        with tf.name_scope('l2norm_prn'):
            p_norm, r_norm, n_norm = l2_norm_tensors(p, r, n)
        # p_norm, r_norm, n_norm = apply_tensors(tf.nn.tanh, p, r, n)
        # with tf.name_scope('add_noise'):
        #     if noise > 1e-6:
        #         p_norm += tf.random_normal(tf.shape(p_norm), mean=0., stddev=noise)
        #         r_norm += tf.random_normal(tf.shape(r_norm), mean=0., stddev=noise)

        with tf.name_scope('discriminate'):
            D_p, D_r = self.D_dis_1(p_norm), self.D_dis_2(r_norm)
            D_pn, D_rn = l2_norm_tensors(D_p, D_r)
            D_pr_sim = inner_dot(D_pn, D_rn)
            # pr_score_D = self.D_dis_3(tf.nn.sigmoid(D_p + D_r))
            # D_pr_sim = tf.nn.tanh(pr_score_D)
            with tf.name_scope('gen2'):
                loss_D = tf.reduce_mean(D_pr_sim)

        with tf.name_scope('generate'):
            # batch_size * 1
            G_pr_sim = inner_dot(p_norm, r_norm, keepdims=True)
            # batch_size * (neg sample size)
            G_pn_sim = tensor_dot(p_norm, tf.transpose(n_norm))
            G_pn_sim_v = tf.reduce_mean(G_pn_sim, axis=1, keepdims=True)
            # G_rn_sim = tensor_dot(r_norm, tf.transpose(n_norm))
            # G_rn_sim_v = tf.reduce_mean(G_rn_sim, axis=1, keepdims=True)
            hinge = tf.maximum(0., margin - G_pr_sim + G_pn_sim_v * l1)
            with tf.name_scope('gen2'):
                reg_G = sum([d.get_norm(order=2) for d in self.W_doc]) * l4
                loss_G_pre = tf.reduce_mean(hinge) + reg_G
                loss_G = loss_G_pre - loss_D * l3

        self.pc_probs = pc_probs
        self.loss_D = loss_D
        self.loss_G_pre = loss_G_pre
        self.loss = self.loss_G = loss_G
        with tf.name_scope('register'):
            tf.summary.histogram(name='p_norm', values=p_norm, family='doc')
            tf.summary.histogram(name='r_norm', values=r_norm, family='doc')
            tf.summary.histogram(name='n_norm', values=n_norm, family='doc')

            tf.summary.histogram(name='G_pr_sim', values=G_pr_sim, family='G_sim')
            tf.summary.histogram(name='G_pn_sim', values=G_pn_sim, family='G_sim')
            # tf.summary.histogram(name='G_rn_sim', values=G_rn_sim, family='G_sim')
            tf.summary.histogram(name='D_pr_sim', values=D_pr_sim, family='D_sim')

            tf.summary.histogram(name='D_p', values=D_p, family='D_out')
            tf.summary.histogram(name='D_r', values=D_r, family='D_out')

            tf.summary.scalar(name='loss_G_pre', tensor=loss_G_pre)
            tf.summary.scalar(name='loss_G', tensor=loss_G)
            tf.summary.scalar(name='loss_D', tensor=loss_D)

    def define_optimizer(self):
        kw_d = dict(learning_rate=2e-3, global_step=0, decay_steps=50, decay_rate=0.95,
                    staircase=True)
        kw_g = dict(learning_rate=3e-3, global_step=0, decay_steps=100, decay_rate=0.99,
                    staircase=True)
        vars_d = get_trainable(*self.D_dis)
        vars_g = list(set(tf.trainable_variables()).difference(set(vars_d)))
        with tf.name_scope('d_optimize'):
            self.train_d_pre = get_train_op(kw_d, self.loss_D, vars_d, name='train_d_pre')
            self.train_d = get_train_op(kw_d, self.loss_D, vars_d, name='train_d')
        with tf.name_scope('g_optimize'):
            self.train_g_pre = get_train_op(kw_g, self.loss_G_pre, vars_g, name='train_g_pre')
            self.train_g = get_train_op(kw_g, self.loss_G, vars_g, name='train_g')

    def train_step(self, sess, fd, epoch, max_epoch, batch, max_batch):
        ptn, dbs = self.pre_train, self.dis_batch_span
        can_train_d = ptn < max_epoch and batch % dbs == 0
        if epoch < ptn:
            sess.run(self.train_g_pre, feed_dict=fd)
            if can_train_d:
                sess.run(self.train_d_pre, feed_dict=fd)
        else:
            sess.run(self.train_g, feed_dict=fd)
            if can_train_d:
                sess.run(self.train_d, feed_dict=fd)
