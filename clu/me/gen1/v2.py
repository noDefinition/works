from tensorflow.distributions import Categorical

from me.gen1.v1 import *


# noinspection PyAttributeOutsideInit,PyMethodMayBeStatic,PyPep8Naming
class V2(V1):
    """ 生成器采样负例, 用 REINFORCE 更新生成器 """
    file = __file__
    
    def define_denses(self):
        super(V2, self).define_weight_params()
        sc = self.scale
        # self.D_dis_1 = Dense.normal(self.e_dim, self.m_dim, sc, activation=tf.nn.relu)
        # self.D_dis_2 = Dense.normal(self.e_dim, self.m_dim, sc, activation=tf.nn.relu)
        # self.D_dis_3 = Dense.normal(self.e_dim, self.m_dim, sc, activation=tf.nn.relu)
        self.D_gen_p = MyDense.normal(self.e_dim, self.e_dim, sc, activation=tf.nn.relu)
        self.D_gen_r = MyDense.normal(self.e_dim, self.e_dim, sc, activation=tf.nn.relu)
        self.D_gen_n = MyDense.normal(self.e_dim, self.e_dim, sc, activation=tf.nn.relu)
    
    # def doc_embed(self, w_embed):
    #     return tf.reduce_mean(w_embed, axis=1, keepdims=False)
    
    def doc_embed(self, w_embed):
        w, q = self.W_1.apply(w_embed), self.W_2.apply(self.Q)
        wq_score = self.W_3.apply(tf.nn.sigmoid(w + q))
        wq_alpha = tf.nn.softmax(wq_score, axis=1)
        wq_apply = element_wise(wq_alpha, w_embed)
        return tf.reduce_sum(wq_apply, axis=1, keepdims=False)
    
    def forward(self, *hyperparams):
        l1, l2, l3, l4, top_k = hyperparams
        
        c_embed = self.c_embed
        with tf.name_scope('doc_embed'):
            p_doc = self.doc_embed(self.p_emb)
            n_doc = tf.concat([self.doc_embed(n) for n in self.n_embs], axis=0)
        with tf.name_scope('clu_assign'):
            # batch_size * clu_num
            c_score = tensor_dot(p_doc, tf.transpose(c_embed))
            c_probs = tf.nn.softmax(c_score, axis=1)
            # batch_size * embed_dim
            r_doc = tensor_dot(c_probs, c_embed)
            r_doc = self.D_r(r_doc, activation=tf.nn.relu)
        
        with tf.name_scope('transform'):
            # c_embed, p_doc, r_doc, n_doc = l2_norm(c_embed, p_doc, r_doc, n_doc)
            p_doc_D, r_doc_D, n_doc_D = p_doc, r_doc, n_doc
            # p_doc_D, r_doc_D, n_doc_D = self.D_dis_1(p_doc), \
            #                             self.D_dis_2(r_doc), \
            #                             self.D_dis_3(n_doc)
            p_doc_G, r_doc_G, n_doc_G = self.D_gen_p(p_doc), \
                                        self.D_gen_r(r_doc), \
                                        self.D_gen_n(n_doc)
            p_doc_D, r_doc_D, n_doc_D = l2_norm_tensors(p_doc_D, r_doc_D, n_doc_D)
            # p_doc_G, r_doc_G, n_doc_G = l2_norm(p_doc_G, r_doc_G, n_doc_G)
        
        with tf.name_scope('generator'):
            # batch_size * 1
            pr_sim_G = element_wise(p_doc_G, r_doc_G, reduce=tf.reduce_sum, axis=-1, keepdims=True)
            # batch_size * (neg sample size)
            rn_sim_G = tensor_dot(r_doc_G, tf.transpose(n_doc_G))
            rn_prob_G = tf.nn.softmax(rn_sim_G, axis=-1)
            # rn_prob_G = rn_sim_G / tf.reduce_sum(rn_sim_G, axis=-1, keepdims=True)
            # batch_size * top_k
            samples = tf.transpose(Categorical(probs=rn_prob_G).sample(sample_shape=top_k))
            rn_logp_G_sel = tf.log(tf.maximum(1e-20, tf.batch_gather(rn_prob_G, samples)))
        with tf.name_scope('discriminator'):
            # pos & res
            # batch_size * 1
            pr_sim_D = element_wise(p_doc_D, r_doc_D, reduce=tf.reduce_sum, axis=1, keepdims=True)
            # pos & neg
            # pn_sim_D = tensor_dot(p_doc_D, tf.transpose(n_doc_D))
            # pn_sim_d_v = tf.reduce_mean(pn_sim_d, axis=1, keepdims=True)
            # pn_sim_d_sel = tf.batch_gather(pn_sim_d, samples)
            # pn_sim_d_sel_v = tf.reduce_mean(pn_sim_d_sel, axis=1, keepdims=True)
            # res & neg
            rn_sim_D = tensor_dot(r_doc_D, tf.transpose(n_doc_D))
            rn_sim_D_avg = tf.reduce_mean(rn_sim_D, axis=1, keepdims=True)
            rn_sim_D_sel = tf.batch_gather(rn_sim_D, samples)
            rn_sim_D_sel_avg = tf.reduce_mean(rn_sim_D_sel, axis=1, keepdims=True)
        
        margin = 1.
        with tf.name_scope('pre_losses'):
            loss_D_pre = tf.reduce_mean(tf.maximum(0., margin - pr_sim_D + rn_sim_D * l1))
            self.loss_G_pre = tf.reduce_mean(pr_sim_G - tf.reduce_mean(rn_sim_G, axis=1, keepdims=True))
        with tf.name_scope('losses'):
            loss_D = tf.reduce_mean(tf.maximum(0., margin - pr_sim_D + rn_sim_D_sel * l1))
            self.loss_G = -tf.reduce_mean(tf.multiply(rn_logp_G_sel, rn_sim_D_sel))
        with tf.name_scope('add_reg'):
            reg_loss = sum([w.get_norm(order=1) for w in [self.W_1, self.W_2, self.W_3,
                                                          self.D_gen_p, self.D_gen_r, self.D_gen_n]])
            self.loss_D_pre = loss_D_pre + reg_loss
            self.loss_D = loss_D + reg_loss
        
        self.c_probs = c_probs
        with tf.name_scope('register'):
            tf.summary.scalar(name='loss_D_pre_no_reg', tensor=loss_D_pre)
            # tf.summary.scalar(name='loss_d_reg', tensor=self.loss_D)
            # tf.summary.scalar(name='loss_g', tensor=self.loss_G)
            # tf.summary.scalar(name='loss_d_pre', tensor=self.loss_D_pre)
            # tf.summary.scalar(name='loss_g_pre', tensor=self.loss_G_pre)
            tf.summary.histogram(name='pr_sim_G', values=pr_sim_G, family='gen')
            tf.summary.histogram(name='rn_sim_G', values=rn_sim_G, family='gen')
            tf.summary.histogram(name='rn_prob_G', values=rn_prob_G, family='gen')
            tf.summary.histogram(name='pr_sim_D', values=pr_sim_D, family='dis')
            tf.summary.histogram(name='rn_sim_D', values=rn_sim_D, family='dis')
            tf.summary.histogram(name='W_1.w', values=self.W_1.w, family='param')
            tf.summary.histogram(name='W_2.w', values=self.W_2.w, family='param')
            tf.summary.histogram(name='W_3.w', values=self.W_3.w, family='param')
            # tf.summary.histogram(name='w_embed', values=self.w_embed, family='param')
            # tf.summary.histogram(name='c_embed', values=self.c_embed, family='param')
            # tf.summary.histogram(name='D_res.w', values=self.D_res.w, family='param')
            # tf.summary.histogram(name='D_gen_1.w', values=self.D_gen_1.w, family='param')
            # tf.summary.histogram(name='D_gen_2.w', values=self.D_gen_2.w, family='param')
            # tf.summary.histogram(name='D_gen_3.w', values=self.D_gen_3.w, family='param')
            # tf.summary.histogram(name='D_dis_1.w', values=self.D_dis_1.w, family='param')
            # tf.summary.histogram(name='D_dis_2.w', values=self.D_dis_2.w, family='param')
            # tf.summary.histogram(name='D_dis_3.w', values=self.D_dis_3.w, family='param')
    
    def define_optimizer(self, kw_d_g):
        kw_d = dict(learning_rate=2e-3, global_step=0, decay_steps=100, decay_rate=0.95, staircase=True)
        kw_g = dict(learning_rate=2e-3, global_step=0, decay_steps=100, decay_rate=0.95, staircase=True)
        # self.vars_d = get_trainable([self.D_res, self.D_dis_1, self.D_dis_2,
        #                              self.W_1, self.W_2, self.W_3, self.Q]) + \
        #               [self.w_embed, self.c_embed]
        self.vars_g = get_trainable(self.D_gen_p, self.D_gen_r, self.D_gen_n)
        self.vars_d = list(set(tf.trainable_variables()).difference(set(self.vars_g)))
        
        self.train_d_pre = get_train_op(kw_d, self.loss_D_pre, self.vars_d)
        self.train_g_pre = get_train_op(kw_g, self.loss_G_pre, self.vars_g)
        self.train_d = get_train_op(kw_d, self.loss_D, self.vars_d)
        self.train_g = get_train_op(kw_g, self.loss_G, self.vars_g)
    
    def train_step(self, sess, fd, epoch, batch_id):
        if epoch < 5:
            sess.run(self.train_d_pre, feed_dict=fd)
            sess.run(self.train_g_pre, feed_dict=fd)
        else:
            sess.run(self.train_d, feed_dict=fd)
            sess.run(self.train_g, feed_dict=fd)
    
    def get_loss(self, sess, fd):
        return 'Gen:{:6f} Dis:{:6f}'.format(*sess.run([self.loss_G, self.loss_D], feed_dict=fd))
