from me.gen1.g1 import *


# noinspection PyAttributeOutsideInit,PyMethodMayBeStatic,PyPep8Naming
class G3(G1):
    """ 对抗进入pairwise loss，同样区分正例与重构 """
    file = __file__
    
    def define_denses(self):
        super(G3, self).define_weight_params()
        sc, ed, md = self.scale, self.e_dim, self.m_dim
        relu = tf.nn.relu
        
        self.D_r = MyDense.normal(md, md, sc, name='D_r', activation=relu)
        
        out_dims_1, acts_1 = au.transpose([(ed * 2, relu), (md, relu), (md, None)])
        self.DL_gen_1 = MyDenses(in_dim=ed, out_dims=out_dims_1, scale=sc, activations=acts_1)
        self.DL_gen_2 = MyDenses(in_dim=ed, out_dims=out_dims_1, scale=sc, activations=acts_1)
        self.DL_gen_3 = MyDenses(in_dim=ed, out_dims=out_dims_1, scale=sc, activations=acts_1)
        
        out_dims_2, acts_2 = au.transpose([(md * 2, relu), (md, None)])
        self.DL_dis_1 = MyDenses(in_dim=md, out_dims=out_dims_2, scale=sc, activations=acts_2)
        self.DL_dis_2 = MyDenses(in_dim=md, out_dims=out_dims_2, scale=sc, activations=acts_2)
        self.DL_dis_3 = MyDenses(in_dim=md, out_dims=out_dims_2, scale=sc, activations=acts_2)
        self.D_dis = [self.DL_dis_1, self.DL_dis_2, self.DL_dis_3]
    
    def forward(self):
        l1, l2, _, l4 = self.l_reg
        mgn, nis = self.margin, self.noise
        c, p, n = self.get_original_cpn()
        with tf.name_scope('gen_deep'):
            c = self.DL_gen_1(c)
            p = self.DL_gen_2(p)
            n = self.DL_gen_3(n)
        
        with tf.name_scope('get_c_probs'):
            pc_score = tensor_dot(p, tf.transpose(c))
            pc_probs = tf.nn.softmax(pc_score, axis=1)
        with tf.name_scope('get_p_res'):
            r = tensor_dot(pc_probs, c)
            r = self.D_r(r, name='get_r')
            # r = tf.nn.relu(r)
        with tf.name_scope('normalize_prn'):
            p_norm, r_norm, n_norm = l2_norm_tensors(p, r, n)
        # with tf.name_scope('add_noise'):
        #     if nis > 1e-6:
        #         p_norm += tf.random_normal(tf.shape(p_norm), mean=0., stddev=nis)
        #         r_norm += tf.random_normal(tf.shape(r_norm), mean=0., stddev=nis)
        
        with tf.name_scope('discriminate'):
            p_D, r_D, n_D = self.DL_dis_1(p), self.DL_dis_2(r), self.DL_dis_3(n)
            p_Dn, r_Dn, n_Dn = l2_norm_tensors(p_D, r_D, n_D)
            # batch_size * 1
            pr_sim_D = inner_dot(p_Dn, r_Dn, keepdims=True)
            pn_sim_D = tensor_dot(p_Dn, tf.transpose(n_Dn))
            pn_sim_D_v = tf.reduce_mean(pn_sim_D, axis=1, keepdims=True)
            # rn_sim_D = tensor_dot(r_Dn, tf.transpose(n_Dn))
            # rn_sim_D_v = tf.reduce_mean(rn_sim_D, axis=1, keepdims=True)
            with tf.name_scope('loss'):
                loss_D = tf.reduce_mean(pr_sim_D - pn_sim_D_v)
        with tf.name_scope('generate'):
            # batch_size * 1
            pr_sim_G = inner_dot(p_norm, r_norm, keepdims=True)
            # batch_size * (neg sample size)
            pn_sim_G = tensor_dot(p_norm, tf.transpose(n_norm))
            pn_sim_G_v = tf.reduce_mean(pn_sim_G, axis=1, keepdims=True)
            # rn_sim_G = tensor_dot(r_norm, tf.transpose(n_norm))
            # rn_sim_G_v = tf.reduce_mean(rn_sim_G, axis=1, keepdims=True)
            # clu_num * embed_dim
            c_embed = l2_norm_tensors(c)
            c_sim = tf.matmul(c_embed, tf.transpose(c_embed))
            c_sim_loss = tf.nn.l2_loss(c_sim - tf.diag(tf.diag_part(c_sim)))
            with tf.name_scope('loss'):
                c_sim_loss = c_sim_loss * l2
                reg_G = sum([d.get_norm(order=2) for d in self.W_doc]) * l4
                loss_G_pre = reg_G + c_sim_loss + tf.reduce_mean(
                    tf.maximum(0., mgn - pr_sim_G + pn_sim_G_v * l1))
                loss_G = reg_G + c_sim_loss + tf.reduce_mean(
                    tf.maximum(0., mgn - pr_sim_D + pn_sim_D_v * l1))
        
        self.pc_probs = pc_probs
        self.loss_D = loss_D
        self.loss_G_pre = loss_G_pre
        self.loss = self.loss_G = loss_G
        with tf.name_scope('register'):
            tf.summary.scalar(name='loss_D', tensor=loss_D)
            tf.summary.scalar(name='loss_G_pre', tensor=loss_G_pre)
            tf.summary.scalar(name='loss_G', tensor=loss_G)
            
            tf.summary.histogram(name='pr_sim_G', values=pr_sim_G, family='G_sim')
            tf.summary.histogram(name='pn_sim_G', values=pn_sim_G, family='G_sim')
            # tf.summary.histogram(name='rn_sim_G', values=rn_sim_G, family='G_sim')
            tf.summary.histogram(name='pr_sim_D', values=pr_sim_D, family='D_sim')
            tf.summary.histogram(name='pn_sim_D', values=pn_sim_D, family='D_sim')
            # tf.summary.histogram(name='rn_sim_D', values=rn_sim_D, family='D_sim')
            
            tf.summary.histogram(name='p', values=p, family='doc')
            tf.summary.histogram(name='r', values=r, family='doc')
            tf.summary.histogram(name='n', values=n, family='doc')
            tf.summary.histogram(name='p_D', values=p_D, family='D_out')
            tf.summary.histogram(name='r_D', values=r_D, family='D_out')
            tf.summary.histogram(name='r_D', values=r_D, family='D_out')
