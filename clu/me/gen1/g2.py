from me.gen1.g1 import *


# noinspection PyAttributeOutsideInit,PyMethodMayBeStatic,PyPep8Naming
class G2(G1):
    """ 判别器分别输入正例与重构（要求对二者使用同样的结构），输出概率值并比较 """

    def define_denses(self):
        super(G2, self).define_denses()
        sc, ed, md = self.scale, self.e_dim, self.m_dim
        self.DL_dis = MyDenses(ed, [md, int(md / 2), 1], sc, (relu, relu, sigmoid))
        self.D_dis = [self.DL_dis]
    
    def forward(self):
        l1, _, _, l4 = self.l_reg
        mgn, nis = self.margin, self.noise
        c, p, n = self.get_original_cpn()
        
        with tf.name_scope('get_c_probs'):
            pc_score = tensor_dot(p, tf.transpose(c))
            pc_probs = tf.nn.softmax(pc_score, axis=1)
        with tf.name_scope('get_p_res'):
            r = tensor_dot(pc_probs, c)
            r = self.D_r(r, name='get_r')
            # r = tf.nn.relu(r)
        
        with tf.name_scope('l2norm_prn'):
            p_norm, r_norm, n_norm = l2_norm_tensors(p, r, n)
        
        with tf.name_scope('discriminate'):
            p_score_D = self.DL_dis(p)
            r_score_D = self.DL_dis(r)
            with tf.name_scope('loss'):
                p_score_D_v = tf.reduce_mean(p_score_D)
                r_score_D_v = tf.reduce_mean(1. - r_score_D)
                loss_D = -(p_score_D_v + r_score_D_v)
        
        with tf.name_scope('generate'):
            # batch_size * 1
            pr_sim_G = inner_dot(p_norm, r_norm, keepdims=True)
            # batch_size * (neg sample size)
            pn_sim_G = tensor_dot(p_norm, tf.transpose(n_norm))
            pn_sim_G_v = tf.reduce_mean(pn_sim_G, axis=1, keepdims=True)
            # rn_sim_G = tensor_dot(r_norm, tf.transpose(n_norm))
            # rn_sim_G_v = tf.reduce_mean(rn_sim_G, axis=1, keepdims=True)
            with tf.name_scope('loss'):
                reg_G = sum([d.get_norm(order=2) for d in self.W_doc]) * l4
                loss_G_pre = tf.reduce_mean(
                    tf.maximum(0., mgn - pr_sim_G + pn_sim_G_v * l1)) + reg_G
                loss_G = loss_G_pre - loss_D
        
        self.pc_probs = pc_probs
        self.loss_D = loss_D
        self.loss_G_pre = loss_G_pre
        self.loss = self.loss_G = loss_G
        
        with tf.name_scope('register'):
            tf.summary.scalar(name='loss_D', tensor=loss_D)
            tf.summary.scalar(name='loss_G_pre', tensor=loss_G_pre)
            tf.summary.scalar(name='loss_G', tensor=loss_G)
            tf.summary.scalar(name='p_score_D_v', tensor=p_score_D_v)
            tf.summary.scalar(name='r_score_D_v', tensor=r_score_D_v)
            
            tf.summary.histogram(name='pr_sim_G', values=pr_sim_G, family='G_sim')
            tf.summary.histogram(name='pn_sim_G', values=pn_sim_G, family='G_sim')
            
            tf.summary.histogram(name='p_score_D', values=p_score_D, family='D_score')
            tf.summary.histogram(name='r_score_D', values=r_score_D, family='D_score')
            
            tf.summary.histogram(name='p', values=p, family='doc')
            tf.summary.histogram(name='r', values=r, family='doc')
            tf.summary.histogram(name='n', values=n, family='doc')
            # tf.summary.histogram(name='p_norm', values=p_norm, family='doc')
            # tf.summary.histogram(name='r_norm', values=r_norm, family='doc')
            # tf.summary.histogram(name='n_norm', values=n_norm, family='doc')
