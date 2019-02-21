from .n1 import *


# noinspection PyAttributeOutsideInit,PyPep8Naming
class N2(N1):
    """ 仅在pairwise内进行对抗，用聚类embedding作为生成器 """
    file = __file__

    def define_denses(self):
        ed, md = self.w_dim, self.m_dim
        initer = self.n_init

        self.W_1 = Dense(ed, ed, kernel=initer, name='W_1')
        self.W_2 = Dense(ed, ed, kernel=initer, name='W_2')
        self.W_3 = Dense(ed, ed, kernel=initer, name='W_3')
        self.Q = tf.get_variable(shape=(1, 1, ed), initializer=initer, name='Q')
        self.W_doc = [self.W_1, self.W_2, self.W_3]

        # self.D_c = Dense(ed, ed, activation=relu, kernel_initializer=initer, name='D_c')
        # self.D_r = Dense(ed, ed, activation=relu, kernel_initializer=initer, name='D_r')
        # self.D_r = Denses(ed, [(ed, relu), (ed, None)], initer, name='D_r')

    def get_c_probs_r(self, e, c, name):
        with tf.name_scope(name):
            # batch_size * clu_num
            c_score = tf.matmul(e, tf.transpose(c))
            c_probs = tf.nn.softmax(c_score, axis=1)
            # batch_size * embed_dim
            r = tf.matmul(c_probs, c)
            # r = self.D_r(r, name='proj_r')
            # r = relu(r, name='relu_r')
        return c_probs, r

    def forward(self):
        l1, l2, l3, l4 = self.l_reg
        c, Pd, Nd = self.get_cpn()

        with tf.name_scope('similarity'):
            pc_probs, Pr = self.get_c_probs_r(Pd, c, 'reconstruct_p')
            nc_probs, Nr = self.get_c_probs_r(Nd, c, 'reconstruct_n')
            with tf.name_scope('loss'):
                Pd_l2, Pr_l2, Nd_l2, Nr_l2 = l2_norm_tensors(Pd, Pr, Nd, Nr)
                PdPr_sim = inner_dot(Pd_l2, Pr_l2, keepdims=True)
                PdNd_sim = tf.matmul(Pd_l2, tf.transpose(Nd_l2))
                PdNd_sim_v = tf.reduce_mean(PdNd_sim, axis=1, keepdims=True)
                # PdPr_PdNd_mgn = tf.maximum(0.0, 1.0 - PdPr_sim + PdNd_sim_v * l1)
                PdPr_PdNd_mgn = - PdPr_sim + PdNd_sim_v * l1
                """merge loss"""
                mgn_loss = tf.reduce_mean(PdPr_PdNd_mgn)
                sim_reward = tf.reduce_mean(PdPr_sim)
                reg_loss = sum(w.get_norm(order=2) for w in self.W_doc)
                sim_loss = mgn_loss - sim_reward * l3
                dis_loss = sim_loss + reg_loss * l4
                gen_loss = - sim_loss

        self.pc_probs = pc_probs
        self.dis_loss = dis_loss
        self.gen_loss = - gen_loss

        histogram(name='Pd', values=Pd, family='origin')
        histogram(name='Nd', values=Nd, family='origin')
        histogram(name='Pd_l2', values=Pd_l2, family='origin')
        histogram(name='Nd_l2', values=Nd_l2, family='origin')

        histogram(name='Pr', values=Pr, family='recons')
        histogram(name='Nr', values=Nr, family='recons')
        histogram(name='Pr_l2', values=Pr_l2, family='recons')
        histogram(name='Nr_l2', values=Nr_l2, family='recons')

        histogram(name='PdNd_sim', values=PdNd_sim, family='sim')
        histogram(name='PdPr_sim', values=PdPr_sim, family='sim')

        scalar(name='mgn_loss', tensor=mgn_loss, family='sim')
        scalar(name='sim_reward', tensor=sim_reward, family='sim')
        scalar(name='sim_loss', tensor=dis_loss, family='sim')

    def define_optimizer(self):
        def record_grad_vars(prompt, grad_vars):
            print(prompt, '- var num:{}'.format(len(grad_vars)))
            for g, v in grad_vars:
                v_name = v.name.replace(':', '_')
                print(' ' * 4, v_name)
                histogram(name=v_name, values=g, family=prompt)

        lr = 1e-3
        vars_d = get_trainable(self.c_embed)
        vars_g = sort_variables(subtract_from(tf.trainable_variables(), vars_d))
        self.dis_op, grad_vars_d = get_train_op(
            lr, self.dis_loss, vars_d, name='dis_op', grads=True)
        self.gen_op, grad_vars_g = get_train_op(
            lr, self.gen_loss, vars_g, name='gen_op', grads=True)
        record_grad_vars('grads_discriminator', grad_vars_d)
        record_grad_vars('grads_generator', grad_vars_g)

    """ runtime below """

    def get_loss(self, sess, fd):
        sl = sess.run([self.dis_loss], feed_dict=fd)
        return {'dis_loss': sl}

    def train_step(self, sess, fd, epoch, max_epoch, batch_id, max_batch_id):
        def train():
            for _ in range(self.dis_step_num):
                sess.run(self.dis_op, feed_dict=fd)
            for _ in range(self.gen_step_num):
                sess.run(self.gen_op, feed_dict=fd)

        control_dependencies(self.use_bn, train)
