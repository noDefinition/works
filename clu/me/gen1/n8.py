from .n6 import *


# noinspection PyAttributeOutsideInit,PyPep8Naming
class N8(N6):
    """ 对embedding进行粗暴的正则化 """
    file = __file__

    def forward(self):
        l1, l2, l3, l4 = self.l_reg
        c, Pd, Nd = self.get_cpn()

        with tf.name_scope('similarity'):
            pc_probs, Pr = self.get_c_probs_r(Pd, c, 'reconstruct_p')
            _, Nr = self.get_c_probs_r(Nd, c, 'reconstruct_n')
            with tf.name_scope('loss'):
                Pd_l2, Pr_l2, Nd_l2, Nr_l2 = l2_norm_tensors(Pd, Pr, Nd, Nr)
                PdPr_sim = inner_dot(Pd_l2, Pr_l2, keepdims=True)
                PdNd_sim = tf.matmul(Pd_l2, tf.transpose(Nd_l2))
                pairwise_margin = tf.maximum(0., self.margin - PdPr_sim + PdNd_sim)
                """merge loss"""
                pairwise = tf.reduce_mean(pairwise_margin)
                pointwise = tf.reduce_mean(PdPr_sim)
                reg_loss = sum(w.get_norm(order=2) for w in self.W_doc)
                sim_loss = l1 * pairwise - l3 * pointwise + l4 * reg_loss
                if l2 > 1e-6:
                    reg_embed = [[self.w_embed], [self.c_embed],
                                 [self.w_embed, self.c_embed]][self.worc]
                    sim_loss += l2 * sum(tf.norm(t, ord=2) for t in reg_embed)

        self.pc_probs = pc_probs
        self.sim_loss = sim_loss
        self.pairwise = pairwise
        self.pointwise = pointwise

    def define_optimizer(self):
        self.optimizer = tf.train.AdamOptimizer(tf.train.exponential_decay(
            learning_rate=5e-3, global_step=0, decay_steps=50, decay_rate=0.99))
        self.sim_opt = self.optimizer.minimize(self.sim_loss)

    """ runtime below """

    def train_step(self, fd, epoch, max_epoch, batch_id, max_batch_id):
        self.sess.run(self.sim_opt, feed_dict=fd)
