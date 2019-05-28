from .n6 import *


# noinspection PyAttributeOutsideInit,PyPep8Naming
class N7(N6):
    file = __file__

    def define_cluster_embed(self, clu_embed):
        super(N7, self).define_cluster_embed(clu_embed)
        self.c_noise = tf.nn.l2_normalize(tf.random_uniform(clu_embed.shape, minval=-1., maxval=1.))

    def define_word_embed(self, word_embed):
        super(N7, self).define_word_embed(word_embed)
        self.w_noise = tf.nn.l2_normalize(tf.random_uniform(word_embed.shape, minval=-1., maxval=1.))

    def define_optimizer(self):
        self.optimizer = tf.train.AdamOptimizer(tf.train.exponential_decay(
            learning_rate=5e-3, global_step=0, decay_steps=50, decay_rate=0.99))
        self.sim_opt = self.optimizer.minimize(self.sim_loss)
        if self.use_adv_nis:
            self.adv_opt = self.optimizer.minimize(self.adv_loss)

    """ runtime below """

    def train_step(self, fd, epoch, max_epoch, batch_id, max_batch_id):
        self.sess.run(self.adv_opt if self.use_adv_nis else self.sim_opt, feed_dict=fd)

