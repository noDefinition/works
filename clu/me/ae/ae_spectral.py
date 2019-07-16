from tensorflow.python.keras.losses import KLD
from utils.logger_utils import print_on_first

from clu.me.vae.vae2 import *


# noinspection PyAttributeOutsideInit
class AeSpectral(VAE2):
    # reconstruct tf-idf, spectral loss
    def get_hyper_params(self, args):
        self.lr: float = args[C.lr]
        self.dim_h: int = args[C.hd]
        self.c_init: int = args[C.cini]
        self.w_init: int = args[C.wini]
        self.sigma: float = args[C.sigma]
        self.c_spectral: float = args[C.cspec]
        self.c_decode: float = args[C.cdecd]
        self.w_train: bool = bool(args[C.wtrn])
        self.pre_train_step: int = args[C.ptn]
        self.pre_train_comb: int = args[C.ptncmb]

    # def define_word_embed(self, word_init):
    #     self.num_w, self.dim_w = word_init.shape
    #     self.num_w += 1
    #     init = [self.x_init, tf.constant_initializer(word_init)][self.w_init]
    #     emb = tf.get_variable('w_emb', word_init.shape, initializer=init, trainable=self.w_train)
    #     pad = tf.zeros((1, word_init.shape[1]), name='w_pad', dtype=f32)  # (1, dw)
    #     self.w_embed = tf.concat([pad, emb], axis=0, name='w_embed')  # (wn+1, dw)

    def define_inputs(self):
        super(AeSpectral, self).define_inputs()
        self.ph_is_pretrain = tf.placeholder_with_default(False, (), name='ph_is_pre_train')
        self.ph_tfidf = tf.placeholder(f32, (None, None), name='p_tfidf')  # (bs, nw)

    def mutual_norm(self, a, b, name):
        assert len(a.shape) == len(b.shape) == 2
        with tf.name_scope(name):
            a = tf.expand_dims(a, axis=1)  # (bs, 1, dw)
            b = tf.expand_dims(b, axis=0)  # (1, bs, dw)
            ab_sq = tf.square(a - b)  # (bs, bs, dw)
            ab_sum = tf.reduce_sum(ab_sq, axis=-1)  # (bs, bs)
            ab_norm = tf.sqrt(ab_sum + 1e-12)  # (bs, bs)
            # ab_norm = ab_sum
        return ab_norm

    def forward(self):
        p_stop = tf.stop_gradient(self.p_lkup, name='p_stop')  # (bs, dw)
        p_lkup = tf.cond(self.ph_is_pretrain, true_fn=lambda: p_stop, false_fn=lambda: self.p_lkup)
        p_mean = self.get_mean_pooling(p_lkup, self.p_mask, name='p_mean')  # (bs, dw)
        uas = [(self.dim_w * 2, relu), (self.dim_w * 2, relu), (self.num_c, None)]
        pd_score = instant_denses(p_mean, uas, name='pd_score')  # (bs, cn)
        pd_probs = softmax(pd_score, axis=-1, name='pd_probs')  # (bs, cn)

        with tf.name_scope('pretrain_loss'):
            pc_score_pre = tf.matmul(p_mean, self.c_embed, transpose_b=True, name='pc_score_pre')
            pc_probs_pre = softmax(pc_score_pre, axis=-1, name='pc_probs_pre')
            pc_probs_pre = tf.stop_gradient(pc_probs_pre)
            dc_kld = KLD(y_true=pc_probs_pre, y_pred=tf.maximum(pd_probs, 1e-12))
            dc_kld_loss = tf.reduce_mean(dc_kld)

        with tf.name_scope('spectral_loss'):
            pmut_norm = self.mutual_norm(p_mean, p_mean, name='pmut_norm')  # (bs, bs)
            radius = - 2 * (self.sigma ** 2)
            pmut_kernel = tf.exp(pmut_norm / radius)
            pred_norm = self.mutual_norm(pd_probs, pd_probs, name='pred_norm')  # (bs, bs)
            spectral_loss = tf.reduce_mean(pmut_kernel * pred_norm, name='spectral_loss')

        with tf.name_scope('decode_loss'):
            p_recon = tf.matmul(pd_probs, self.c_embed)  # (bs, dw)
            p_decode = tf.concat([pd_probs, p_recon], axis=1)
            uas = [(self.dim_w * 2, relu), (self.dim_w * 2, relu), (self.num_w, None)]
            decode_tfidf = instant_denses(p_decode, uas, name='decode_tfidf')  # (bs, nw)
            decode_cos_sim = self.cos_sim(self.ph_tfidf, decode_tfidf, name='decode_cos_sim')
            decode_cos_loss = - tf.reduce_mean(decode_cos_sim, name='decode_cos_loss')

        self.pre_train_loss = tf.add_n([
                                           [dc_kld_loss],
                                           [decode_cos_loss],
                                           [dc_kld_loss, decode_cos_loss],
                                       ][self.pre_train_comb], name='pre_train_loss')
        self.train_loss = tf.add_n([
            spectral_loss * self.c_spectral if self.c_spectral > 1e-6 else 0.,
            decode_cos_loss * self.c_decode if self.c_decode > 1e-6 else 0.,
        ], name='total_loss')
        self.pc_probs = pd_probs

        self.merge_mid = su.merge([
            histogram(name='pmut_norm', values=pmut_norm, family='kernel'),
            histogram(name='pmut_kernel', values=pmut_kernel, family='kernel'),
            histogram(name='pred_norm', values=pred_norm, family='kernel'),
            histogram(name='decode_tfidf', values=decode_tfidf, family='decode'),
            histogram(name='decode_cos_sim', values=decode_cos_sim, family='decode'),
            histogram(name='pd_probs', values=self.pc_probs, family='clu'),
        ])
        self.merge_loss = su.merge([
            scalar(name='spectral_loss', tensor=spectral_loss, family='loss'),
            scalar(name='decode_cos_loss', tensor=decode_cos_loss, family='loss'),
            scalar(name='pre_train_loss', tensor=self.pre_train_loss, family='loss'),
            scalar(name='total_loss', tensor=self.train_loss, family='loss'),
        ])

    def define_optimizer(self):
        lr = tf.train.exponential_decay(
            learning_rate=self.lr, global_step=self.global_step, decay_steps=200, decay_rate=0.99)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=lr, name='AAAAdam_cross')
        self.train_op = self.optimizer.minimize(self.train_loss, name='train_op')
        self.pre_optimizer = tf.train.AdamOptimizer(learning_rate=lr, name='Adam_pre')
        self.pre_train_op = self.pre_optimizer.minimize(self.pre_train_loss, name='pre_train_op')
        # for g, v in self.pre_optimizer.compute_gradients(self.pre_train_loss):
        #     if g is not None:
        #         print(g, v)

    """ runtime """

    def train_step(self, docarr: List[Document], epoch_i: int, batch_i: int, **kwargs):
        fd = self.get_fd(docarr, is_train=True)
        if epoch_i < self.pre_train_step:
            print_on_first(epoch_i, f'  --phase 1--  pre-training on epoch {epoch_i}')
            fd[self.ph_is_pretrain] = True
            self.sess.run(self.pre_train_op, feed_dict=fd)
        else:
            print_on_first(epoch_i, f'  --phase 2--  formal training on epoch {epoch_i}')
            self.sess.run(self.train_op, feed_dict=fd)
        self.global_step += 1
