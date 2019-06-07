import numpy as np
from clu.me import C
from utils.deep.funcs import *
from tensorflow.keras.layers import Masking, CuDNNLSTM
from typing import List


# noinspection PyAttributeOutsideInit
class VAE1:
    def __init__(self, args: dict):
        self.lr: float = args[C.lr]
        self.dim_h: int = args[C.hd]
        self.smooth: float = args.get(C.smt, 1)
        self.coeff_kl: float = args.get(C.ckl, 1)
        # self.dim_m: int = args[C.md]
        # self.dropout: float = args[C.drp]
        # assert 0 <= self.dropout <= 1
        assert 0 <= self.smooth <= 1
        assert 0 <= self.coeff_kl <= 1

        self.x_init = tf.contrib.layers.xavier_initializer()
        self.global_step = tf.Variable(0, name='global_step')
        self.global_step_inc = tf.assign(self.global_step, tf.add(self.global_step, 1))
        self.sess: tf.Session = None

    def define_cluster_embed(self, clu_init):
        self.num_c, self.dim_c = clu_init.shape
        init = tf.constant_initializer(clu_init)
        self.c_embed = tf.get_variable('c_embed', clu_init.shape, initializer=init)  # (cn, dw)
        init = tf.zeros_initializer()
        self.c_logvar = tf.get_variable('c_logvar', clu_init.shape, initializer=init)  # (cn, dw)

    def define_word_embed(self, word_init):
        # trainable = [False, True][self.w_train]
        self.num_w, self.dim_w = word_init.shape
        self.num_w += 1
        init = tf.constant_initializer(word_init)
        emb = tf.get_variable('w_emb', word_init.shape, initializer=init)
        pad = tf.zeros([1, word_init.shape[1]], name='w_pad', dtype=f32)
        self.w_embed = tf.concat([pad, emb], axis=0, name='w_embed')

    def define_inputs(self):
        lk = tf.nn.embedding_lookup
        self.ph_if_sample = tf.placeholder(tf.bool, (), name='if_sample')
        self.ph_is_train = tf.placeholder(tf.bool, (), name='is_train')
        # self.ph_drop_keep = tf.placeholder_with_default(1., (), name='dropout_keep')
        self.p_wids = tf.placeholder(i32, (None, None), name='p_seq')  # (bs, tn)
        self.p_mask = self.get_mask(self.p_wids, False, name='p_mask')  # (bs, tn)
        self.p_lkup = lk(self.w_embed, self.p_wids, name='p_lkup')  # (bs, tn, dw)

        p_truncate = self.p_wids[:, 1:]  # (bs, tn-1), drop the first word
        p_padding = tf.zeros([tf.shape(self.p_wids)[0], 1], name='p_padding', dtype=i32)  # (bs, 1)
        self.p_shift = tf.concat([p_truncate, p_padding], axis=1, name='p_shift')  # (bs, tn)
        self.p_shift_mask = self.get_mask(self.p_shift, False, name='p_sh_mk')  # (bs, tn)
        # self.p_shift_lkup = lk(self.w_embed, self.p_shift, name='p_sh_lk')  # (bs,tn,1)

    @staticmethod
    def get_mask(wids, expand_last: bool, name: str):
        with tf.name_scope(name):
            mask = tf.cast(tf.not_equal(wids, 0), dtype=f32)  # (bs, tn)
            if expand_last:
                mask = tf.expand_dims(mask, axis=-1)  # (bs, tn, 1)
        return mask

    def define_weight_params(self):
        self.decode_lstm = CuDNNLSTM(self.dim_h, return_state=True, return_sequences=True)

    def sample_z(self, mu, var, name: str):
        with tf.name_scope(name):
            gaussian = tf.random_normal(shape=tf.shape(mu), name='gaussian')
            # z = mu + tf.exp(var * 0.5) * gaussian
            z = mu + var * gaussian  # (bs, dw)
            sample = tf.cond(self.ph_is_train, true_fn=lambda: z, false_fn=lambda: mu)
        return sample

    @staticmethod
    def get_normal_kl(mu, var, name: str):
        with tf.name_scope(name):
            # log_var = log_var * 0.5
            # neg_log_var = - tf.reduce_mean(var)
            # var_square = tf.reduce_mean(tf.exp(var))
            # mu_square = tf.reduce_mean(tf.square(mu))
            mu_square = tf.square(mu)  # (bs, dw)
            neg_log_var = - tf.log(var)  # (bs, dw)
            elements = tf.add_n([mu_square, neg_log_var, var])  # (bs, dw)
            kld = tf.reduce_mean(tf.reduce_sum(elements, axis=-1))  # * 0.5
        return kld

    # @staticmethod
    # def top_k(inputs, k: int, name='top_k'):
    #     with tf.variable_scope(name):
    #         inputs = tf.transpose(inputs, [0, 2, 1])
    #         inputs = tf.nn.top_k(inputs, k).values
    #         inputs = tf.transpose(inputs, [0, 2, 1])
    #     return inputs
    #
    # def top_k_mean(self, inputs, k: int, name='top_k_mean'):
    #     with tf.variable_scope(name):
    #         top_k_values = self.top_k(inputs, k)
    #         o = tf.reduce_mean(top_k_values, 1)
    #     return o

    def get_doc_embed(self, inputs, name):
        with tf.name_scope(name):
            doc_embed = tf.reduce_mean(inputs, axis=1)  # (bs, tn, dw)
        return doc_embed  # (bs, dw)

    def forward(self):
        # p_rep = self.top_k_mean(self.p_lkup, k=10, name='p_rep')  # (bs, dw)
        p_rep = self.get_doc_embed(self.p_lkup, name='p_rep')  # (bs, dw)
        pc_score = tf.matmul(p_rep, self.c_embed, transpose_b=True, name='pc_score')  # (bs, cn)
        pc_probs = tf.nn.softmax(pc_score, axis=1, name='pc_probs')  # (bs, cn)
        pc_recon = tf.matmul(pc_probs, self.c_embed, name='pc_recon')  # (bs, dw)
        back_stdvar = tf.exp(self.c_logvar * 0.5, name='back_stdvar')  # (bs, dw)
        pc_stdvar = tf.matmul(pc_probs, back_stdvar, name='pc_stdvar')  # (bs, dw)
        z_p = self.sample_z(pc_recon, pc_stdvar, name='z_p')  # (bs, ed)
        # z_p_d = instant_denses(z_p, [(self.dim_h, None)], name='z_p_dense')  # (bs, hd)
        z_p_d = z_p

        decoder_output, _, _ = self.decode_lstm(
            self.p_lkup, initial_state=[z_p_d, z_p_d], training=self.ph_is_train)
        uas = [(self.dim_m, relu), (self.num_w, None)]
        decode_preds = instant_denses(decoder_output, uas, 'decode_preds')  # (bs, tn, nw)

        p_shift_hot = tf.one_hot(
            indices=self.p_shift, depth=self.num_w, name='p_shift_hot',
            on_value=self.smooth, off_value=(1 - self.smooth) / (self.num_w - 1))  # (bs, tn, nw)
        right_ce = tf.nn.softmax_cross_entropy_with_logits_v2(
            labels=p_shift_hot, logits=decode_preds, name='right_ce')  # (bs, tn)
        right_ce_mask = tf.multiply(right_ce, self.p_shift_mask, name='right_ce_mask')
        right_ce_loss = tf.reduce_mean(right_ce_mask, name='right_ce_loss')

        decode_1, decode2 = decode_preds[1:, :, :], decode_preds[:1, :, :]
        wrong_decode_preds = tf.concat([decode_1, decode2], axis=0, name='wrong_decode_preds')
        wrong_ce = tf.nn.softmax_cross_entropy_with_logits_v2(
            labels=p_shift_hot, logits=wrong_decode_preds, name='wrong_ce')  # (bs, tn)
        wrong_ce_mask = tf.multiply(wrong_ce, self.p_shift_mask, name='wrong_ce_mask')
        wrong_ce_loss = - tf.reduce_mean(wrong_ce_mask, name='wrong_ce_loss')

        z_prior_kl = self.get_normal_kl(pc_recon, pc_stdvar, name='z_prior_kl')
        self.total_loss = tf.add_n([
            right_ce_loss,
            wrong_ce_loss,
            z_prior_kl * self.coeff_kl,
        ], name='total_loss')

        self.z_p = z_p
        self.z_p_d = z_p_d
        self.pc_probs = pc_probs
        self.decode_preds = decode_preds

        self.merge_mid = su.merge([
            histogram(name='z_p', values=z_p, family='z'),
            histogram(name='z_p_d', values=z_p_d, family='z'),
            histogram(name='pc_probs', values=pc_probs, family='clu'),
            histogram(name='pc_recon', values=pc_recon, family='clu'),
            histogram(name='pc_stdvar', values=pc_stdvar, family='clu'),
            histogram(name='decode_preds', values=decode_preds, family='decode'),
        ])
        self.merge_loss = su.merge([
            scalar(name='right_ce_loss', tensor=right_ce_loss, family='loss'),
            scalar(name='wrong_ce_loss', tensor=wrong_ce_loss, family='loss'),
            scalar(name='z_prior_kl', tensor=z_prior_kl, family='loss'),
            scalar(name='total_loss', tensor=self.total_loss, family='loss'),
        ])

    def define_optimizer(self):
        lr = tf.train.exponential_decay(
            learning_rate=self.lr, global_step=self.global_step, decay_steps=200, decay_rate=0.99)
        optimizer = tf.train.AdamOptimizer(learning_rate=lr, name='AAAAdam')
        self.cross_op = optimizer.minimize(self.total_loss)

    def build(self, word_init, clu_init):
        self.define_word_embed(word_init)
        self.define_cluster_embed(clu_init)
        self.define_weight_params()
        self.define_inputs()
        self.forward()
        self.define_optimizer()

    """ runtime below """

    def get_fd(self, p_seq: List[List[int]], is_train: bool) -> dict:
        return {self.p_wids: p_seq, self.ph_is_train: is_train}

    def train_step(self, p_seq, epoch_i, batch_i, *args, **kwargs):
        fd = self.get_fd(p_seq, is_train=True)
        self.sess.run(self.cross_op, feed_dict=fd)
        self.sess.run(self.global_step_inc)

    def predict(self, p_seq) -> List[int]:
        pc_probs = self.run_parts(p_seq, self.pc_probs)
        return np.argmax(pc_probs, axis=1).reshape(-1)

    def run_parts(self, p_seq, *parts):
        assert len(parts) > 0
        fd = self.get_fd(p_seq, is_train=False)
        parts_res = self.sess.run(parts, feed_dict=fd)
        return parts_res if len(parts_res) > 1 else parts_res[0]

    def run_merge(self, p_seq, merge):
        fd = self.get_fd(p_seq, is_train=False)
        return self.sess.run(merge, feed_dict=fd)

    def set_session(self, sess):
        self.sess = sess

    def save(self, file: str):
        tf.train.Saver().save(self.sess, file)

    def load(self, file: str):
        tf.train.Saver().restore(self.sess, file)
