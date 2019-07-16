import numpy as np
from typing import List

from clu.me import C
from utils.deep.funcs import *
from utils.doc_utils import Document


# noinspection PyAttributeOutsideInit,PyPep8Naming
class SBX(object):
    def __init__(self, args: dict):
        self.lr: float = args[C.lr]
        self.dim_h: int = args[C.hd]
        self.c_init: int = args[C.cini]
        self.w_init: int = args[C.wini]
        self.use_bias: bool = bool(args[C.useb])
        self.c_reg: float = args[C.creg]
        self.span: float = args[C.span]

        self.x_init = tf.contrib.layers.xavier_initializer()
        self.global_step: int = 0
        self.alpha: float = 0
        self.sess: tf.Session = None

    def define_cluster_embed(self, clu_init):
        self.num_c, self.dim_c = clu_init.shape
        init = [self.x_init, tf.constant_initializer(clu_init)][self.c_init]
        self.c_embed = tf.get_variable('c_embed', clu_init.shape, initializer=init)  # (cn, dw)

    def define_word_embed(self, word_init):
        self.num_w, self.dim_w = word_init.shape
        self.num_w += 1
        init = [self.x_init, tf.constant_initializer(word_init)][self.w_init]
        emb = tf.get_variable('w_emb_core', word_init.shape, initializer=init)  # (wn, dw)
        pad = tf.zeros([1, word_init.shape[1]], name='w_pad', dtype=f32)  # (1, dw)
        self.w_embed = tf.concat([pad, emb], axis=0, name='w_embed')  # (wn+1, dw)

    def define_inputs(self):
        self.ph_is_train = tf.placeholder(tf.bool, (), name='is_train')
        self.ph_drop_keep = tf.placeholder_with_default(1., (), name='dropout_keep')
        self.ph_global_step = tf.placeholder(i32, (), name='global_step')
        self.ph_alpha = tf.placeholder(f32, (), name='alpha')

        self.p_wids = tf.placeholder(i32, (None, None), name='p_seq')  # (bs, tn)
        self.p_mask = self.get_mask(self.p_wids, True, name='p_mask')  # (bs, tn)
        lookup = tf.nn.embedding_lookup
        self.p_lkup = lookup(self.w_embed, self.p_wids, name='p_lkup')  # (bs, tn, dw)

    @staticmethod
    def get_mask(wids, expand_last: bool, name: str):
        with tf.name_scope(name):
            mask = tf.cast(tf.not_equal(wids, 0), dtype=f32)  # (bs, tn)
            if expand_last:
                mask = tf.expand_dims(mask, axis=-1)  # (bs, tn, 1)
        return mask

    def define_weight_params(self):
        pass

    def get_doc_embed(self, inputs, mask, name: str):
        assert len(inputs.shape) == 3
        with tf.name_scope(name):
            doc_len = tf.reduce_sum(mask, axis=[1, 2])  # (bs, 1)
            doc_emb = tf.reduce_sum(inputs, axis=1) / doc_len  # (bs, dw)
        return doc_emb  # (bs, dw)

    def reg_diag(self, inputs, name: str):
        # X may be no dim, reg the last dim
        with tf.name_scope(name):
            aTa = tf.matmul(inputs, inputs, transpose_a=True)  # (X, cn, cn)
            eye = tf.expand_dims(tf.eye(self.num_c), 0)  # (1, cn, cn)
            # aTa_eye_norm = tf.norm(aTa - eye, ord=2, axis=[-1, -2], keepdims=False)  # (X, )
            aTa_eye_sq_sum = tf.reduce_sum(tf.square(aTa - eye), axis=[1, 2])
            aTa_eye_norm = tf.sqrt(aTa_eye_sq_sum)
            reg = tf.reduce_mean(aTa_eye_norm)
            # print('reg diag', name, inputs.shape, aTa.shape, eye.shape, aTa_eye_norm.shape)
        return reg

    def cos_sim(self, a, b, name: str, axis: int = -1):
        with tf.name_scope(name):
            a = tf.nn.l2_normalize(a, axis=axis)
            b = tf.nn.l2_normalize(b, axis=axis)
            dot = tf.reduce_sum(a * b, axis=axis)
        return dot

    def forward(self):
        H = self.p_lkup  # (bs, tn, dw)
        H_mask = self.p_mask  # (bs, tn, 1)
        uas = [(self.dim_h, tanh), (self.num_c, None)]
        proj_H = instant_denses(H, uas, use_bias=self.use_bias, name='proj_H')  # (bs, tn, cn)
        with tf.name_scope('proj_H_mask'):
            proj_H += (H_mask - 1.) * 1e12
        A = softmax(proj_H, axis=1, name='A')  # (bs, tn, cn)
        M = tf.matmul(A, H, transpose_a=True, name='M')  # (bs, cn ,dw)

        C = tf.expand_dims(self.c_embed, axis=0, name='C')  # (1, cn, dw)
        C_T = tf.transpose(C, perm=[0, 2, 1], name='C_T')  # (1, dw, cn)
        CM_cos = self.cos_sim(C, M, name='CM_cos')  # (bs, cn)
        with tf.name_scope('CM_dist'):
            CM_dist = (1 - CM_cos) * 0.5  # (bs, cn)
        with tf.name_scope('CM_dist_alpha'):
            CM_dist_alpha = - self.ph_alpha * CM_dist  # (bs, cn)
        sig_H = softmax(CM_dist_alpha, axis=1, name='sig_H')  # (bs, cn)
        with tf.name_scope('J_1'):
            sigH_CMdist = tf.reduce_sum(sig_H * CM_dist, axis=-1)
            J_1 = tf.reduce_mean(sigH_CMdist)
        P_A = self.reg_diag(A, name='P_A')
        P_C = self.reg_diag(C_T, name='P_C')

        self.total_loss = tf.add_n([
            J_1,
            (P_A + P_C) * self.c_reg if self.c_reg > 1e-4 else 0.,
        ], name='total_loss')
        print('forward shapes', M.shape, A.shape, CM_dist.shape)

        self.pc_probs = CM_cos
        self.merge_mid = su.merge([
            histogram(name='A', values=A, family='encode'),
            histogram(name='M', values=M, family='encode'),
            histogram(name='CM_cos', values=CM_cos, family='decode'),
        ])
        self.merge_loss = su.merge([
            scalar(name='J_1', tensor=J_1, family='loss'),
            # scalar(name='P_A', tensor=P_A, family='loss'),
            # scalar(name='P_C', tensor=P_C, family='loss'),
            scalar(name='total_loss', tensor=self.total_loss, family='loss'),
        ])
        self.test_parts = [H, proj_H, A, M, ]
        # self.test_parts = [H, proj_H, CM_cos, sig_H, J_1]

    def define_optimizer(self):
        lr = tf.train.exponential_decay(
            learning_rate=self.lr, global_step=self.ph_global_step,
            decay_steps=200, decay_rate=0.99
        )
        optimizer = tf.train.AdamOptimizer(learning_rate=lr, name='ADDDDDdam')
        self.cross_op = optimizer.minimize(self.total_loss)

    def build(self, word_init, clu_init):
        self.define_word_embed(word_init)
        self.define_cluster_embed(clu_init)
        self.define_weight_params()
        self.define_inputs()
        self.forward()
        self.define_optimizer()

    """ runtime below """

    def get_fd(self, docarr: List[Document], is_train: bool) -> dict:
        p_seq = [doc.tokenids for doc in docarr]
        return {self.p_wids: p_seq, self.ph_is_train: is_train, self.ph_alpha: self.alpha}

    def predict(self, docarr: List[Document]) -> List[int]:
        pc_probs = self.run_parts(docarr, self.pc_probs)
        return np.argmax(pc_probs, axis=1).reshape(-1)

    def run_parts(self, docarr: List[Document], *parts):
        assert len(parts) > 0
        fd = self.get_fd(docarr, is_train=False)
        parts_res = self.sess.run(parts, feed_dict=fd)
        return parts_res if len(parts_res) > 1 else parts_res[0]

    def train_step(self, docarr: List[Document], epoch_i: int, batch_i: int, *args, **kwargs):
        step = epoch_i // self.span
        self.alpha = np.sign(step) * np.power(self.span, step - 6)
        fd = self.get_fd(docarr, is_train=True)
        fd[self.ph_global_step] = self.global_step
        self.sess.run(self.cross_op, feed_dict=fd)
        self.global_step += 1

    def run_merge(self, docarr: List[Document], merge):
        fd = self.get_fd(docarr, is_train=False)
        return self.sess.run(merge, feed_dict=fd)

    def set_session(self, sess):
        self.sess = sess

    def save(self, file: str):
        tf.train.Saver().save(self.sess, file)

    def load(self, file: str):
        tf.train.Saver().restore(self.sess, file)
