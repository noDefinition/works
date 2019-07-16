import numpy as np
from clu.me import C
from utils.deep.funcs import *
from utils.doc_utils import Document


# noinspection PyAttributeOutsideInit
class BaseAE(object):
    def __init__(self, args: dict):
        self.x_init = tf.contrib.layers.xavier_initializer()
        self.global_step = 0
        self.debug_parts = []
        self.sess = None
        self.pc_probs = None
        self.train_op = None
        self.train_loss = None
        self.get_hyper_params(args)

    def get_hyper_params(self, args):
        self.lr: float = args[C.lr]
        self.dim_h: int = args[C.hd]
        self.c_init: int = args[C.cini]
        self.w_init: int = args[C.wini]

    def define_cluster_embed(self, clu_init):
        self.num_c, self.dim_c = clu_init.shape
        init = [self.x_init, tf.constant_initializer(clu_init)][self.c_init]
        self.c_embed = tf.get_variable('c_embed', clu_init.shape, initializer=init)  # (cn, dw)

    def define_word_embed(self, word_init):
        self.num_w, self.dim_w = word_init.shape
        self.num_w += 1
        init = [self.x_init, tf.constant_initializer(word_init)][self.w_init]
        emb = tf.get_variable('w_kernel', word_init.shape, initializer=init)  # (wn, dw)
        pad = tf.zeros([1, word_init.shape[1]], name='w_padding', dtype=f32)  # (1, dw)
        self.w_embed = tf.concat([pad, emb], axis=0, name='w_embed')  # (wn+1, dw)

    def define_weight_params(self):
        pass

    def define_inputs(self):
        lk = tf.nn.embedding_lookup
        # self.ph_if_sample = tf.placeholder(tf.bool, (), name='if_sample')
        # self.ph_drop_keep = tf.placeholder_with_default(1., (), name='dropout_keep')
        self.ph_is_train = tf.placeholder(tf.bool, (), name='is_train')
        self.p_wids = tf.placeholder(i32, (None, None), name='p_seq')  # (bs, tn)
        self.p_mask = self.get_mask(self.p_wids, False, name='p_mask')  # (bs, tn)
        self.p_lkup = lk(self.w_embed, self.p_wids, name='p_lkup')  # (bs, tn, dw)

    @staticmethod
    def get_mask(wids, expand_last: bool, name: str):
        with tf.name_scope(name):
            mask = tf.cast(tf.not_equal(wids, 0), dtype=f32)  # (bs, tn)
            if expand_last:
                mask = tf.expand_dims(mask, axis=-1)  # (bs, tn, 1)
        return mask

    @staticmethod
    def get_pc_probs_softmax(doc_emb, clu_emb, name: str):
        with tf.name_scope(name):
            pc_score = tf.matmul(doc_emb, clu_emb, transpose_b=True, name='pc_score')  # (bs, cn)
            pc_probs = tf.nn.softmax(pc_score, axis=1, name='pc_probs')  # (bs, cn)
        return pc_probs

    @staticmethod
    def cos_sim(a, b, name: str, axis: int = -1):
        with tf.name_scope(name):
            a = tf.nn.l2_normalize(a, axis=axis)
            b = tf.nn.l2_normalize(b, axis=axis)
            dot = tf.reduce_sum(a * b, axis=axis)
        return dot

    @staticmethod
    def get_mean_pooling(inputs, mask, name: str):
        with tf.name_scope(name):
            doc_len = tf.reduce_sum(mask, axis=1, keepdims=True)  # (bs, 1)
            doc_emb = tf.reduce_sum(inputs, axis=1) / doc_len  # (bs, dw)
        return doc_emb  # (bs, dw)

    def forward(self):
        raise ValueError('NEED IMPLEMENTATION')

    def define_optimizer(self):
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr, name='BaseAE_Adam')
        self.train_op = self.optimizer.minimize(self.train_loss)

    def build(self, word_init: np.ndarray, clu_init: np.ndarray):
        self.define_word_embed(word_init)
        self.define_cluster_embed(clu_init)
        self.define_weight_params()
        self.define_inputs()
        self.forward()
        self.define_optimizer()

    """ runtime below """

    def get_fd(self, docarr: List[Document], is_train: bool, **kwargs) -> dict:
        p_seq = [doc.tokenids for doc in docarr]
        return {self.p_wids: p_seq, self.ph_is_train: is_train}

    def predict(self, docarr: List[Document]) -> List[int]:
        pc_probs = self.run_parts(docarr, self.pc_probs)
        return np.argmax(pc_probs, axis=1).reshape(-1)

    def train_step(self, docarr: List[Document], epoch_i: int, batch_i: int, **kwargs):
        fd = self.get_fd(docarr, is_train=True)
        self.sess.run(self.train_op, feed_dict=fd)
        self.global_step += 1

    def run_parts(self, docarr: List[Document], *parts):
        assert len(parts) > 0
        fd = self.get_fd(docarr, is_train=False)
        parts_res = self.sess.run(parts, feed_dict=fd)
        return parts_res if len(parts_res) > 1 else parts_res[0]

    def run_merge(self, docarr: List[Document], merge):
        fd = self.get_fd(docarr, is_train=False)
        return self.sess.run(merge, feed_dict=fd)

    def set_session(self, sess):
        self.sess = sess

    def save(self, file: str):
        tf.train.Saver().save(self.sess, file)

    def load(self, file: str):
        tf.train.Saver().restore(self.sess, file)
