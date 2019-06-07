from .n5 import *


# noinspection PyAttributeOutsideInit,PyPep8Naming
class N10(N5):
    """ self attention """
    file = __file__

    def define_inputs(self):
        super(N10, self).define_inputs()
        ed, md = self.dim_w, self.dim_m
        ph, lk = tf.placeholder, tf.nn.embedding_lookup
        shape = (None,)
        self.p_did = ph(i32, shape=shape)
        self.n_dids = [ph(i32, shape=shape) for _ in range(self.neg_n)]
        self.d_embed = tf.get_variable('d_embed', shape=[30000, ed])

    def define_denses(self):
        super(N10, self).define_denses()
        ed, md, init = self.dim_w, self.dim_m, self.x_init
        # self.
        self.FFN = Denses(ed, [(ed, relu), (ed, None)], kernel=self.x_init)

    def multi_dim_att(self, w_emb):
        self_score = tf.matmul(w_emb, tf.transpose(w_emb, perm=[0, 2, 1]))
        diag_mask = - 1e32 * tf.expand_dims(tf.eye(tf.shape(self_score)[1]), axis=0)
        self_alpha = tf.nn.softmax(self_score + diag_mask, axis=2)
        w_res = tf.matmul(self_alpha, w_emb)
        return self_alpha, w_res
