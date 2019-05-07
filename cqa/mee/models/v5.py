from .v1 import *


# noinspection PyAttributeOutsideInit
class V5(V1):
    """ transformer式 用户-单词 + 问题-答案 """

    def forward(self):
        a_lkup, a_mask = self.a_lkup, self.a_mask  # (bs, la, dw), (bs, la, 1)
        u_lkup = tf.expand_dims(self.u_lkup, axis=1, name='u_lkup')  # (bs, 1, dw)
        ua_lkup = tf.concat([u_lkup, a_lkup], axis=1, name='ua_lkup')  # (bs, 1+la, dw)
        u_mask = tf.ones([tf.shape(a_mask)[0], 1, 1], name='u_mask')  # (bs, 1, 1)
        ua_mask = tf.concat([u_mask, a_mask], axis=1, name='ua_mask')  # (bs, 1+la, 1)

        # ua_norm = tf.layers.batch_normalization(ua_lkup, training=self.is_train, momentum=0.5)
        uas = [(self.dim_w, None)]
        ua_lkup = instant_denses(ua_lkup, uas, 'ua_lkup_dense', w_reg=self.l2_reg)
        mh = MultiHeadLayer('ua_multi_head', w_init=self.x_init, w_reg=self.l2_reg)
        self_attention = mh.apply(ua_lkup, ua_lkup, ua_lkup, 8, 'ua_lkup_multi_head', ua_mask)
        self.mask_h, self.heads_prb, ua_satt = self_attention
        # ua_rep = ua_satt[:, 0, :] + self.u_lkup  # (bs, dw)
        # ua_rep = ua_satt[:, 0, :]  # (bs, dw)
        ua_rep = self.top_k_mean(ua_satt[:, 1:, :], 10, name='ua_rep_mean_pool')
        print('ua_rep.shape', ua_rep.shape)
        # # ua_norm = tf.layers.batch_normalization(ua_rep, training=self.is_train, momentum=0.5)
        # ua_rep_ffn = denses(ua_rep, [(self.dim_w, None)], 'ua_rep_ffn')  # (bs, dw)
        # ua_rep_1 = ua_rep + ua_rep_ffn  # (bs, dw)

        # q_avg = self.embed_rep(q_lkup, q_mask, 'q_emb_rep')  # (1, dw)
        # qua_cat = tf.expand_dims(tf.concat([q_avg, ua_rep_1], axis=0), axis=0)  # (1, 1+bs, dw)
        # # qua_norm = tf.layers.batch_normalization(qua_cat, training=self.is_train, momentum=0.5)
        # qua_satt = multi_head_self_attn(qua_cat, None, num_h=8)  # (1, 1+bs, dw)
        # qua_rep_0 = qua_satt + qua_cat  # (1, 1+bs, dw)

        # # qua_norm = tf.layers.batch_normalization(qua_rep_0, training=self.is_train, momentum=0.5)
        # qua_fnn = denses(qua_rep_0, [(self.dim_w, None)], 'qua_rep_fnn')  # (1, 1+bs, dw)
        # qua_rep_1 = qua_rep_0 + qua_fnn  # (1, 1+bs, dw)

        uas = [(self.dim_w, relu), (1, None)]
        pred_score = instant_denses(ua_rep, uas, 'pred_score_dense', w_reg=self.l2_reg)
        self.pred_probs = tf.nn.softmax(tf.squeeze(pred_score, axis=1))
        self.true_probs = tf.nn.softmax(self.true_scores / self.temper)

    def define_optimizer(self):
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            opt = tf.train.AdamOptimizer(learning_rate=self.lr)
            self.train_op = opt.minimize(self.total_loss)

    def fuck_mask_prob(self, ql, al, ul, vl):
        if not hasattr(self, 'masks'):
            self.masks = list()
        if not hasattr(self, 'probs'):
            self.probs = list()
        mask_h, h_probs = self.sess.run(
            [self.mask_h, self.heads_prb], feed_dict=self.get_fd(ql, al, ul))
        self.masks.append(mask_h)
        self.probs.append(h_probs)
