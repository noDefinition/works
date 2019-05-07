from .v1 import *


# noinspection PyAttributeOutsideInit
class T1(V1):
    def define_tags_embed(self, tags_embed):
        init = tf.constant_initializer(tags_embed)
        self.tags_embed = tf.get_variable('tags_embed', initializer=init, shape=tags_embed.shape)

    def define_inputs(self):
        super(T1, self).define_inputs()
        self.q_tags = tf.placeholder(i32, (None,), name='q_tags')
        self.t_lkup = tf.nn.embedding_lookup(self.tags_embed, self.q_tags, name='t_lkup ')

    def forward(self):
        # t_emb = self.
        a_emb = self.embed_rep(self.a_lkup, self.a_mask, name='a_emb', k=10)
        au_cat = tf.concat([a_emb, self.u_lkup], axis=-1, name='au_cat')  # (bs, dw)

        au_final = au_cat
        uas = [(self.dim_w, relu), (self.dim_w, relu), (1, None)]
        preds = instant_denses(au_final, uas, name='preds', w_reg=self.l2_reg)
        self.pred_probs = self.pred_scores = tf.squeeze(preds, axis=1, name='pred_scores')

    def define_optimizer(self):
        true_pw = pairwise_sub(self.true_scores, name='true_pw')
        ones_upper_tri = tf.matrix_band_part(tf.ones_like(true_pw), 0, -1, name='ones_upper_tri')
        true_sign = tf.sign(true_pw * ones_upper_tri, name='true_sign')
        pred_pw = pairwise_sub(self.pred_scores, name='pred_pw')
        margin_pw = tf.maximum(1. - pred_pw * true_sign, 0., name='margin_pw')

        margin_loss = tf.reduce_sum(margin_pw, name='margin_loss')
        if self.l2_reg is not None:
            reg_vars = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
            reg_loss = tf.add_n(reg_vars, name='reg_loss')
        else:
            reg_loss = tf.constant(0., name='reg_loss')

        self.total_loss = tf.add(margin_loss, reg_loss, name='total_loss')
        opt = tf.train.AdamOptimizer(learning_rate=self.lr, name='adam_opt')
        self.train_op = opt.minimize(self.total_loss, name='train_op')
