from clu.me.vae.vae1 import *


# noinspection PyAttributeOutsideInit
class VAE4(VAE1):
    # sample before reconstruct

    def __init__(self, args):
        super(VAE4, self).__init__(args)
        self.position_sample: int = args[C.psmp]  # 0:pre  1:post

    @staticmethod
    def get_pc_probs_softmax(doc_emb, clu_emb):
        with tf.name_scope('get_pc_probs_softmax'):
            pc_score = tf.matmul(doc_emb, clu_emb, transpose_b=True, name='pc_score')  # (bs, cn)
            pc_probs = tf.nn.softmax(pc_score, axis=1, name='pc_probs')  # (bs, cn)
        return pc_probs

    def forward(self):
        p_lkup = self.p_lkup
        c_embed = self.c_embed
        dim_h = int(c_embed.shape[-1])
        p_rep = self.get_doc_embed(p_lkup, name='p_rep')  # (bs, dw)
        uas = [(self.dim_h, tanh), (self.dim_h, tanh), (dim_h * 2, None)]

        if self.position_sample == 0:
            z_mu_var = instant_denses(p_rep, uas, name='z_mu_var')  # (bs, dw)
            z_mu, z_log_var = z_mu_var[:, dim_h:], z_mu_var[:, :dim_h]  # (bs, dw), (bs, dw)
            z_std = tf.exp(z_log_var / 2, name='z_std')  # (bs, dw)
            z_sample = self.sample_z(z_mu, z_std, name='z_sample')  # (bs, dw)
            pc_probs = self.get_pc_probs_softmax(z_sample, c_embed)
            pc_recon = tf.matmul(pc_probs, c_embed, name='pc_recon')  # (bs, dw)
            decode_bias = pc_recon
        elif self.position_sample == 1:
            pc_probs = self.get_pc_probs_softmax(p_rep, c_embed)
            pc_recon = tf.matmul(pc_probs, c_embed, name='pc_recon')  # (bs, dw)
            z_mu_var = instant_denses(pc_recon, uas, name='z_mu_var')  # (bs, dw)
            z_mu, z_log_var = z_mu_var[:, dim_h:], z_mu_var[:, :dim_h]  # (bs, dw), (bs, dw)
            z_std = tf.exp(z_log_var / 2, name='z_std')  # (bs, dw)
            z_sample = self.sample_z(z_mu, z_std, name='z_sample')  # (bs, dw)
            decode_bias = z_sample
        else:
            raise ValueError('invalid psmp:', self.position_sample)

        decode_bias = tf.expand_dims(decode_bias, axis=1, name='zpd_expand')  # (bs, 1, dw)
        decode_inputs = tf.add(p_lkup, decode_bias, name='decode_inputs')  # (bs, tn, dw)
        decoder_output, _, _ = self.decode_lstm(decode_inputs, training=self.ph_is_train)
        uas = [(self.dim_h, relu), (self.dim_h * 4, relu), (self.num_w, None)]
        decode_preds = instant_denses(decoder_output, uas, 'decode_preds')  # (bs, tn, nw)

        p_shift_hot = tf.one_hot(
            indices=self.p_shift, depth=self.num_w, name='p_shift_hot',
            on_value=self.smooth, off_value=(1 - self.smooth) / (self.num_w - 1))  # (bs, tn, nw)
        right_ce = tf.nn.softmax_cross_entropy_with_logits_v2(
            labels=p_shift_hot, logits=decode_preds, name='right_ce')  # (bs, tn)
        right_ce_mask = tf.multiply(right_ce, self.p_shift_mask, name='right_ce_mask')
        right_ce_loss = tf.reduce_mean(right_ce_mask, name='right_ce_loss')

        z_prior_kl = self.get_normal_kl(z_mu, z_std, name='z_prior_kl')

        self.total_loss = tf.add_n([
            right_ce_loss,
            self.coeff_kl * z_prior_kl,
        ], name='total_loss')
        self.pc_probs = pc_probs
        self.merge_mid = su.merge([
            histogram(name='pc_probs', values=pc_probs, family='clu'),
            histogram(name='pc_recon', values=pc_recon, family='clu'),
            histogram(name='z_mu', values=z_mu, family='sample'),
            histogram(name='z_std', values=z_std, family='sample'),
            histogram(name='z_p', values=z_sample, family='sample'),
            histogram(name='decode_preds', values=decode_preds, family='decode'),
        ])
        self.merge_loss = su.merge([
            scalar(name='total_loss', tensor=self.total_loss, family='loss'),
        ])

    def define_optimizer(self):
        lr = tf.train.exponential_decay(
            learning_rate=self.lr, global_step=self.global_step, decay_steps=200, decay_rate=0.99)
        optimizer = tf.train.AdamOptimizer(learning_rate=lr, name='AAAAdam')
        var_list = list()
        for var in list(tf.trainable_variables()):
            print(var.name, end='  ')
            # if not ('_emb' in var.name or '_log' in var.name):
            if True:
                var_list.append(var)
                print('added')
            else:
                print()
        # exit()
        self.cross_op = optimizer.minimize(self.total_loss, var_list=var_list, name='cross_op')
