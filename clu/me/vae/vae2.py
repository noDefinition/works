from clu.me.vae.vae1 import *


# noinspection PyAttributeOutsideInit
class VAE2(VAE1):
    def define_weight_params(self):
        super(VAE2, self).define_weight_params()
        self.encode_lstm = CuDNNLSTM(self.dim_h, return_sequences=True)

    def get_lstm_encode(self, inputs, mask, name):
        assert len(mask.shape) == 2
        with tf.name_scope(name):
            encode_outputs = self.encode_lstm(inputs)  # (bs, tn, hd)
            mask = tf.cast(mask, i32)  # (bs, tn)
            b_seq_len = tf.reduce_sum(mask, axis=1, keepdims=True)  # (bs, 1)
            output_slices = tf.batch_gather(encode_outputs, b_seq_len)  # (bs, 1, hd)
            output_slices = tf.squeeze(output_slices, axis=1)  # (bs, hd)
        return output_slices

    def forward(self):
        p_rep = self.get_lstm_encode(self.p_lkup, self.p_mask, name='p_rep')  # (bs, dw)
        pc_score = tf.matmul(p_rep, self.c_embed, transpose_b=True, name='pc_score')  # (bs, cn)
        pc_probs = tf.nn.softmax(pc_score, axis=1, name='pc_probs')  # (bs, cn)
        pc_recon = tf.matmul(pc_probs, self.c_embed, name='pc_recon')  # (bs, dw)
        back_stdvar = tf.exp(self.c_logvar * 0.5, name='back_stdvar')  # (bs, dw)
        pc_stdvar = tf.matmul(pc_probs, back_stdvar, name='pc_stdvar')  # (bs, dw)
        z_p = self.sample_z(pc_recon, pc_stdvar, name='z_p')  # (bs, ed)
        z_p_d = instant_denses(z_p, [(self.dim_h, None)], name='z_p_dense')  # (bs, hd)

        decoder_outputs, _, _ = self.decode_lstm(
            self.p_lkup, initial_state=[z_p_d, z_p_d], training=self.ph_is_train)
        uas = [(self.dim_m, tanh), (self.num_w, None)]
        decode_preds = instant_denses(decoder_outputs, uas, 'decode_preds')  # (bs, tn, nw)

        p_shift_hot = tf.one_hot(
            indices=self.p_shift, depth=self.num_w, name='p_shift_hot',
            on_value=self.smooth, off_value=(1 - self.smooth) / (self.num_w - 1))  # (bs, tn, nw)
        right_cross_ent = tf.nn.softmax_cross_entropy_with_logits_v2(
            labels=p_shift_hot, logits=decode_preds, name='cross_entropy')  # (bs, tn)
        right_cross_ent = tf.multiply(right_cross_ent, self.p_shift_mask, name='cross_entropy_mask')

        cross_loss = tf.reduce_mean(right_cross_ent, name='cross_loss')
        z_prior_kl = self.get_normal_kl(pc_recon, pc_stdvar, name='z_prior_kl') * self.coeff_kl
        self.total_loss = tf.add_n([cross_loss, z_prior_kl], name='total_loss')

        # self.z_p = z_p
        # self.z_p_d = z_p_d
        self.pc_probs = pc_probs
        self.decode_preds = decode_preds

        zp_hist = histogram(name='z_p', values=z_p, family='z')
        zpd_hist = histogram(name='z_p_d', values=z_p_d, family='z')
        pc_prob_hist = histogram(name='pc_probs', values=pc_probs, family='clu')
        pc_recon_hist = histogram(name='pc_recon', values=pc_recon, family='clu')
        pc_logvar_hist = histogram(name='pc_stdvar', values=pc_stdvar, family='clu')
        dp_hist = histogram(name='decode_preds', values=decode_preds, family='decode')
        loss_hist = scalar(name='cross_loss', tensor=cross_loss, family='loss')
        kl_hist = scalar(name='z_prior_kl', tensor=z_prior_kl, family='loss')
        tot_hist = scalar(name='total_loss', tensor=self.total_loss, family='loss')
        self.merge_loss = su.merge([loss_hist, kl_hist, tot_hist])
        self.merge_mid = su.merge([
            zp_hist, zpd_hist, dp_hist, pc_prob_hist, pc_recon_hist, pc_logvar_hist,
        ])
