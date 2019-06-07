from clu.me.vae.vae1 import *


# noinspection PyAttributeOutsideInit
class VAE3(VAE1):
    def forward(self):
        # p_lkup = instant_denses(self.p_lkup, [(self.dim_h, None)], name='dense_plkup')
        # c_embed = instant_denses(self.c_embed, [(self.dim_h, None)], name='dense_cembed')
        # c_logvar = instant_denses(self.c_embed, [(self.dim_h, None)], name='dense_clogvar')
        p_lkup = self.p_lkup
        c_embed = self.c_embed
        c_logvar = self.c_logvar

        p_rep = self.get_doc_embed(p_lkup, name='p_rep')  # (bs, dw)
        pc_score = tf.matmul(p_rep, c_embed, transpose_b=True, name='pc_score')  # (bs, cn)
        pc_probs = tf.nn.softmax(pc_score, axis=1, name='pc_probs')  # (bs, cn)
        pc_recon = tf.matmul(pc_probs, c_embed, name='pc_recon')  # (bs, dw)
        back_stdvar = tf.exp(c_logvar * 0.5, name='back_stdvar')  # (bs, dw)
        pc_stdvar = tf.matmul(pc_probs, back_stdvar, name='pc_stdvar')  # (bs, dw)
        z_p = self.sample_z(pc_recon, pc_stdvar, name='z_p')  # (bs, dw)
        # z_p_d = instant_denses(z_p, [(self.dim_h, None)], name='z_p_dense')  # (bs, dh)
        z_p_d = z_p  # (bs, dw)

        # decode_inputs = p_lkup
        decode_inputs = tf.add(
            p_lkup, tf.expand_dims(z_p_d, axis=1, name='zpd_expand'), name='decode_inputs')
        decoder_output, _, _ = self.decode_lstm(decode_inputs, training=self.ph_is_train)
        uas = [(self.dim_h * 2, relu), (self.dim_h * 4, relu), (self.num_w, None)]
        decode_preds = instant_denses(decoder_output, uas, 'decode_preds')  # (bs, tn, nw)

        p_shift_hot = tf.one_hot(
            indices=self.p_shift, depth=self.num_w, name='p_shift_hot',
            on_value=self.smooth, off_value=(1 - self.smooth) / (self.num_w - 1))  # (bs, tn, nw)
        right_ce = tf.nn.softmax_cross_entropy_with_logits_v2(
            labels=p_shift_hot, logits=decode_preds, name='right_ce')  # (bs, tn)
        right_ce_mask = tf.multiply(right_ce, self.p_shift_mask, name='right_ce_mask')
        right_ce_loss = tf.reduce_mean(right_ce_mask, name='right_ce_loss')

        self.total_loss = right_ce_loss
        self.pc_probs = pc_probs

        self.merge_mid = su.merge([
            histogram(name='z_p', values=z_p, family='z'),
            histogram(name='z_p_d', values=z_p_d, family='z'),
            histogram(name='pc_probs', values=pc_probs, family='clu'),
            histogram(name='pc_recon', values=pc_recon, family='clu'),
            histogram(name='pc_stdvar', values=pc_stdvar, family='clu'),
            histogram(name='decode_preds', values=decode_preds, family='decode'),
        ])
        self.merge_loss = su.merge([
            scalar(name='total_loss', tensor=self.total_loss, family='loss'),
        ])

    def define_optimizer(self):
        lr = tf.train.exponential_decay(
            learning_rate=self.lr, global_step=self.global_step, decay_steps=200, decay_rate=0.99)
        optimizer = tf.train.AdamOptimizer(learning_rate=lr, name='AAAAdam')
        # optimizer = tf.train.AdagradOptimizer(learning_rate=lr, name='AAAAAAAAAdagrad')
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
