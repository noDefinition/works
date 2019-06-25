from clu.me.vae.vae1 import *


# noinspection PyAttributeOutsideInit
class VAE3(VAE1):
    def __init__(self, args):
        super(VAE3, self).__init__(args)
        self.prob_type: int = args[C.ptp]  # 0:softmax  1:kernel
        self.kernel_alpha: float = args[C.kalp] if self.prob_type == 1 else None

    def get_pc_probs_kernel(self, doc_emb, clu_emb):
        with tf.name_scope('pc_probs'):
            doc_emb_exp = tf.expand_dims(doc_emb, axis=1)  # (bs, 1, dw)
            clu_emb_exp = tf.expand_dims(clu_emb, axis=0)  # (1, cn, dw)
            doc_clu_sub = doc_emb_exp - clu_emb_exp  # (bs, cn, dw)
            kernel = tf.norm(doc_clu_sub, axis=-1, keepdims=False)  # (bs, cn)
            kernel = kernel / self.kernel_alpha + 1.0  # (bs, cn)
            kernel = tf.pow(kernel, - (self.kernel_alpha + 1) / 2)  # (bs, cn)
            kernel_sum = tf.reduce_sum(kernel, axis=-1, keepdims=True)  # (bs, 1)
            pc_probs = kernel / kernel_sum  # (bs, cn)
        return pc_probs

    def get_pc_probs_softmax(self, doc_emb, clu_emb):
        with tf.name_scope('pc_probs'):
            pc_score = tf.matmul(doc_emb, clu_emb, transpose_b=True, name='pc_score')  # (bs, cn)
            pc_probs = tf.nn.softmax(pc_score, axis=1, name='pc_probs')  # (bs, cn)
        return pc_probs

    def forward(self):
        # p_lkup = instant_denses(self.p_lkup, [(self.dim_h, None)], name='dense_plkup')
        # c_embed = instant_denses(self.c_embed, [(self.dim_h, None)], name='dense_cembed')
        # c_logvar = instant_denses(self.c_embed, [(self.dim_h, None)], name='dense_clogvar')
        p_lkup = self.p_lkup
        c_embed = self.c_embed
        c_logvar = self.c_logvar

        p_rep = self.get_doc_embed(p_lkup, name='p_rep')  # (bs, dw)
        if self.prob_type == 0:
            pc_probs = self.get_pc_probs_softmax(p_rep, c_embed)
        # elif self.prob_type == 1:
        #     pc_probs = self.get_pc_probs_kernel(p_rep, c_embed)
        else:
            raise ValueError('invalid prob_type:', self.prob_type)
        pc_recon = tf.matmul(pc_probs, c_embed, name='pc_recon')  # (bs, dw)
        back_stdvar = tf.exp(c_logvar * 0.5, name='back_stdvar')  # (bs, dw)
        pc_stdvar = tf.matmul(pc_probs, back_stdvar, name='pc_stdvar')  # (bs, dw)
        z_p = self.sample_z(pc_recon, pc_stdvar, name='z_p')  # (bs, dw)
        # z_p_d = instant_denses(z_p, [(self.dim_h, None)], name='z_p_dense')  # (bs, dh)
        # z_p_d = z_p  # (bs, dw)

        z_p_d = tf.expand_dims(z_p, axis=1, name='zpd_expand')  # (bs, 1, dw)
        decode_inputs = tf.add(p_lkup, z_p_d, name='decode_inputs')  # (bs, tn, dw)
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

        decode_1, decode2 = decode_preds[1:, :, :], decode_preds[:1, :, :]
        wrong_decode_preds = tf.concat([decode_1, decode2], axis=0, name='wrong_decode_preds')
        wrong_ce = tf.nn.softmax_cross_entropy_with_logits_v2(
            labels=p_shift_hot, logits=wrong_decode_preds, name='wrong_ce')  # (bs, tn)
        wrong_ce_mask = tf.multiply(wrong_ce, self.p_shift_mask, name='wrong_ce_mask')
        wrong_ce_loss = tf.reduce_mean(wrong_ce_mask, name='wrong_ce_loss')

        self.total_loss = tf.add_n([
            right_ce_loss,
            - self.coeff_kl * wrong_ce_loss,
        ], name='total_loss')
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
