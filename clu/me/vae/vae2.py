from clu.me.vae.vae1 import *


# noinspection PyAttributeOutsideInit
class VAE2(VAE1):
    # reconstruct tf-idf

    def get_hyper_params(self, args):
        super(VAE2, self).get_hyper_params(args)
        self.reg_mu: bool = bool(args[C.regmu])  # 0:no reg 1:reg

    def define_weight_params(self):
        self.encode_lstm = CuDNNLSTM(self.dim_h, return_sequences=True)

    def define_inputs(self):
        super(VAE2, self).define_inputs()
        self.ph_tfidf = tf.placeholder(f32, (None, None), name='p_tfidf')  # (bs, nw)

    # def get_lstm_encode(self, inputs, mask, name):
    #     assert len(mask.shape) == 2
    #     encode_outputs = self.encode_lstm(inputs)  # (bs, tn, hd)
    #     with tf.name_scope(name):
    #         mask = tf.cast(mask, i32)  # (bs, tn)
    #         b_seq_len = tf.reduce_sum(mask, axis=1, keepdims=True)  # (bs, 1)
    #         output_slices = tf.batch_gather(encode_outputs, b_seq_len)  # (bs, 1, hd)
    #         output_slices = tf.squeeze(output_slices, axis=1)  # (bs, hd)
    #     return output_slices

    def get_mean_pooling(self, inputs, mask, name: str):
        with tf.name_scope(name):
            doc_len = tf.reduce_sum(mask, axis=1, keepdims=True)  # (bs, 1)
            # doc_len = tf.maximum(doc_len, tf.ones_like(doc_len))
            doc_emb = tf.reduce_sum(inputs, axis=1) / doc_len  # (bs, dw)
        return doc_emb  # (bs, dw)

    def get_pc_probs_softmax(self, doc_emb, clu_emb, name: str):
        with tf.name_scope(name):
            pc_score = tf.matmul(doc_emb, clu_emb, transpose_b=True, name='pc_score')  # (bs, cn)
            pc_probs = tf.nn.softmax(pc_score, axis=1, name='pc_probs')  # (bs, cn)
        return pc_probs

    def cos_sim(self, a, b, name: str, axis: int = -1):
        with tf.name_scope(name):
            a = tf.nn.l2_normalize(a, axis=axis)
            b = tf.nn.l2_normalize(b, axis=axis)
            dot = tf.reduce_sum(a * b, axis=axis)
        return dot

    def forward(self):
        # p_rep = self.get_lstm_encode(self.p_lkup, self.p_mask, name='p_rep')  # (bs, dw)
        p_rep = self.get_mean_pooling(self.p_lkup, None, name='p_rep')  # (bs, dw)
        pc_probs = self.get_pc_probs_softmax(p_rep, self.c_embed, name='pc_probs')  # (bs, cn)
        z_mu = tf.matmul(pc_probs, self.c_embed, name='z_mu')  # (bs, dw)
        # stdvar = tf.exp(self.c_logvar * 0.5, name='stdvar')  # (bs, dw)
        uas = [(self.dim_c, tanh), (self.dim_c, None)]
        z_logvar = instant_denses(p_rep, uas, 'z_logvar')  # (bs, dw)
        z_std = tf.exp(z_logvar * 0.5, name='z_std')  # (bs, dw)
        z_sample = self.sample_z(z_mu, z_std, name='z_sample')  # (bs, dw)

        uas = [(self.dim_h, relu), (self.num_w, None)]
        decode_tfidf = instant_denses(z_sample, uas, 'decode_tfidf')  # (bs, nw)
        decode_cos_sim = self.cos_sim(self.ph_tfidf, decode_tfidf, name='decode_cos_sim')
        # recon_cos_sim = tf.reduce_mean(decode_cos_sim, name='recon_cos_sim')

        decode_shift = tf.concat([decode_tfidf[2:, :], decode_tfidf[:2, :]], axis=0)
        shift_cos_sim = self.cos_sim(self.ph_tfidf, decode_shift, name='shift_cos_sim')
        # wrong_cos_sim = tf.reduce_mean(shift_cos_sim, name='wrong_cos_sim')

        margin = tf.maximum(0., 1. - decode_cos_sim + shift_cos_sim, name='pairwise_margin')
        margin_loss = tf.reduce_mean(margin, name='margin_loss')
        z_prior_kl = self.get_normal_kl(z_mu, z_std, reg_mu=self.reg_mu, name='z_prior_kl')
        self.train_loss = tf.add_n([
            margin_loss,
            z_prior_kl * self.coeff_kl if self.coeff_kl > 1e-5 else 0.,
        ], name='total_loss')

        self.pc_probs = pc_probs
        self.merge_mid = su.merge([
            histogram(name='pc_probs', values=pc_probs, family='clu'),
            histogram(name='z_mu', values=z_mu, family='sample'),
            histogram(name='z_std', values=z_std, family='sample'),
            histogram(name='z_sample', values=z_sample, family='sample'),
            histogram(name='decode_tfidf', values=decode_tfidf, family='decode'),
            histogram(name='decode_cos_sim', values=decode_cos_sim, family='decode'),
            histogram(name='shift_cos_sim', values=shift_cos_sim, family='decode'),
            histogram(name='margin', values=margin, family='decode'),
        ])
        self.merge_loss = su.merge([
            scalar(name='z_prior_kl', tensor=z_prior_kl, family='gen2'),
            scalar(name='margin_loss', tensor=margin_loss, family='gen2'),
            scalar(name='total_loss', tensor=self.train_loss, family='gen2'),
        ])

    """ runtime """

    def get_fd(self, docarr: List[Document], is_train: bool, **kwargs):
        from scipy.sparse import vstack
        p_seq = [doc.tokenids for doc in docarr]
        fd = {self.p_wids: p_seq, self.ph_is_train: is_train}
        # with_tfidf = kwargs.get('with_tfidf', True)
        # if with_tfidf:
        p_tfidf_list = [doc.tfidf for doc in docarr]
        p_tfidf = vstack(p_tfidf_list).todense()
        fd[self.ph_tfidf] = p_tfidf
        return fd

    def predict(self, docarr: List[Document]) -> List[int]:
        fd = self.get_fd(docarr, is_train=False, with_tfidf=False)
        pc_probs = self.sess.run(self.pc_probs, feed_dict=fd)
        return np.argmax(pc_probs, axis=1).reshape(-1)
