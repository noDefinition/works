from clu.me.ae.ae_base import *
from utils.doc_utils import get_tfidf


# noinspection PyAttributeOutsideInit
class AeSoft(AEBase):
    def get_hyper_params(self, args: CluArgs):
        super(AeSoft, self).get_hyper_params(args)
        # self.pretrain_step: int = args.ptn
        self.use_decoder: int = args.ptncmb
        # self.alpha: float = args.alpha
        # self.beta: float = args.beta
        # self.gamma: float = args.gamma

    def forward(self):
        with tf.name_scope('encoder'):
            # c_embed = tf.stop_gradient(self.c_embed)
            c_embed = self.c_embed
            # uas = [(self.dim_w * 4, relu), (self.dim_w * 4, relu), (self.dim_w, None)]
            # encode_z = instant_denses(self.p_tfidf, uas, name='encode_z')  # (bs, dw)
            encode_z = self.get_mean_pooling(self.p_lkup, self.p_mask, name='encode_z')
            pc_socre = tf.matmul(encode_z, c_embed, transpose_b=True)  # (bs, nc)
            pc_probs = tf.nn.softmax(pc_socre, axis=-1, name='pc_probs')  # (bs, nc)
            recon_z = tf.matmul(pc_probs, c_embed)

        with tf.name_scope('decoder'):
            uas = [(self.dim_w * 4, relu), (self.dim_w * 4, relu), (self.num_w, sigmoid)]
            decoder = build_denses(uas, name='decoder')
            decode_input = tf.concat([encode_z, recon_z], axis=-1)  # (bs, dw * 2)
            decode_tfidf = postpone_denses(decode_input, decoder, name='decode_score')
            recon_dist = - self.cos_sim(decode_tfidf, self.p_tfidf, name='recon_dist')
            recon_loss = tf.reduce_mean(recon_dist)

        with tf.name_scope('decoder_pre'):
            pre_decode_input = tf.concat([encode_z, encode_z], axis=-1)  # (bs, dw * 2)
            pre_decode_tfidf = postpone_denses(pre_decode_input, decoder, name='pre_decode_score')
            pre_recon_dist = - self.cos_sim(pre_decode_tfidf, self.p_tfidf, name='pre_recon_dist')
            pre_recon_loss = tf.reduce_mean(pre_recon_dist)

        self.encode_z = encode_z
        self.pc_probs = pc_probs
        self.train_loss = recon_loss
        self.pre_loss = pre_recon_loss
        self.merge_some = su.merge([
            histogram(name='encode_z', values=encode_z, family='encode'),
            histogram(name='pc_probs', values=pc_probs, family='encode'),
            histogram(name='decode_tfidf', values=decode_tfidf, family='decode'),
            histogram(name='recon_dist', values=recon_dist, family='decode'),
        ])
        self.merge_batch = su.merge([
            scalar(name='recon_loss', tensor=recon_loss, family='loss'),
        ])
        self.ph_c_new = tf.placeholder(f32, (None, None))
        self.c_assign = tf.assign(self.c_embed, self.ph_c_new, validate_shape=True)

    def define_optimizer(self):
        super(AeSoft, self).define_optimizer()
        self.pre_adam = tf.train.AdamOptimizer(learning_rate=self.lr, name='PRE_Adam')
        self.pre_op = self.pre_adam.minimize(self.pre_loss)

    """ runtime """

    def get_fd(self, docarr: List[Document], is_train: bool, **kwargs) -> dict:
        fd = super(AeSoft, self).get_fd(docarr, is_train, **kwargs)
        fd[self.p_tfidf] = get_tfidf(docarr)
        return fd

    def train_step(self, docarr: List[Document], epoch_i: int, batch_i: int, **kwargs):
        if not self.use_decoder:
            return
        fd = self.get_fd(docarr, is_train=True)
        # if epoch_i < self.pretrain_step:
        #     self.sess.run(self.pre_op, feed_dict=fd)
        # else:
        self.sess.run(self.train_op, feed_dict=fd)
        # self.sess.run(self.c_slide_assign, feed_dict=fd)
        self.global_step += 1

    def on_epoch_end(self, epoch_i: int, batches: List[List[Document]]):
        from utils.logger_utils import print_on_first
        # if epoch_i < self.pretrain_step - 1:
        #     print_on_first(epoch_i, '    ++ no operation')
        #     return
        # elif epoch_i == self.pretrain_step - 1:
        #     from sklearn.cluster import KMeans
        #     print_on_first(epoch_i, '    ++ do kmeans')
        #     z_list = [self.run_parts(batch, self.encode_z) for batch in batches]
        #     z = np.concatenate(z_list, axis=0)  # (bs, dw)
        #     kmeans = KMeans(n_clusters=self.num_c, n_init=20, n_jobs=6)
        #     kmeans.fit(z)
        #     self.assign_c_new(kmeans.cluster_centers_)
        # if epoch_i < self.pretrain_step:
        #     print_on_first(epoch_i, '    ++ no operation')
        #     return
        # else:
        print_on_first(epoch_i, '    ++ soft assign')
        z_list = []
        y_list = []
        for batch in batches:
            z, y = self.run_parts(batch, self.encode_z, self.pc_probs)
            z_list.append(z)
            y_list.append(y)
        z = np.concatenate(z_list, axis=0)  # (bs, dw)
        y = np.concatenate(y_list, axis=0)  # (bs, cn)
        # pd.DataFrame(data=y).to_csv('./y.csv')
        y_sum = np.expand_dims(np.sum(y, axis=0), axis=1)  # (cn, 1)
        yz_mul = np.expand_dims(y, axis=2) * np.expand_dims(z, axis=1)  # (bs, cn, dw)
        yz_sum = np.sum(yz_mul, axis=0)  # (cn, dw)
        c_new = yz_sum / np.maximum(y_sum, 1e-32)  # (cn, dw)
        self.sess.run(self.c_assign, feed_dict={self.ph_c_new: c_new})
