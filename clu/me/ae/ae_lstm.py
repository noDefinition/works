from tensorflow.keras.layers import CuDNNLSTM
from clu.me.ae.ae_base import *
from utils.logger_utils import print_on_first


# noinspection PyAttributeOutsideInit
class AELstm(AEBase):
    def get_hyper_params(self, args):
        super(AELstm, self).get_hyper_params(args)
        self.alpha: float = args[C.alpha]
        self.smooth: float = args[C.lbsmt]
        self.pre_train_step: int = args[C.ptn]

    def define_weight_params(self):
        self.decode_lstm = CuDNNLSTM(self.dim_h, return_state=True, return_sequences=True)

    def define_inputs(self):
        super(AELstm, self).define_inputs()
        self.ph_is_pretrain = tf.placeholder_with_default(False, (), name='ph_is_pretrain')
        lk = tf.nn.embedding_lookup
        p_truncate = self.p_wids[:, 1:]  # (bs, tn-1), drop the first word
        p_padding = tf.zeros([tf.shape(self.p_wids)[0], 1], name='p_padding', dtype=i32)  # (bs, 1)
        self.p_lsh = tf.concat([p_truncate, p_padding], axis=1, name='p_lsh')  # (bs, tn)
        self.p_lsh_mask = self.get_mask(self.p_lsh, False, name='p_lsh_mask')  # (bs, tn)
        self.p_lsh_lkup = lk(self.w_embed, self.p_lsh, name='p_lsh_lkup')  # (bs, tn, dw)

    def forward(self):
        c_embed_T = tf.transpose(self.c_embed)  # (dw, cn)

        with tf.name_scope('encoder'):
            p_mean = self.get_mean_pooling(self.p_lkup, self.p_mask, name='p_mean')  # (bs, dw)
            pc_score = tf.matmul(p_mean, c_embed_T, name='pc_score')  # (bs, cn)
            pc_probs = tf.nn.softmax(pc_score, axis=1, name='pc_probs')  # (bs, cn)
            # p_recon = tf.matmul(pc_probs, self.c_embed)

        with tf.name_scope('word_cluster'):
            wc_score = tf.tensordot(self.p_lkup, c_embed_T, axes=1)  # (bs, tn, cn)
            wc_probs = tf.nn.softmax(wc_score, axis=-1)  # (bs, tn, cn)
            # pc_probs_e = tf.expand_dims(pc_probs, axis=1)  # (bs, 1, cn)
            # pwc_probs = self.alpha * pc_probs_e + (1 - self.alpha) * wc_probs  # (bs, tn, cn)
            wc_recon = tf.tensordot(wc_probs, self.c_embed, axes=1)  # (bs, tn, dw)

        with tf.name_scope('decoder'):
            # inputs_concat = tf.concat([self.p_lkup, wc_probs], axis=-1)  # (bs, tn, dw+dw)
            inputs_concat = self.alpha * self.p_lkup + (1 - self.alpha) * wc_recon  # (bs, tn, dw)
            inputs_concat_sg = tf.stop_gradient(inputs_concat)  # (bs, dw + cn)
            decode_inputs = tf.cond(
                self.ph_is_pretrain,
                true_fn=lambda: inputs_concat_sg,
                false_fn=lambda: inputs_concat,
            )
            decoder_output, _, _ = self.decode_lstm(decode_inputs)
            # uas = [(self.dim_h, tanh), (self.dim_h * 2, tanh), (self.num_w, None)]
            uas = [(self.num_w, None)]
            decode_preds = instant_denses(decoder_output, uas, 'decode_preds')  # (bs, tn, nw)

        with tf.name_scope('decoder_loss'):
            decode_label = tf.one_hot(  # (bs, tn, nw)
                indices=self.p_lsh, depth=self.num_w, name='decode_label',
                on_value=self.smooth, off_value=(1 - self.smooth) / (self.num_w - 1))
            ce = tf.nn.softmax_cross_entropy_with_logits_v2(
                labels=decode_label, logits=decode_preds, name='ce')  # (bs, tn)
            ce_masked = tf.multiply(ce, self.p_lsh_mask, name='ce_masked')  # (bs, tn)
            ce_loss = tf.reduce_mean(ce_masked, name='ce_loss')

        self.train_loss = ce_loss
        self.pc_probs = pc_probs
        # self.pwc_probs = tf.reduce_mean(pwc_probs, axis=1)  # (bs, cn)
        self.merge_mid = su.merge([
            histogram(name='p_mean', values=p_mean, family='encoder'),
            histogram(name='pc_probs', values=pc_probs, family='encoder'),
            histogram(name='wc_probs', values=wc_probs, family='word_cluster'),
            # histogram(name='pwc_probs', values=pwc_probs, family='word_cluster'),
            histogram(name='decode_preds', values=decode_preds, family='decoder'),
        ])
        self.merge_loss = su.merge([
            scalar(name='ce_loss', tensor=ce_loss, family='decoder'),
            scalar(name='total_loss', tensor=self.train_loss, family='comprehensive'),
        ])

    # def define_optimizer(self):
    #     self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr, name='AAdam_spec')
    #     self.train_op = self.optimizer.minimize(self.train_loss, name='train_op')
    #     self.encode_op = self.optimizer.minimize(self.encode_loss, name='encode_op')
    #     self.decode_op = self.optimizer.minimize(self.decode_loss, name='decode_op')

    def train_step(self, docarr: List[Document], epoch_i: int, batch_i: int, **kwargs):
        fd = self.get_fd(docarr, is_train=True)
        if epoch_i < self.pre_train_step:
            print_on_first(epoch_i, '   <--> pretraining')
            fd[self.ph_is_pretrain] = True
        print_on_first(epoch_i, '   >--< full train')
        self.sess.run(self.train_op, feed_dict=fd)
        self.global_step += 1
