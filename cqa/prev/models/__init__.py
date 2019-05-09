import dataset
import tensorflow as tf

import utils
from utils import logger

log = logger.log

q_maxlen = 20
a_maxlen = 200
nb_features = 7

from multiprocessing import Process, Queue


def iter_and_put(it, queue):
    print('sub precess {} start!'.format(os.getpid()))
    for i in it:
        queue.put(i)


@tf.custom_gradient
def scale_grad(x):
    def grad(dy):
        return 10 * dy

    return x, grad


def merge_data(data_list):
    ret = {}
    keys = list(data_list[0].keys())
    for k in keys:
        ret[k] = []
    for d in data_list:
        for k in keys:
            ret[k].extend(d[k])
    return ret


class BasicPair:
    add_global_args = True
    Data = dataset.tf_data
    tc = {}

    def init(self):
        self.good_log = False
        self.tb_dirs = []
        self.l2_weights = []
        self.data = self.Data(self.dataset)

    def get_initializer(self):
        if self.args.init_f == 'normal':
            return tf.random_normal_initializer(stddev=self.args.init_v)
        return tf.random_uniform_initializer(minval=-self.args.init_v, maxval=self.args.init_v)

    def user_emb(self, u, name='UserEmb'):
        trainable = True
        trainable = False

        if self.args.init_user == 'random':
            initializer = self.get_initializer()
        else:
            initializer = tf.constant_initializer(self.data.user2vec)

        with tf.variable_scope(name):
            emb_w = tf.get_variable(
                name='emb_w',
                shape=(self.data.nb_users, self.args.dim_k),
                initializer=initializer,
                trainable=trainable,
            )
            if self.args.train_user == 'train':
                self.emb_ws.add(emb_w)
            elif self.args.train_user == 'dense':
                emb_w = tf.layers.dense(
                    emb_w,
                    self.args.dim_k,
                    name='MapUserVec',
                )

            if self.phase == 'pos':
                tf.summary.histogram('user_emb_w', emb_w)

            o = tf.nn.embedding_lookup(emb_w, u)
        return o

    def text_embs(self, t, dim_k=None, init=None, train=None, name='TextEmb'):
        if init is None:
            init = self.args.init_word
        if train is None:
            train = self.args.train_word
        if dim_k is None or init == 'word2vec':
            dim_k = self.args.dim_k
        trainable = False
        if init == 'random':
            initializer = self.get_initializer()
        else:
            initializer = tf.constant_initializer(self.data.wordVec)
        # initializer = tf.random_uniform_initializer(minval = 0, maxval = 0.1)
        with tf.variable_scope(name):
            pad = tf.zeros((1, dim_k))
            emb_w = tf.get_variable(
                name='emb_w',
                shape=(self.data.nb_words, dim_k),
                initializer=initializer,
                trainable=trainable,
            )
            if train == 'train':
                self.emb_ws.add(emb_w)
            elif train == 'dense':
                emb_w = tf.layers.dense(emb_w, dim_k, name='MapWordVec')

            if self.phase == 'pos':
                tf.summary.histogram('text_emb_w', emb_w)

            emb_w = tf.concat([pad, emb_w], 0)

            if self.args.text_emb_relu:
                emb_w = tf.nn.relu(emb_w)
            o = tf.nn.embedding_lookup(emb_w, t)
            mask = tf.cast(tf.not_equal(t, 0), tf.float32)
            mask = tf.expand_dims(mask, -1)

        return o, mask

    def l2_loss(self, l2):
        return lambda w: l2 * tf.nn.l2_loss(w)

    def add_l2(self):
        return None
        l2 = tf.losses.get_regularization_loss()
        if self.l2_weights:
            l2 += tf.add_n([self.args.l2_w * tf.nn.l2_loss(w) for w in self.l2_weights])
        return l2

    ''' *******************************************funcs begin ***********************************'''
    ''' *******************************************funcs begin ***********************************'''
    ''' *******************************************funcs begin ***********************************'''

    def mask_max(self, t, mask):
        # (bs, tl, k), (bs, tl)
        with tf.variable_scope('mask_max'):
            t = t * mask
            t = t - (1 - mask) * 1e30
            m = tf.reduce_max(t, -2)
        return m

    def mask_sum(self, t, mask):
        # (bs, tl, k), (bs, tl)
        with tf.variable_scope('mask_sum'):
            s = tf.reduce_sum(t * mask, -2)
        return s

    def mask_mean(self, t, mask):
        with tf.variable_scope('mask_mean'):
            s = self.mask_sum(t, mask)
            d = tf.reduce_sum(mask, -2)
        return s / d

    def top_k(self, t, k):
        with tf.variable_scope('TopK'):
            t = tf.transpose(t, [0, 2, 1])
            t = tf.nn.top_k(t, k).values
            t = tf.transpose(t, [0, 2, 1])
        return t

    def top_k_sum(self, t, k):
        with tf.variable_scope('TopKSum'):
            t = self.top_k(t, k)  # (bs, k, dim_k)
            s = tf.reduce_sum(t, -2)  # (bs, dim_k)
        return s

    def mask_top_k_sum(self, t, mask, k):
        # t:(bs, tl, dim_k), mask: (bs, tl, 1)
        with tf.variable_scope('MaskTopKSum'):
            t = t * mask
            t = t - (1 - mask) * 1e30
            t = self.top_k(t, k)  # (bs, k, dim_k)
            m = self.top_k(mask, k)  # (bs, k, 1)
            t = t * m
            s = tf.reduce_sum(t, -2)  # (bs, dim_k)
        return s

    def top_k_mean(self, t, k):
        with tf.variable_scope('TopKMean'):
            t = self.top_k(t, k)  # (bs, k, dim_k)
            s = tf.reduce_mean(t, -2)  # (bs, dim_k)
        return s

    def mask_top_k_mean(self, t, mask, k):
        # t:(bs, tl, dim_k), mask: (bs, tl, 1)
        with tf.variable_scope('MaskTopKMeam'):
            t = t * mask
            t = t - (1 - mask) * 1e30
            t = self.top_k(t, k)  # (bs, k, dim_k)
            m = self.top_k(mask, k)  # (bs, k, 1)
            s = self.mask_mean(t, m)  # (bs, dim_k)
        return s

    def lstm(self, t):
        with tf.variable_scope('LSTM'):
            cell = tf.nn.rnn_cell.LSTMCell(
                self.args.dim_k,
                initializer=self.get_initializer(),
                # reuse = True,
                name='lstm_cell',
            )
            seq, final_state = tf.nn.dynamic_rnn(cell, t, dtype=tf.float32)
        return seq

    def features_rep(self, f, dim_k=None):
        if dim_k is None:
            dim_k = self.args.dim_k
        with tf.variable_scope('FeaturesRep'):
            f = tf.layers.batch_normalization(f)
            f = tf.layers.dense(
                f,
                dim_k,
                # activation = tf.nn.relu,
                name='featuresT',
            )
            return f

    def text_rep(self, t, mask):
        if self.args.text_mode == 'top_k_sum':
            t = self.top_k_sum(t, self.args.top_k)
        elif self.args.text_mode == 'mask_top_k_sum':
            t = self.mask_top_k_sum(t, mask, self.args.top_k)
        elif self.args.text_mode == 'top_k_mean':
            t = self.top_k_mean(t, self.args.top_k)
        elif self.args.text_mode == 'mask_top_k_mean':
            t = self.mask_top_k_mean(t, mask, self.args.top_k)
        elif self.args.text_mode == 'max':
            t = tf.reduce_max(t, -2)
        elif self.args.text_mode == 'sum':
            t = tf.reduce_sum(t, -2)
        elif self.args.text_mode == 'mean':
            t = tf.reduce_mean(t, -2)
        elif self.args.text_mode == 'mask_max':
            t = self.mask_max(t, mask)
        elif self.args.text_mode == 'mask_sum':
            t = self.mask_sum(t, mask)
        elif self.args.text_mode == 'mask_mean':
            t = self.mask_mean(t, mask)
        elif self.args.text_mode == 'mean-max':
            t = tf.expand_dims(t * mask, -1)
            t = tf.layers.average_pooling2d(t, (self.args.top_k, 1), (1, 1))
            t = tf.squeeze(t, [-1])
            t = tf.reduce_max(t, -1)
        elif self.args.text_mode == 'max-mean':
            t = tf.expand_dims(t * mask, -1)
            t = tf.layers.max_pooling2d(t, (self.args.top_k, 1), (1, 1))
            t = tf.squeeze(t, [-1])
            t = tf.reduce_mean(t, -1)
        else:
            print('Error');
            input()
        return t

    def pair_mul_add(self, ts):
        with tf.variable_scope('PairWise'):
            n = len(ts)
            o = tf.add_n([ts[i] * ts[j] for i in range(n - 1) for j in range(i + 1, n)])
        return o

    def dot(self, a, b, axis=-1):
        return tf.reduce_sum(a * b, axis)

    def cos(self, a, b, axis=-1):
        ab = self.dot(a, b, axis)
        aa = tf.sqrt(self.dot(a, a, axis))
        bb = tf.sqrt(self.dot(b, b, axis))
        return ab / (aa * bb + 1e-7)

    ''' *******************************************funcs over ***********************************'''
    ''' *******************************************funcs over ***********************************'''

    def TopLayer(self, o):
        shape = o.shape
        if len(shape) == 1:
            return o
        if len(shape) != 2:
            log('ndim mast be 1 or 2, but got shape: {}'.format(shape), red=True)
        if shape[-1] == 1:
            return tf.reshape(o, (-1,))
        with tf.variable_scope('TopLayer'):
            for i, n in enumerate(self.args.fc):
                # o = tf.layers.batch_normalization(o)
                o = tf.layers.dense(o, n, activation=tf.nn.relu, name='dense_{}'.format(i), )
                if 0 < self.args.dp < 1:
                    o = tf.layers.dropout(o, rate=self.args.dp)
            # o = tf.layers.batch_normalization(o)
            o = tf.layers.dense(o, 1, name='dense_{}'.format(len(self.args.fc)), )
            # use_bias = False,
            o = tf.reshape(o, (-1,))
        return o

    def MLP(self, x, units, activation=tf.nn.relu, name='MLP'):
        with tf.variable_scope(name):
            for i, n in enumerate(units):
                x = tf.layers.dense(
                    x,
                    n,
                    activation=activation,
                    name='dense_{}'.format(i),
                )
        return x

    def get_question_rep(self, q):
        return None
        # return self.text_embs(q)

    def rep(self, ans, user):
        q, q_mask = self.question_rep
        q = self.mask_mean(q, q_mask)
        ans, a_mask = self.text_embs(ans)  # (bs, al, k)
        ans = self.mask_mean(ans, a_mask)
        user = self.user_emb(user)
        # with tf.variable_scope('PairWise'):
        # o = 0 + q * ans + q * user + ans * user
        # o = q * user + ans * user + q * ans
        o = self.pair_mul_add([q, ans, user])
        return o

    def bias(self, ans, user):
        o = 0
        if self.args.ans_bias:
            a, a_mask = self.text_embs(ans, init='random', train='train', name='ForTopK')
            # a = self.text_rep(a, a_mask)
            a = self.mask_top_k_mean(a, a_mask, 5)
            a = tf.layers.dense(a, 1, name='AnsBias')
            a = tf.squeeze(a, [1])
            o = o + a
        if self.args.user_bias:
            u = self.user_emb(user, name='UserBiasEmb')
            u = tf.layers.dense(u, 1, name='UserBiasDense')
            u = tf.squeeze(u, [1])
            o = o + u
        return o

    def make_inputs(self):
        if self.args.rank:
            self.inp_q = tf.placeholder(tf.int32, (None, q_maxlen), name='question')

            self.inp_ap = tf.placeholder(tf.int32, (None, a_maxlen), name='pos_ans')
            self.inp_up = tf.placeholder(tf.int32, (None,), name='pos_user')

            self.inp_an = tf.placeholder(tf.int32, (None, a_maxlen), name='neg_ans')
            self.inp_un = tf.placeholder(tf.int32, (None,), name='neg_user')

            if self.args.add_features:
                self.inp_fp = tf.placeholder(tf.float32, (None, nb_features), name='pos_feature')
                self.inp_fn = tf.placeholder(tf.float32, (None, nb_features), name='neg_feature')
                self.train_inputs = [self.inp_q, self.inp_ap, self.inp_up, self.inp_fp, self.inp_an,
                                     self.inp_un, self.inp_fn]
                self.pred_inputs = [self.inp_q, self.inp_ap, self.inp_up, self.inp_fp]
            else:
                self.train_inputs = [self.inp_q, self.inp_ap, self.inp_up, self.inp_an, self.inp_un]
                self.pred_inputs = [self.inp_q, self.inp_ap, self.inp_up]
        else:
            self.inp_q = tf.placeholder(tf.int32, (None, q_maxlen), name='question')

            self.inp_a = tf.placeholder(tf.int32, (None, a_maxlen), name='ans')
            self.inp_u = tf.placeholder(tf.int32, (None,), name='user')
            self.out_y = tf.placeholder(tf.float32, (None,), name='output_y')
            if self.args.add_features:
                self.inp_f = tf.placeholder(tf.float32, (None, nb_features), name='feature')
                self.train_inputs = [self.inp_q, self.inp_a, self.inp_u, self.inp_f, self.out_y]
                self.pred_inputs = [self.inp_q, self.inp_a, self.inp_u, self.inp_f]
            else:
                self.train_inputs = [self.inp_q, self.inp_a, self.inp_u, self.out_y]
                self.pred_inputs = [self.inp_q, self.inp_a, self.inp_u]

    def make_model(self):
        tf.set_random_seed(123456)

        self.make_inputs()
        self.emb_ws = set()
        self._make_model()  # get loss
        opt = tf.train.AdamOptimizer
        if self.args.opt == 'adam':
            opt = tf.train.AdamOptimizer
        elif self.args.opt == 'adadelta':
            opt = tf.train.AdadeltaOptimizer
        elif self.args.opt == 'adagrad':
            opt = tf.train.AdagradOptimizer

        optimizer = opt(learning_rate=self.args.lr)
        if self.emb_ws:
            if self.args.lr2 is None:
                self.minimize = optimizer.minimize(self.loss,
                                                   var_list=tf.trainable_variables() + list(
                                                       self.emb_ws))
                self.minimize2 = None
            else:
                self.minimize = optimizer.minimize(self.loss)
                opt2 = opt(learning_rate=self.args.lr2)
                self.minimize2 = opt2.minimize(self.loss, var_list=list(self.emb_ws))
        else:
            self.minimize = optimizer.minimize(self.loss)
            self.minimize2 = None

        self.sess = utils.get_session(gpus=self.gpu)

        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)
        '''
        if self.run_summary:
            ws = tf.global_variables('train/.*/emb_w[^/]') + tf.trainable_variables()
            for g, w in optimizer.compute_gradients(self.loss, ws):
                tf.summary.histogram(w.name, w)
                if g is not None:
                    tf.summary.histogram('grad/' + w.name, g)
        '''

    def _make_model(self):
        if self.args.rank:
            if self.args.add_features:
                with tf.variable_scope("train", reuse=tf.AUTO_REUSE):
                    self.question_rep = self.get_question_rep(self.inp_q)
                    p_score = self.TopLayer(self.rep(self.inp_ap, self.inp_up, self.inp_fp))
                with tf.variable_scope("train", reuse=tf.AUTO_REUSE):
                    n_score = self.TopLayer(self.rep(self.inp_an, self.inp_un, self.inp_fn))
            else:
                self.phase = 'pos'
                with tf.variable_scope("train", reuse=tf.AUTO_REUSE):
                    self.question_rep = self.get_question_rep(self.inp_q)
                    p_score = self.TopLayer(self.rep(self.inp_ap, self.inp_up))
                    with tf.variable_scope("bias"):
                        p_bias = self.bias(self.inp_ap, self.inp_up)
                    if p_bias is not None:
                        p_score = p_score + p_bias
                self.phase = 'neg'
                with tf.variable_scope("train", reuse=tf.AUTO_REUSE):
                    n_score = self.TopLayer(self.rep(self.inp_an, self.inp_un))
                    with tf.variable_scope("bias"):
                        n_bias = self.bias(self.inp_an, self.inp_un)
                    if n_bias is not None:
                        n_score = n_score + n_bias

            diff_score = p_score - n_score
            self.loss = tf.reduce_mean(
                tf.maximum(self.args.mc - diff_score, tf.zeros_like(diff_score)))
            l2 = self.add_l2()
            if l2 is not None:
                self.loss += l2
            # tf.summary.scalar("loss", self.loss)
            self.outputs = p_score
            # tf.summary.scalar("diff_score", tf.reduce_mean(diff_score))
        else:
            with tf.variable_scope("train", reuse=tf.AUTO_REUSE):
                self.question_rep = self.get_question_rep(self.inp_q)
                # with tf.variable_scope("train", reuse=tf.AUTO_REUSE):
                score = self.TopLayer(self.rep(self.inp_a, self.inp_u))
            self.loss = tf.losses.mean_squared_error(self.out_y, score)
            # self.loss = tf.reduce_mean((self.out_y - score) ** 2)
            l2 = self.add_l2()
            if l2 is not None:
                self.loss += l2
            self.outputs = score

    def prt_info(self):
        log('nb data: {}'.format(self.data.nb_train_data))
        self.nb_batch = int(np.ceil(self.data.nb_train_data / self.args.batch_size))
        log('nb batch: {}'.format(self.nb_batch))
        self.args.batch_steps = min(self.args.batch_steps, self.nb_batch)
        log('all data need {:.0f} epochs'.format(np.ceil(self.nb_batch / self.args.batch_steps)))
        log('early_stop: {}'.format(self.early_stop))

    def pre_fit(self):
        # self.good_log = False
        self.data.set_args(
            pair_mode=self.args.pair_mode,
            rank=self.args.rank,
            add_features=self.args.add_features,
            wordVec_size=self.args.dim_k if self.args.init_word == 'word2vec' else 0,
        )
        # print('change size:', self.args.dim_k); input()

        if self.args.rank:
            data_generator = self.data.gen_train_pair(batch_size=self.args.batch_size)
        else:
            data_generator = self.data.gen_data(batch_size=self.args.batch_size)
        queue_size = 10
        train_data_queue = Queue(queue_size)
        self.preload_process = Process(target=iter_and_put, args=(data_generator, train_data_queue))
        self.preload_process.daemon = True
        self.preload_process.start()

        def load_from_queue():
            while True:
                yield train_data_queue.get()

        self.data_generator = load_from_queue()

    def after_fit(self):
        ret, t = self.data.evaluate('test', self.predict)
        tf.reset_default_graph()
        self.sess.close()
        self.preload_process.terminate()
        return ret

    def fit(self, args, args_i, args_n):
        self.args_i = args_i
        self.args_n = args_n
        self.args = utils.Object(**args)

        self.pre_fit()
        self.prt_info()
        self.make_model()
        unique_fn = '{}-{}'.format(logger.unique_fn, self.args_i)

        tensorboard_dir = 'tensorboard/{}'.format(unique_fn)
        self.tb_dirs.append(tensorboard_dir)
        train_writer = tf.summary.FileWriter(tensorboard_dir, self.sess.graph)

        saver = tf.train.Saver()

        summ_loss = tf.Summary()
        summ_loss_v = summ_loss.value.add()
        summ_loss_v.tag = 'loss_per_batch'

        summaries = tf.summary.merge_all()

        batch_cnt = 0
        best_vali = None
        brk = 0
        # ret, _ = self.data.evaluate('vali', self.predict)
        # print(ret)
        has_ckpt = False
        try:
            for epochs in range(self.max_epochs):
                loss = []
                progress_bar = utils.ProgressBar(self.args.batch_steps, msg='training')
                for step in range(self.args.batch_steps):
                    batch = next(self.data_generator)
                    data = dict(zip(self.train_inputs, batch))
                    # print(len(self.train_inputs), len(batch)); input()
                    # print(data); input()
                    if step == 0 and summaries is not None:
                        summ = self.sess.run(summaries, data)
                        train_writer.add_summary(summ, global_step=batch_cnt)

                    if self.minimize2 is None:
                        _, _loss = self.sess.run([self.minimize, self.loss], data)
                    else:
                        _, _, _loss = self.sess.run([self.minimize, self.minimize2, self.loss],
                                                    data)

                    batch_cnt += 1
                    loss.append(_loss)
                    summ_loss_v.simple_value = _loss
                    train_writer.add_summary(summ_loss, global_step=batch_cnt)
                    progress_bar.make_a_step()
                    self.good_log = True
                train_time = progress_bar.stop()

                vali, vali_time = self.data.evaluate('vali', self.predict)
                if vali.is_better_than(best_vali):
                    brk = 0
                    best_vali = vali
                    saver.save(self.sess, 'save/{}_model.ckpt'.format(unique_fn))
                    has_ckpt = True
                else:
                    brk += 1

                if self.run_test:
                    test, test_time = self.data.evaluate('test', self.predict)
                    vali = '{} {}'.format(vali, test)
                    vali_time += test_time

                msg = '#{}/{}, loss: {:.5f}, vali: {}, brk: {}, time: {:.1f}s {:.1f}s'.format(
                    epochs + 1, self.max_epochs, np.mean(loss), vali, brk, train_time, vali_time)
                log(msg, i=-1, red=(brk == 0))
                if self.early_stop > 0 and brk >= self.early_stop:
                    break

        except KeyboardInterrupt:
            utils.timer.stop()
            log('KeyboardInterrupt')
        except Exception as e:
            utils.timer.stop()
            log('Exception: {}'.format(e), red=True)
        if has_ckpt:
            saver.restore(self.sess, 'save/{}_model.ckpt'.format(unique_fn))
        return self.after_fit()

    def predict(self, data):
        data = dict(zip(self.pred_inputs, data))
        y = self.sess.run(self.outputs, data)
        # y = np.arange(len(y))
        # np.random.shuffle(y)
        return y


class PairWise(BasicPair):
    tc = dict(BasicPair.tc)
    # tc['top_k'] = [5]
    # tc['text_mode'] = ['mask_max']
    tc['text_mode'] = ['mask_mean', 'mean']

    def get_question_rep(self, q):
        q, q_mask = self.text_embs(q)
        return self.text_rep(q, q_mask)

    def rep(self, a, u):
        q = self.question_rep
        a, a_mask = self.text_embs(a)
        a = self.text_rep(a, a_mask)
        u = self.user_emb(u)
        o = self.pair_mul_add([q, a, u])
        return o


from .Baselines import *
from .sklean_models import *
from .RepMatch import *
from .WordMatch import *
from .WoQ import *
from .Gate import *
from .Att import *
from .TopK import *
from .Concat import *


def main():
    print('hello world, models_prev.py')


if __name__ == '__main__':
    main()
