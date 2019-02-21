import os
import time
from multiprocessing import Process, Queue

import numpy as np

from cqa.simple import dataset
from cqa.simple.models.MyModel import *

q_maxlen = 20
a_maxlen = 200


def iter_and_put(it, queue):
    print('sub precess {} start'.format(os.getpid()))
    for i in it:
        queue.put(i)


def get_session(memory_rate=1.0, gpus='0'):
    # os.environ['CUDA_VISIBLE_DEVICE'] = '0,1,2,3'
    # os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    gpu_options = tf.GPUOptions(
        per_process_gpu_memory_fraction=memory_rate,
        visible_device_list=gpus, allow_growth=True
    )
    config = tf.ConfigProto(gpu_options=gpu_options)
    session = tf.Session(config=config)
    return session


# noinspection PyAttributeOutsideInit
class BasicPair:
    Data = dataset.tf_data

    def __init__(self,
                 dataset,
                 run_name=None,
                 dim_k=64,
                 batch_size=32,
                 batch_steps=10000,
                 max_epochs=100,
                 early_stop=5,
                 gpu='3',
                 lr=1e-3,
                 fc=[512, 256],
                 ):
        if run_name is None:
            self.run_name = time.strftime("%Y-%m-%d %H.%M.%S", time.localtime())
        else:
            self.run_name = run_name

        self.dataset = dataset
        self.dim_k = dim_k
        self.fc = fc
        self.lr = lr

        self.gpu = str(gpu)
        self.batch_size = batch_size
        self.batch_steps = batch_steps
        self.max_epochs = max_epochs
        self.early_stop = early_stop

        if dataset == 'test':
            self.batch_steps = 10

        self.text_mode = 'mean-max'
        self.data = self.Data(self.dataset)

    def user_emb(self, u, name='UserEmb'):
        with tf.variable_scope(name):
            emb_w = tf.get_variable(
                name='emb_w', shape=(self.data.nb_users, self.dim_k),
                initializer=tf.random_uniform_initializer(minval=-0.05, maxval=0.05),
            )
            o = tf.nn.embedding_lookup(emb_w, u)
        return o

    def text_emb(self, t, dim_k=None, init='random', train='train', name='TextEmb'):
        if dim_k is None:
            dim_k = self.dim_k
        if init == 'word2vec':
            dim_k = len(self.data.wordVec[0])
        if init == 'random':
            initializer = tf.random_uniform_initializer(minval=-0.05, maxval=0.05)
        else:
            initializer = tf.constant_initializer(self.data.wordVec)
        if train == 'train':
            trainable = True
        else:
            trainable = False

        with tf.variable_scope(name):
            pad = tf.zeros(shape=(1, dim_k))
            emb_w = tf.get_variable(
                name='emb_w', shape=(self.data.nb_words, dim_k),
                initializer=initializer, trainable=trainable,
            )
            if train == 'dense':
                emb_w = tf.layers.dense(emb_w, dim_k, name='MapWordVec')
            emb_w = tf.concat([pad, emb_w], 0)
            o = tf.nn.embedding_lookup(emb_w, t)
            mask = tf.cast(tf.not_equal(t, 0), tf.float32)
            mask = tf.expand_dims(mask, -1)
        return o, mask

    ''' *********************************** funcs begin ***********************************'''

    def mask_max(self, t, mask):
        # (bs, tl, k), (bs, tl)
        with tf.variable_scope('mask_max'):
            t = t * mask - (1 - mask) * 1e30
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
            t = t * mask - (1 - mask) * 1e30
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
            t = t * mask - (1 - mask) * 1e30
            t = self.top_k(t, k)  # (bs, k, dim_k)
            m = self.top_k(mask, k)  # (bs, k, 1)
            s = self.mask_mean(t, m)  # (bs, dim_k)
        return s

    def features_rep(self, f, dim_k=None):
        if dim_k is None:
            dim_k = self.dim_k
        with tf.variable_scope('FeaturesRep'):
            f = tf.layers.batch_normalization(f) * 0.01
            f = tf.layers.dense(f, dim_k, name='featuresT')  # activation = tf.nn.relu,
            return f

    def text_rep(self, t, mask):
        if self.text_mode == 'top_k_sum':
            t = self.top_k_sum(t, 5)
        elif self.text_mode == 'mask_top_k_sum':
            t = self.mask_top_k_sum(t, mask, 5)
        elif self.text_mode == 'top_k_mean':
            t = self.top_k_mean(t, 5)
        elif self.text_mode == 'mask_top_k_mean':
            t = self.mask_top_k_mean(t, mask, 5)
        elif self.text_mode == 'max':
            t = tf.reduce_max(t, -2)
        elif self.text_mode == 'sum':
            t = tf.reduce_sum(t, -2)
        elif self.text_mode == 'mean':
            t = tf.reduce_mean(t, -2)
        elif self.text_mode == 'mask_max':
            t = self.mask_max(t, mask)
        elif self.text_mode == 'mask_sum':
            t = self.mask_sum(t, mask)
        elif self.text_mode == 'mask_mean':
            t = self.mask_mean(t, mask)
        elif self.text_mode == 'mean-max':
            t = tf.expand_dims(t * mask, -1)
            t = tf.layers.average_pooling2d(t, (5, 1), (1, 1))
            t = tf.squeeze(t, [-1])
            t = tf.reduce_max(t, -1)
        elif self.text_mode == 'max-mean':
            t = tf.expand_dims(t * mask, -1)
            t = tf.layers.max_pooling2d(t, (5, 1), (1, 1))
            t = tf.squeeze(t, [-1])
            t = tf.reduce_mean(t, -1)
        else:
            print('Error')
            input()
        return t

    # def lstm(self, t):
    #     with tf.variable_scope('LSTM'):
    #         cell = tf.nn.rnn_cell.LSTMCell(
    #             self.dim_k,
    #             initializer=self.get_initializer(),
    #             # reuse = True,
    #             name='lstm_cell',
    #         )
    #         seq, final_state = tf.nn.dynamic_rnn(cell, t, dtype=tf.float32)
    #     return seq

    # def dot(self, a, b, axis=-1):
    #     return tf.reduce_sum(a * b, axis)

    # def cos(self, a, b, axis=-1):
    #     ab = self.dot(a, b, axis)
    #     aa = tf.sqrt(self.dot(a, a, axis))
    #     bb = tf.sqrt(self.dot(b, b, axis))
    #     return ab / (aa * bb + 1e-7)

    ''' ****************************** funcs over ******************************'''

    def top_layers(self, o, fc=None, name='TopLayer'):
        if fc is None:
            fc = self.fc
        shape = o.shape
        if len(shape) == 1:
            return o
        if len(shape) != 2:
            log('ndim mast be 1 or 2, but got shape: {}'.format(shape), red=True)
        if shape[-1] == 1:
            return tf.reshape(o, (-1,))
        with tf.variable_scope(name):
            for i, n in enumerate(fc):
                # o = tf.layers.batch_normalization(o)
                o = tf.layers.dense(o, n, activation=tf.nn.relu, name='dense_{}'.format(i))
            # o = tf.layers.batch_normalization(o)
            o = tf.layers.dense(o, 1, name='dense_{}'.format(len(fc)))  # use_bias = False,
            o = tf.reshape(o, (-1,))
        return o

    def rep(self, que, ans, user, features):
        ans, ans_mask = self.text_emb(ans)  # (bs, al, k)
        ans = self.text_rep(ans, ans_mask)
        user = self.user_emb(user)
        o = tf.concat([ans, user], axis=-1)
        return o

    def make_inputs(self):
        self.inp_q0 = tf.placeholder(tf.int32, (None, q_maxlen), name='pos_question')
        self.inp_a0 = tf.placeholder(tf.int32, (None, a_maxlen), name='pos_answer')
        self.inp_u0 = tf.placeholder(tf.int32, (None,), name='pos_user')
        self.inp_f0 = tf.placeholder(tf.float32, (None, self.Data.nb_features), name='pos_feature')

        self.inp_q1 = tf.placeholder(tf.int32, (None, q_maxlen), name='neg_question')
        self.inp_a1 = tf.placeholder(tf.int32, (None, a_maxlen), name='neg_ans')
        self.inp_u1 = tf.placeholder(tf.int32, (None,), name='neg_user')
        self.inp_f1 = tf.placeholder(tf.float32, (None, self.Data.nb_features), name='neg_feature')

        self.train_inputs = [self.inp_q0, self.inp_a0, self.inp_u0, self.inp_f0,
                             self.inp_q1, self.inp_a1, self.inp_u1, self.inp_f1]
        self.pred_inputs = [self.inp_q0, self.inp_a0, self.inp_u0, self.inp_f0]

    def make_model(self):
        with tf.variable_scope("train", reuse=tf.AUTO_REUSE):
            score0 = self.top_layers(self.rep(self.inp_q0, self.inp_a0, self.inp_u0, self.inp_f0))
            score1 = self.top_layers(self.rep(self.inp_q1, self.inp_a1, self.inp_u1, self.inp_f1))
        diff_score = score0 - score1
        self.loss = tf.reduce_mean(tf.maximum(1 - diff_score, tf.zeros_like(diff_score)))
        self.outputs = score0

    def compile(self):
        tf.set_random_seed(123456)
        self.make_inputs()
        self.make_model()  # get loss
        optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
        self.minimize = optimizer.minimize(self.loss)
        self.sess = get_session(gpus=self.gpu)
        self.sess.run(tf.global_variables_initializer())

        data_generator = self.data.gen_train_pair(batch_size=self.batch_size)
        queue_size = 10
        train_data_queue = Queue(queue_size)
        self.preload_process = Process(target=iter_and_put, args=(data_generator, train_data_queue))
        self.preload_process.daemon = True
        self.preload_process.start()

        def load_from_queue():
            while True:
                yield train_data_queue.get()

        self.data_generator = load_from_queue()

    def fit(self):
        brk = 0
        best_valid = None
        has_ckpt = False
        saver = tf.train.Saver()
        is_pbar_start = False
        try:
            for epochs in range(self.max_epochs):
                loss = []
                pbar = tqdm(total=self.batch_steps, ncols=100, leave=False, desc='Training')
                is_pbar_start = True
                for step in range(self.batch_steps):
                    batch = next(self.data_generator)
                    data = dict(zip(self.train_inputs, batch))
                    _, _loss = self.sess.run([self.minimize, self.loss], data)
                    loss.append(_loss)
                    pbar.update(1)
                pbar.close()
                is_pbar_start = False

                vali = self.evaluate('vali')
                if vali.is_better_than(best_valid):
                    brk = 0
                    best_valid = vali
                    saver.save(self.sess, 'save/{}_model.ckpt'.format(self.run_name))
                    has_ckpt = True
                else:
                    brk += 1

                # if self.run_test:
                # test = self.data.evaluate('test', self.predict)
                # vali = '{} {}'.format(vali, test)

                msg = '#{}/{}, loss: {:.5f}, vali: {:.5f}, brk: {}'.format(
                    epochs + 1, self.max_epochs, np.mean(loss), vali, brk)
                print(msg)
                if self.early_stop <= brk:
                    break

        except KeyboardInterrupt:
            if is_pbar_start:
                pbar.close()
                is_pbar_start = False
            print('KeyboardInterrupt')
        except Exception as e:
            if is_pbar_start:
                pbar.close()
                is_pbar_start = False
            print('Exception: {}'.format(e))
        if has_ckpt:
            saver.restore(self.sess, 'save/{}_model.ckpt'.format(self.run_name))

    def terminate(self):
        tf.reset_default_graph()
        self.sess.close()
        self.preload_process.terminate()

    def predict(self, data):
        data = dict(zip(self.pred_inputs, data))
        return self.sess.run(self.outputs, data)

    def evaluate(self, dataname='test'):
        return self.data.evaluate(dataname, self.predict)


def main():
    print('hello world, models_prev.py')


if __name__ == '__main__':
    main()
