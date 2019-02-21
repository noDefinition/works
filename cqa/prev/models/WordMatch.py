#########################################################################
# File Name: WordMatch.py
# Author: wwang
# mail: 750636248@qq.com
# Created Time: 2018年08月21日 星期二 14时19分09秒
#########################################################################

import numpy as np, sys, math, os
import tensorflow as tf
from . import utils
from . import dataset
import time
from . import log
from . import q_maxlen, a_maxlen
from . import BasicPair

class MaxSimQA(BasicPair):
    tc = dict(BasicPair.tc)
    def match(self, q, a, name = 'match'):
        with tf.variable_scope(name):
            q, q_mask = q
            a, a_mask = a

            q = tf.expand_dims(q, 2) #(bs, ql, 1, k)
            a = tf.expand_dims(a, 1) #(bs, 1, al, k)

            m = self.cos(q, a) #(bs, ql, al)
            sim_a = tf.reduce_max(m, 1) #(bs, al)
            sim_q = tf.reduce_max(m, 2) #(bs, ql)
            #sim = tf.reduce_sum(sim_a, -1) + tf.reduce_sum(sim_q, -1)
            #sim = tf.expand_dims(sim, -1)

            sim = self.mask_mean(tf.expand_dims(sim_a, -1), a_mask)
            sim += self.mask_mean(tf.expand_dims(sim_q, -1), q_mask)
            #print(sim)
        return sim

    def rep(self, a, user):
        a, a_mask = self.text_embs(a)
        q, q_mask = self.question_rep

        m = self.match((q, q_mask), (a, a_mask))

        return m


class SimScore(MaxSimQA):
    tc = dict(MaxSimQA.tc)
    def match(self, q, a, u, name = 'match'):
        with tf.variable_scope(name):
            q, q_mask = q
            a, a_mask = a

            u = tf.expand_dims(u, 1) #(bs, 1, k)
            ug = tf.layers.dense(
                    u,
                    1,
                    activation = tf.nn.sigmoid,
                    kernel_initializer = self.get_initializer(),
                    name = 'user_gate',
                    ) #(bs, 1, 1)
            a = a * (1 - ug) + u * ug

            q = tf.expand_dims(q, 2) #(bs, ql, 1, k)
            a = tf.expand_dims(a, 1) #(bs, 1, al, k)

            m = self.cos(q, a) #(bs, ql, al)
            sim_a = tf.reduce_max(m, 1) #(bs, al)
            sim_q = tf.reduce_max(m, 2) #(bs, ql)
            #sim = tf.reduce_sum(sim_a, -1) + tf.reduce_sum(sim_q, -1)
            #sim = tf.expand_dims(sim, -1)

            sim_a = self.mask_mean(tf.expand_dims(sim_a, -1), a_mask)
            sim_q = self.mask_mean(tf.expand_dims(sim_q, -1), q_mask)
            #print(sim)
        return sim_a, sim_q

    def rep(self, a, user):
        a, a_mask = self.text_embs(a)
        q, q_mask = self.question_rep
        u = self.user_emb(user)

        m = self.match((q, q_mask), (a, a_mask), u)

        return tf.concat(m, -1)

class PairScore(BasicPair):
    tc = dict(BasicPair.tc)
    #tc['top_k'] = [5]
    #tc['text_mode'] = ['mask_top_k_sum']
    #tc['add_user'] = [True]
    #tc['add_ans'] = [True]
    #tc['lr'] = [1e-5]

    def Cnn(self, m, name = 'CNN'):
        with tf.variable_scope(name):
            activation = tf.nn.relu

            k_size = (5, 5)
            p_size = (2, 5)
            m = tf.layers.conv2d(
                    m,
                    filters = self.args.dim_k // 2,
                    kernel_size = k_size,
                    activation = activation,
                    kernel_initializer = self.get_initializer(),
                    )
            m = tf.layers.max_pooling2d(m, p_size, p_size) #(10, 40)
            m = tf.layers.BatchNormalization()(m)

            k_size = (5, 5)
            p_size = (2, 5)
            m = tf.layers.conv2d(
                    m,
                    filters = self.args.dim_k // 4,
                    kernel_size = k_size,
                    activation = activation,
                    kernel_initializer = self.get_initializer(),
                    )
            m = tf.layers.max_pooling2d(m, p_size, p_size) #(5, 8)
            m = tf.layers.BatchNormalization()(m)

        return m

    def match(self, q, a, u, name = 'match'):
        # (bs, ql, k), (bs, al, k), (bs, k)
        with tf.variable_scope(name):
            qu = q * tf.expand_dims(u, 1) #(bs, ql, k)
            au = a * tf.expand_dims(u, 1) #(bs, al, k)
            qa = tf.expand_dims(q, 2) * tf.expand_dims(a, 1) #(bs, ql, al, k)

            qu = tf.layers.BatchNormalization()(qu)
            au = tf.layers.BatchNormalization()(au)
            qa = tf.layers.BatchNormalization()(qa)

            qua = (qa + tf.expand_dims(qu, 2) + tf.expand_dims(au, 1)) / 3.0 #(bs, ql, al, k)
            m = self.Cnn(qua)

        return tf.layers.flatten(m)

    def rep(self, a, user):
        q, q_mask = self.question_rep
        a, a_mask = self.text_embs(a) # (bs, al, k)
        #a_rep = self.text_rep(a)
        u = self.user_emb(user)
        m = self.match(q, a, u)
        return m

        if self.args.add_ans:
            ab = self.text_rep(a, a_mask)
            m = tf.concat([m, ab], -1)

        if self.args.add_user:
            #ub = self.user_emb(user)
            ub = u
            m = tf.concat([m, ub], -1)
        return m

class ConcatMatch(BasicPair):
    tc = dict(BasicPair.tc)
    tc['use_mlp'] = [True]
    tc['top_k'] = [5]
    tc['dim_k'] = [16]

    def match(self, q, a, u, name = 'match'):
        # (bs, ql, k), (bs, al, k)
        with tf.variable_scope(name):
            q = tf.expand_dims(q, 2) #(bs, ql, 1, k)
            q = tf.concat([q for i in range(a_maxlen)], 2)
            a = tf.expand_dims(a, 1) #(bs, 1, al, k)
            a = tf.concat([a for i in range(q_maxlen)], 1)
            qa = tf.concat([q, a], -1) #(bs, ql, al, 2 * k)
            qa = tf.reshape(qa, (-1, q_maxlen * a_maxlen, 2 * self.args.dim_k))
            u = tf.expand_dims(u, 1) # (bs, 1, k)
            u = tf.concat([u for i in range(q_maxlen * a_maxlen)], 1)
            qau = tf.concat([qa, u], -1)
            o = self.MLP(qau, [self.args.dim_k * 2, self.args.dim_k])
            o = self.top_k(o, self.args.top_k) #(bs, 5, k)
            o = tf.reduce_sum(o, 1) #(bs, k)
        return o

    def rep(self, a, user):
        q, q_mask = self.question_rep
        a, a_mask = self.text_embs(a) # (bs, al, k)
        user = self.user_emb(user)
        m = self.match(q, a, user)
        a = self.top_k(a, self.args.top_k)
        a = tf.reduce_sum(a, -2)
        return tf.concat([m, a, user], -1)

class CMatch(BasicPair):
    tc = dict(BasicPair.tc)
    tc['top_k'] = [5]
    tc['mk'] = [10]
    tc['text_mode'] = ['mask_top_k_mean']
    #tc['dim_k'] = [8, 16, 32]

    def Cnn(self, m, name = 'CNN'):
        with tf.variable_scope(name):
            activation = None
            m = tf.layers.conv2d(
                    m,
                    filters = self.args.mk,
                    kernel_size = 5,
                    activation = activation,
                    kernel_initializer = self.get_initializer(),
                    )
            m = tf.layers.max_pooling2d(m, (2, 3), (2, 3))
            m = tf.layers.conv2d(
                    m,
                    filters = self.args.mk,
                    kernel_size = 5,
                    activation = activation,
                    kernel_initializer = self.get_initializer(),
                    )
            m = tf.layers.max_pooling2d(m, (2, 3), (2, 3))
        return m

    def match(self, q, a, u, name = 'match'):
        # (bs, ql, k), (bs, al, k)
        with tf.variable_scope(name):
            q = tf.layers.dense(
                    q,
                    self.args.mk,
                    use_bias = False,
                    name = 'dense_tq',
                    kernel_initializer = self.get_initializer(),
                    )
            a = tf.layers.dense(
                    a,
                    self.args.mk,
                    use_bias = False,
                    name = 'dense_ta',
                    kernel_initializer = self.get_initializer(),
                    )
            u = tf.layers.dense(
                    u,
                    self.args.mk,
                    name = 'dense_tu',
                    kernel_initializer = self.get_initializer(),
                    )

            q = tf.expand_dims(q, 2) #(bs, ql, 1, k)

            a = tf.expand_dims(a, 1) #(bs, 1, al, k)

            u = tf.expand_dims(u, 1) # (bs, 1, k)
            u = tf.expand_dims(u, 1) # (bs, 1, 1, k)

            qau = tf.nn.relu(q + a + u)
            o = self.Cnn(qau)
            o = tf.reshape(o, (-1, o.shape[1] * o.shape[2], o.shape[3]))

            o = self.MLP(o, [self.args.dim_k * 2, self.args.dim_k])
            #o = self.top_k(o, self.args.top_k) #(bs, 5, k)
            o = tf.reduce_mean(o, 1) #(bs, k)
        return o

    def rep(self, a, user):
        q, q_mask = self.question_rep
        a, a_mask = self.text_embs(a) # (bs, al, k)

        user = self.user_emb(user)
        m = self.match(q, a, user)

        a = self.text_rep(a, a_mask)

        return tf.concat([m, a, user], -1)

class MIX(BasicPair):
    tc = dict(BasicPair.tc)
    tc['top_k'] = [5]
    tc['text_mode'] = ['mask_top_k_mean']
    tc['add_user'] = [True]
    #tc['dim_k'] = [8, 16, 32]
    tc['l2_text_cnn'] = [0]
    tc['text_cnn_n'] = [3]
    #tc['att'] = [[], ['tw'], ['sp'], ['tw', 'sp']]
    tc['att'] = [['tw', 'sp']]
    #tc['att'] = [['tw', 'sp']]
    #tc['lr'] = [1e-4, 1e-5]
    #tc['emb_lr'] = [1e-3, 1e-4]
    #tc['opt'] = ['adadelta', 'adam', 'adagrad']

    def text_cnn(self, t, max_size, name = 'TextCnn'):
        activation = tf.nn.relu
        ret = []
        with tf.variable_scope(name):
            for s in range(max_size):
                s += 1
                o = tf.layers.conv1d(
                        t,
                        filters = self.args.dim_k,
                        kernel_size = s,
                        padding='same',
                        activation = activation,
                        kernel_initializer = self.get_initializer(),
                        name = 'text_cnn_size_{}'.format(s),
                        ) #(bs, tl, k)
                #o = tf.expand_dims(o, -1) #(bs, tl, k, 1)
                ret.append(o)
        #return tf.concat(ret, -1) #(bs, tl, k, max_size)
        return ret

    def att_tw(self, t, name = 'AttTw'):
        #initializer = tf.ones_initializer()
        #initializer = tf.Variable(np.array(self.data.word_idf))
        #initializer = tf.constant_initializer(np.array(self.data.word_idf))
        initializer = tf.constant_initializer(self.data.word_idf)
        #print(initializer)
        with tf.variable_scope(name):
            pad = tf.zeros((1, 1))
            emb_w = tf.get_variable(
                    name = 'att_tw',
                    shape = (self.data.nb_words, 1),
                    initializer = initializer,
                    )
            emb_w = tf.concat([pad, emb_w], 0)
            o = tf.nn.embedding_lookup(emb_w, t)
        return o #(bs, tl, 1)

    def att_sp(self, maxlen, name = 'AttSp'):
        initializer = tf.ones_initializer()
        with tf.variable_scope(name):
            emb_w = tf.get_variable(
                    name = 'att_sp',
                    shape = (1, maxlen, 1),
                    initializer = initializer,
                    )
        return emb_w #(1, tl, 1)

    def CNN(self, m, m_mask, name = 'MatchCnn'):
        activation = tf.nn.relu
        with tf.variable_scope(name):
            m = m * m_mask
            #for size in ((3, 5), (2, 4), (1, 3)):
            for size in ((3, 5), (2, 4)):
                m = tf.layers.conv2d(
                        m,
                        filters = self.args.dim_k,
                        kernel_size = size,
                        padding='same',
                        activation = activation,
                        kernel_initializer = self.get_initializer(),
                        )
                m = m * m_mask
                m = tf.layers.max_pooling2d(m, size, size)
                m_mask = tf.layers.max_pooling2d(m_mask, size, size)
            def reshape(w):
                return tf.reshape(w, (-1, w.shape[1] * w.shape[2], w.shape[3]))
            m = reshape(m)
            m_mask = reshape(m_mask)
        return m, m_mask

    def get_question_rep(self, inp_q):
        q, q_mask = self.text_embs(inp_q)
        q_cnn = self.text_cnn(q, self.args.text_cnn_n)
        q_tw = self.att_tw(inp_q)
        q_sp = self.att_sp(q_maxlen, name = 'que_sp')
        #q_sp = self.att_sp(q_maxlen)
        return q_cnn, q_mask, q_tw, q_sp

    def make_matrix(self, a, b, name):
        #(bs, al, k), (bs, bl, k)
        a = tf.expand_dims(a, 2)
        b = tf.expand_dims(b, 1)
        m = tf.multiply(a, b, name = name)
        if len(m.shape) < 4:
            m = tf.expand_dims(m, -1)
        return m

    def rep(self, inp_a, inp_u):
        q_cnn, q_mask, q_tw, q_sp = self.question_rep

        a, a_mask = self.text_embs(inp_a)
        ab = self.text_rep(a, a_mask)
        a_cnn = self.text_cnn(a, self.args.text_cnn_n)

        if 1:
            m = []
            q_cnn = [tf.expand_dims(q, 2) for q in q_cnn]
            a_cnn = [tf.expand_dims(a, 1) for a in a_cnn]
            for q in q_cnn:
                for a in a_cnn:
                    o = self.cos(q, a)
                    o = tf.expand_dims(o, -1)
                    m.append(o)
            m_match = tf.concat(m, -1)
        else:
            q_cnn = tf.expand_dims(q_cnn, 2) #(bs, ql, 1, k, ms)
            q_cnn = tf.expand_dims(q_cnn, -1) #(bs, ql, 1, k, ms, 1)
            q_cnn = tf.concat([q_cnn for i in range(self.args.text_cnn_n)], -1)
            a_cnn = tf.expand_dims(a_cnn, 1) #(bs, 1, al, k, ms)
            a_cnn = tf.expand_dims(a_cnn, -2) #(bs, 1, al, k, 1, ms)
            a_cnn = tf.concat([a_cnn for i in range(self.args.text_cnn_n)], -2)
            m = self.cos(q_cnn, a_cnn, axis = 3)
            m_match = tf.reshape(m, (-1, ) + tuple(m.shape[1: 3]) + (m.shape[3] * m.shape[4], )) #(bs, ql, al, 9)

        m_mask = self.make_matrix(q_mask, a_mask, name = 'mask_mul') #(bs, al, ql, 1)

        a_tw = self.att_tw(inp_a)
        a_sp = self.att_sp(a_maxlen, name = 'ans_sp')
        #a_sp = self.att_sp(a_maxlen)
        mm = []
        if 'tw' in self.args.att:
            m_att_tw = self.make_matrix(q_tw, a_tw, name = 'att_tw_mul')
            mm.append(m_att_tw)
        if 'sp' in self.args.att:
            m_att_sp = self.make_matrix(q_sp, a_sp, name = 'att_sp_mul')
            mm.append(m_att_sp)
        if len(mm) > 1:
            m = tf.concat([m_match * m for m in mm], -1) #(bs, al, ql, 18)
        elif len(mm) == 1:
            m = m_match * mm[0]
        else:
            m = m_match
        m, m_mask = self.CNN(m, m_mask)

        u = self.user_emb(inp_u)
        o = self.mask_mean(m, m_mask)
        if self.args.add_user:
            return tf.concat([o, ab, u], -1)
        return tf.concat([o, u], -1)


def main():
    print('hello world, WordMatch.py')

if __name__ == '__main__':
    main()

