#########################################################################
# File Name: Gate.py
# Author: wwang
# mail: 750636248@qq.com
# Created Time: 2018年08月21日 星期二 14时20分21秒
#########################################################################

import numpy as np, sys, math, os
import tensorflow as tf
from . import utils
from . import dataset
import time
from . import log
from . import q_maxlen, a_maxlen
from . import BasicPair

class QueGate(BasicPair):
    tc = dict(BasicPair.tc)
    tc['top_k'] = [5]
    #tc['l2_w'] = [0.01, 1e-4]
    #tc['l2_w'] = [0, 0.001, 1, 100]
    #tc['dim_k'] = [8, 16, 32]
    tc['add_user'] = [False]
    def rep(self, a, user):
        q, q_mask = self.question_rep
        qw = tf.layers.dense(
                q,
                self.args.dim_k,
                kernel_regularizer = lambda w: self.args.l2_w * tf.nn.l2_loss(w),
                kernel_initializer = self.get_initializer(),
                )
        tf.summary.histogram("gate/qw", qw)
        #a, a_mask = self.text_embs(a, name = 'AnsEmb') # (bs, al, k)
        a, a_mask = self.text_embs(a) # (bs, al, k)
        m = tf.matmul(a, tf.transpose(qw, [0, 2, 1]))
        tf.summary.histogram("gate/m", m)
        gate = tf.expand_dims(tf.sigmoid(tf.reduce_max(m, -1)), -1)
        tf.summary.histogram("gate/v", gate)
        a = a * gate
        a = self.top_k(a, self.args.top_k)
        a = tf.reduce_sum(a, -2)
        if self.args.add_user:
            user = self.user_emb(user)
            return tf.concat([a, user], -1)
        return a

class QueGate2(BasicPair):
    tc = dict(BasicPair.tc)
    tc['top_k'] = [5]
    #tc['l2_w'] = [0.01, 1e-4]
    tc['l2_w'] = [0, 0.001, 1, 100]
    #tc['dim_k'] = [8, 16, 32]
    tc['add_user'] = [False]
    def rep(self, a, user):
        q, q_mask = self.question_rep
        q = self.top_k(q, self.args.top_k)
        q = tf.expand_dims(tf.reduce_sum(q, -2), -1) #(bs, k, 1)
        tf.summary.histogram("gate/q", q)
        #a, a_mask = self.text_embs(a, name = 'AnsEmb') # (bs, al, k)
        a, a_mask = self.text_embs(a) # (bs, al, k)
        m = tf.matmul(a, q) #(bs, al, 1)
        tf.summary.histogram("gate/m", m)
        a = a * tf.sigmoid(m)
        a = self.top_k(a, self.args.top_k)
        a = tf.reduce_sum(a, -2)
        if self.args.add_user:
            user = self.user_emb(user)
            return tf.concat([a, user], -1)
        return a


def main():
    print('hello world, Gate.py')

if __name__ == '__main__':
    main()

