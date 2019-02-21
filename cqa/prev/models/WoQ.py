#########################################################################
# File Name: WoQ.py
# Author: wwang
# mail: 750636248@qq.com
# Created Time: 2018年08月21日 星期二 14时24分18秒
#########################################################################

import numpy as np, sys, math, os
import tensorflow as tf
from . import utils
from . import dataset
import time
from . import log
from . import q_maxlen, a_maxlen
from . import BasicPair

class Ans(BasicPair):
    tc = dict(BasicPair.tc)
    #tc['top_k'] = [5, 1, 3, 7, 20]
    #tc['text_mode'] = ['top_k_sum', 'top_k_mean', 'sum', 'mean', 'mask_sum', 'mask_mean']
    #tc['lr'] = [1e-3, 3e-4, 1e-4, 3e-5]
    #tc['lr'] = [3e-4]
    #tc['lr'] = [1e-5]
    #tc['text_mode'] = ['mask_mean']
    #tc['text_mode'] = ['mean-max']
    tc['top_k'] = [5]
    tc['text_mode'] = ['mask_top_k_mean']
    tc['add_user'] = [True, False]
    #tc['add_features'] = [False]

    def get_question_rep(self, q):
        pass

    def rep(self, ans, user):
        return tf.zeros_like(user, dtype = 'float32')

    def bias(self, ans, user):
        a, a_mask = self.text_embs(ans, init = 'random', train = 'train', name = 'ForTopK')
        a = self.text_rep(a, a_mask)
        a = tf.layers.dense(a, 1, name = 'AnsBias')
        if self.args.add_user:
            u = self.user_emb(user)
            u = tf.layers.dense(u, 1, name = 'UserBias')
            a = a + u
        return tf.squeeze(a, [1])


class Features(BasicPair):
    tc = dict(BasicPair.tc)
    tc['add_features'] = [True]
    tc['fc'] = [[]]
    tc['lr'] = [1e-6]

    #def get_question_rep(self, q):
        #pass

    def rep(self, a, user, f):
        return self.features_rep(f)

class AnsSum(Ans):
    tc = dict(Ans.tc)
    tc['top_k'] = [5]
    #tc['lr'] = [1e-3, 3e-4, 1e-4, 3e-5]
    tc['lr'] = [3e-4]
    tc['text_mode'] = ['mask_top_k_sum']
    tc['add_user'] = [False]

    def rep(self, a, user):
        a = super().rep(a, user) #(bs, k)
        return tf.reduce_sum(a, -1) #(bs, )


class User(BasicPair):
    tc = dict(BasicPair.tc)
    #tc['l2_top'] = [0, 1e-2, 1e-4, 1e-6]
    def rep(self, a, user):
        user = self.user_emb(user)
        return user

def main():
    print('hello world, WoQ.py')

if __name__ == '__main__':
    main()

