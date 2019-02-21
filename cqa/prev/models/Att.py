import numpy as np, sys, math, os
import tensorflow as tf
from . import utils
from . import dataset
import time
from . import log
from . import q_maxlen, a_maxlen
from . import BasicPair

class AttLayers:
    tc = {}
    #tc['att_method'] = ['']
    #tc['att_mode'] = ['dot']

    def auto_att(self, text, mask, user):
        return eval('self.{}'.format(self.args.att_method))(text, mask, user)

    def output(self, text, mask, att_w):
        att_w = att_w * mask
        att_w = att_w - (1.0 - mask) * 1e30

        att_w = tf.nn.softmax(att_w, 1) #(bs, tl, k)
        text = tf.reduce_sum(text * att_w, 1) #(bs, k)

        return text

    # weight = ai * qj
    def AQ_dot_att(self, q, q_mask, a, a_mask, name = 'AQAtt'):
        #(bs, ql, k), (bs, ql, 1), (bs, al, k), (bs, al, 1)
        with tf.variable_scope(name):
            att_w = self.dot(tf.expand_dims(q, 2), tf.expand_dims(a, 1)) #(bs, ql, al)
            att_w = att_w * q_mask
            att_w = att_w - (1.0 - q_mask) * 1e30
            att_w = tf.nn.softmax(att_w, -2) #(bs, ql, al)
            att_a = tf.matmul(tf.transpose(att_w, [0, 2, 1]), q) #(bs, al, k)
        return att_a

    # weight = word @ user
    def dot_att(self, text, mask, user, name = 'DotAtt'):
        #(bs, tl, k), (bs, tl, 1), (bs, k)
        with tf.variable_scope(name):
            user = tf.expand_dims(user, -1) #(bs, k, 1)
            att_w = tf.matmul(text, user) #(bs, tl, 1)
            return self.output(text, mask, att_w)

    def MLP_att(self, text, mask, user = None, name = 'Att'):
        #(bs, tl, k), (bs, k)
        with tf.variable_scope(name):
            if user is not None:
                user = tf.expand_dims(user, 1) #(bs, 1, k)
                uu = tf.tile(user, [1, text.shape[1], 1])
                t = tf.concat([text, uu], -1) # (bs, tl, 2 * k)
            else:
                t = text
            att_w = tf.layers.dense(t, self.args.dim_k, activation = tf.nn.tanh, name = 'AttWDense1')
            att_w = tf.layers.dense(att_w, 1, name = 'AttWDense2') # (bs, tl, 1)
            return self.output(text, mask, att_w)

    def dim_att(self, text, mask, user = None, name = 'SelfDimAtt'):
        #(bs, tl, k)
        with tf.variable_scope(name):
            if user is not None:
                user = tf.expand_dims(user, 1) #(bs, 1, k)
                uu = tf.tile(user, [1, text.shape[1], 1])
                t = tf.concat([text, uu], -1) # (bs, tl, 2 * k)
            else:
                t = text
            att_w = tf.layers.dense(t, self.args.dim_k, activation = tf.nn.tanh, name = 'AttWDense1')
            att_w = tf.layers.dense(att_w, self.args.dim_k, name = 'AttWDense2')
            return self.output(text, mask, att_w)

    # weight = [word[-1], word[0], word[1]] @ W @ user
    def cnn_att(self):
        pass

class AnsAtt(BasicPair, AttLayers):
    tc = dict(BasicPair.tc)
    tc.update(AttLayers.tc)
    #tc['att_mode'] = ['dim', 'norm']
    tc['att_mode'] = ['norm']
    tc['user_att'] = [False, True]

    def rep(self, ans, user):
        a, a_mask = self.text_embs(ans)
        if self.args.user_att:
            u = self.user_emb(user)
        else:
            u = None
        if self.args.att_mode == 'dim':
            o = self.dim_att(a, a_mask, u)
        else:
            o = self.MLP_att(a, a_mask, u)
        return o

class QueAtt(BasicPair, AttLayers):
    tc = dict(BasicPair.tc)
    tc.update(AttLayers.tc)

    def get_question_rep(self, q):
        return self.text_embs(q, name = 'QueTextEmb')

    def rep(self, ans, user):
        q, q_mask = self.question_rep
        a, a_mask = self.text_embs(ans, name = 'AnsTextEmb') #(bs, al, k)
        u = self.user_emb(user)
        q = self.self_dim_att(q, q_mask)

        o = self.dim_att(a, a_mask, q + u) #(bs, k)
        return o

class QueDimAtt(BasicPair, AttLayers):
    tc = dict(BasicPair.tc)
    #tc['top_k'] = [5]
    #tc['text_mode'] = ['mask_max']
    #tc['text_mode'] = ['mask_top_k_mean']
    #tc['add_user'] = [True]
    #tc['add_ans'] = [False]
    #tc['lr'] = [1e-5]

    def get_question_rep(self, q):
        q, q_mask = self.text_embs(q)
        #q = self.text_rep(q, q_mask)
        q = self.mask_mean(q, q_mask)
        #q = self.lstm(q)
        return q

    def rep(self, ans, user):
        q = self.question_rep
        a, a_mask = self.text_embs(ans) #(bs, al, k)
        u = self.user_emb(user)
        o = self.dim_att(a, a_mask, q + u) #(bs, k)
        o = o + q + u
        return o

class SimpleAtt(BasicPair, AttLayers):
    tc = dict(BasicPair.tc)

    def rep(self, a, user):
        q, q_mask = self.question_rep
        a, a_mask = self.text_embs(a) # (bs, al, k)
        #ans = self.top_k(ans, self.args.top_k)
        user = self.user_emb(user)

        q = self.dot_w_att(q, q_mask, user, name = 'AttQ')
        a = self.dot_w_att(a, a_mask, user, name = 'AttA')

        return self.pair_mul_add([q, a, user])

class AQAtt(BasicPair, AttLayers):
    tc = dict(BasicPair.tc)
    tc['top_k'] = [5]
    #tc['text_mode'] = ['mask_max']
    tc['text_mode'] = ['mask_top_k_mean']
    #tc['add_user'] = [True]
    #tc['lr'] = [1e-5]

    def rep(self, a, user):
        q, q_mask = self.question_rep
        a, a_mask = self.text_embs(a) # (bs, al, k)
        #ans = self.top_k(ans, self.args.top_k)
        ar = self.AQ_dot_att(q, q_mask, a, a_mask)
        ab = self.text_rep(ar + a, a_mask)

        ub = self.user_emb(user, name = 'ub')
        return tf.concat([ab, ub], -1)


def main():
    print('hello world, Att.py')

if __name__ == '__main__':
    main()

