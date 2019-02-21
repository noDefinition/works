import tensorflow as tf

from . import BasicPair


class SimQA(BasicPair):
    tc = dict(BasicPair.tc)
    tc['top_k'] = [5]
    # tc['text_mode'] = ['top_k_sum', 'top_k_mean', 'sum', 'mean', 'mask_sum', 'mask_mean']
    # tc['text_mode'] = ['mask_mean']
    tc['text_mode'] = ['mask_top_k_sum']
    tc['add_user'] = [True]
    
    def get_question_rep(self, q):
        q, q_mask = self.text_embs(q)
        return self.mask_mean(q, q_mask)
    
    def match(self, q, a, name='match'):
        with tf.variable_scope(name):
            m = tf.expand_dims(self.dot(q, a), -1)
        return m
    
    def rep(self, a, user):
        a, a_mask = self.text_embs(a)
        am = self.mask_mean(a, a_mask)
        
        q = self.question_rep
        m = self.match(q, am)
        
        o = self.text_rep(a, a_mask)
        o = tf.concat([o, m], -1)
        
        if self.args.add_user:
            user = self.user_emb(user)
            o = tf.concat([o, user], -1)
        return o


class KMatch(BasicPair):
    tc = dict(BasicPair.tc)
    tc['top_k'] = [5]
    tc['text_mode'] = ['mask_top_k_mean']
    
    def get_question_rep(self, q):
        q, q_mask = self.text_embs(q)
        return self.text_rep(q, q_mask)
        # return self.mask_mean(q, q_mask)
    
    def rep(self, a, user):
        q = self.question_rep  # (bs, k)
        a, a_mask = self.text_embs(a)  # (bs, al, k)
        
        with tf.variable_scope('match'):
            w = tf.get_variable(
                name='w',
                shape=(self.args.dim_k, self.args.dim_k, self.args.dim_k),
                regularizer=lambda w: self.args.l2_ue * tf.nn.l2_loss(w),
                initializer=self.get_initializer(),
            )
            qw = tf.tensordot(q, w, [1, 0])  # (bs, k, k)
            a = tf.matmul(a, qw)  # (bs, al, k)
            a = self.text_rep(a, a_mask)
        
        user = self.user_emb(user)
        return tf.concat([a, user], -1)


from .TopK import TopKSum
from .Att import AttLayers


class SimpleMatch(TopKSum, AttLayers):
    tc = dict(BasicPair.tc)
    tc['top_k'] = [5]
    tc['use_mlp'] = [False]
    tc['fc'] = [[]]
    
    def match(self, a, b, name):
        # (bs, k), (bs, k)
        with tf.variable_scope(name):
            aw = tf.layers.dense(
                a,
                self.args.dim_k,
                use_bias=False,
                kernel_initializer=self.get_initializer(),
            )
            o = self.dot(aw, b)
            o = tf.expand_dims(o, -1)
        return o
    
    def rep(self, a, user):
        q, q_mask = self.question_rep
        # a, a_mask = self.text_embs(a, name = 'AnsEmb') # (bs, al, k)
        a, a_mask = self.text_embs(a)  # (bs, al, k)
        a_rep = self.text_rep(a)
        user = self.user_emb(user)
        au_rep = self.dot_w_att(a, a_mask, user, name='AttA')
        
        match_a = self.match(a_rep, q, 'match_aq')
        match_u = self.match(user, q, 'match_uq')
        match_au = self.match(au_rep, q, 'match_auq')
        bias_a = tf.layers.dense(
            a_rep,
            1,
            use_bias=False,
            kernel_initializer=self.get_initializer(),
        )
        bias_u = tf.layers.dense(
            user,
            1,
            use_bias=False,
            kernel_initializer=self.get_initializer(),
        )
        bias_au = tf.layers.dense(
            au_rep,
            1,
            use_bias=False,
            kernel_initializer=self.get_initializer(),
        )
        
        select = tf.layers.dense(
            q,
            6,
            activation=tf.nn.softmax,
            kernel_initializer=self.get_initializer(),
        )
        o = tf.concat([match_a, match_u, match_au, bias_a, bias_u, bias_au], -1)
        o = self.dot(select, o)
        return o


class QAMul(BasicPair, AttLayers):
    tc = dict(BasicPair.tc)
    tc.update(AttLayers.tc)
    tc['mul_mode'] = ['dot', 'mul']
    
    def get_question_rep(self, q):
        q, q_mask = self.text_embs(q, name='QueTextEmb')
        q = self.self_dim_att(q, q_mask)
        return q
    
    def rep(self, ans, user):
        q = self.question_rep
        a, a_mask = self.text_embs(ans, name='AnsTextEmb')  # (bs, al, k)
        a = self.self_dim_att(a, a_mask)
        if self.args.mul_mode == 'dot':
            o = self.dot(a, q)
        else:
            o = a * q
        return o


def main():
    print('hello world, RepMatch.py')


if __name__ == '__main__':
    main()
