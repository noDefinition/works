import tensorflow as tf

from . import BasicPair
from . import a_maxlen, q_maxlen


class AAAI17(BasicPair):
    tc = dict(BasicPair.tc)
    tc['init_word'] = ['random']
    tc['train_word'] = ['train']

    def get_question_rep(self, q):
        return self.text_embs(q)

    def rep(self, ans, user):
        q, q_mask = self.question_rep
        # q = self.lstm(q)
        q = self.mask_mean(q, q_mask)
        ans, a_mask = self.text_embs(ans)  # (bs, al, k)
        # ans = self.lstm(ans)
        ans = self.mask_mean(ans, a_mask)
        user = self.user_emb(user)
        qw = tf.get_variable(
            name='qw',
            shape=(self.args.dim_k, self.args.dim_k),
        )
        q = tf.matmul(q, qw)
        qa = self.dot(q, ans)
        qu = self.dot(q, user)
        return qa * qu


class AAAI15(BasicPair):
    tc = dict(BasicPair.tc)

    def get_question_rep(self, q):
        return self.text_embs(q)

    def rep(self, a, user):
        with tf.variable_scope('S-matrix'):
            q, q_mask = self.question_rep  # (bs, ql, k)
            q = tf.expand_dims(q, 2)  # (bs, ql, 1, k)
            a, a_mask = self.text_embs(a)  # (bs, al, k)
            a = tf.expand_dims(a, 1)  # (bs, 1, al, k)
            # qa = tf.reduce_sum(q * a, -1) #(bs, ql, al)
            # qq = tf.sqrt(tf.reduce_sum(q * q, -1)) #(bs, ql, 1)
            # aa = tf.sqrt(tf.reduce_sum(a * a, -1)) #(bs, 1, al)
            # m = qa / (qq * aa + 1e-7)
            m = self.cos(q, a)
            m = tf.expand_dims(m, -1)
        with tf.variable_scope('CNN'):
            m = tf.layers.conv2d(m, 20, 5, kernel_initializer=self.get_initializer(), )
            m = tf.layers.max_pooling2d(m, (2, 3), (2, 3))
            m = tf.layers.conv2d(m, 50, 5, kernel_initializer=self.get_initializer(), )
            m = tf.layers.max_pooling2d(m, (2, 3), (2, 3))
            o = tf.layers.flatten(m)
        return o


class IJCAI15(BasicPair):
    tc = dict(BasicPair.tc)

    def get_question_rep(self, q):
        q_emb, q_mask = self.text_embs(q)
        return self.cnn_topk(q_emb, q_maxlen), q_mask

    def top_k(self, t, k):
        t = tf.transpose(t, [0, 2, 1])
        t = tf.nn.top_k(t, k).values
        t = tf.transpose(t, [0, 2, 1])
        return t

    def cnn_topk(self, t, maxlen):
        # print(t, maxlen); input()
        with tf.variable_scope('CNN_tk'):
            ns = 4
            t = tf.layers.conv1d(
                t,
                filters=10,
                kernel_size=3,
                kernel_initializer=self.get_initializer(),
            )
            t = self.top_k(t, ns + (maxlen - ns) // 2)
            t = tf.layers.conv1d(
                t,
                filters=5,
                kernel_size=3,
                kernel_initializer=self.get_initializer(),
            )
            t = self.top_k(t, ns)
            o = tf.layers.flatten(t)
        return o

    def rep(self, a, user):
        q, q_mask = self.question_rep  # (bs, ns)
        a, a_mask = self.text_embs(a)  # (bs, al, k)
        a = self.cnn_topk(a, a_maxlen)  # (bs, ns)
        with tf.variable_scope('match'):
            ns, r = 20, 5
            M = tf.get_variable(
                name='M',
                shape=(ns, ns, r),
                initializer=self.get_initializer(),
            )
            V = tf.get_variable(
                name='V',
                shape=(2 * ns, r),
                initializer=self.get_initializer(),
            )
            b = tf.get_variable(
                name='b',
                shape=(1, r),
                initializer=self.get_initializer(),
            )
            qm = tf.tensordot(q, M, axes=[1, 0])  # (bs, ns, r)
            a2 = tf.expand_dims(a, 1)  # (bs, 1, ns)
            qma = tf.matmul(a2, qm)  # (bs, 1, r)
            qma = tf.reshape(qma, (-1, r))  # (bs, r)
            qa = tf.concat([q, a], -1)  # (bs, ns * 2)
            qav = tf.matmul(qa, V)  # (bs, r)
            o = qma + qav + b
            o = tf.nn.relu(o)
            return o


def main():
    print('hello world, Baselines.py')


if __name__ == '__main__':
    main()
