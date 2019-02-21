from cqa.baselinee.b1 import *


class Temp(B1):
    def define_inputs(self):
        super(Temp, self).define_inputs()
        self.train_inp = [self.inp_q0, self.inp_a0, self.inp_u0,
                          self.inp_q1, self.inp_a1, self.inp_u1]
        self.pred_inp = [self.inp_q0, self.inp_a0, self.inp_u0]

    def dot(self, a, b, axis=-1):
        return tf.reduce_sum(a * b, axis)

    def get_fd(self, q0, a0, u0, q1=None, a1=None, u1=None):
        if not (q1 is None or a1 is None or u1 is None):
            return dict(zip(self.train_inp, [q0, a0, u0, q1, a1, u1]))
        else:
            return dict(zip(self.pred_inp, [q0, a0, u0]))


class AAAI15(Temp):
    def rep(self, ques, answ, user):
        with tf.variable_scope('S-matrix'):
            q_emb, q_mask = self.text_emb(ques)
            a_emb, a_mask = self.text_emb(answ)
            q_emb = tf.tile(tf.expand_dims(q_emb, axis=0), [tf.shape(a_emb)[0], 1, 1])
            q_emb_l2 = tf.nn.l2_normalize(q_emb, axis=-1)  # (bs, nq, wd)
            a_emb_l2 = tf.nn.l2_normalize(a_emb, axis=-1)
            a_emb_l2 = tf.transpose(a_emb_l2, perm=[0, 2, 1])  # (bs, wd, na)
            m = tf.matmul(q_emb_l2, a_emb_l2)  # (bs, nq, na)
            m = tf.expand_dims(m, -1)
        with tf.variable_scope('CNN'):
            m = tf.layers.conv2d(m, 20, 5, kernel_initializer=self.x_init)
            m = tf.layers.max_pooling2d(m, (2, 3), (2, 3))
            m = tf.layers.conv2d(m, 50, 5, kernel_initializer=self.x_init)
            m = tf.layers.max_pooling2d(m, (2, 3), (2, 3))
            o = tf.layers.flatten(m)
        return o


class AAAI17(Temp):
    def rep(self, ques, answ, user):
        ques = tf.expand_dims(ques, axis=0)
        q, q_mask = self.text_emb(ques)
        a, a_mask = self.text_emb(answ)
        q_rep = self.text_rep(q, q_mask)
        a_rep = self.text_rep(a, a_mask)
        u_rep = self.user_emb(user)
        M = tf.get_variable('q_m', shape=(self.w_dim, self.w_dim), initializer=self.x_init)
        q_rep = tf.matmul(q_rep, M)
        # q_rep = tf.layers.dense(q_rep, self.w_dim, tf.nn.relu, kernel_initializer=self.x_init)
        qa = self.dot(q_rep, a_rep)
        qu = self.dot(q_rep, u_rep)
        return qa * qu

    def top_layers(self, t, name='TopLayer'):
        return t


class IJCAI15(Temp):
    def cnn_topk(self, t, maxlen):
        # print(t, maxlen); input()
        with tf.variable_scope('CNN_tk'):
            ns = 4
            t = tf.layers.conv1d(t, filters=10, kernel_size=3, kernel_initializer=self.x_init)
            t = self.top_k(t, ns + (maxlen - ns) // 2)
            t = tf.layers.conv1d(t, filters=5, kernel_size=3, kernel_initializer=self.x_init)
            t = self.top_k(t, ns)
            o = tf.layers.flatten(t)
        return o

    # def get_question_rep(self, q):
    #     q_emb, q_mask = self.text_emb(q)
    #     return self.cnn_topk(q_emb, q_maxlen), q_mask

    def rep(self, ques, answ, user):
        with tf.variable_scope('match'):
            ns, r = 20, 5
            M = tf.get_variable(name='M', shape=(ns, ns, r), initializer=self.x_init)
            V = tf.get_variable(name='V', shape=(2 * ns, r), initializer=self.x_init)
            b = tf.get_variable(name='b', shape=(1, r), initializer=self.x_init)
        q, q_mask = self.question_rep  # (bs, ns)
        answ, a_mask = self.text_embs(answ)  # (bs, al, k)
        answ = self.cnn_topk(answ, a_maxlen)  # (bs, ns)
        qm = tf.tensordot(q, M, axes=[1, 0])  # (bs, ns, r)
        a = tf.expand_dims(answ, 1)  # (bs, 1, ns)
        qma = tf.matmul(a, qm)  # (bs, 1, r)
        qma = tf.reshape(qma, (-1, r))  # (bs, r)
        qa = tf.concat([q, answ], -1)  # (bs, ns * 2)
        qav = tf.matmul(qa, V)  # (bs, r)
        return tf.nn.relu(qma + qav + b)
